"""Build R0..R13 from a per-frame ChannelAssignment + envelope + noise state."""

from __future__ import annotations

from .. import AY_CLOCK_HZ
from ..synth.hw_envelope import ENV_SHAPE_NONE, EnvelopeState
from ..synth.pitch_quantizer import freq_to_tp, tp_split
from ..synth.registers import RegisterFrame, mixer_byte
from ..synth.voice_scheduler import ChannelAssignment, VoiceState
from ..synth.volume_quantizer import amp_to_vol, envelope_bit, voice_gain


def _midi_to_tp(midi: float, clock_hz: int) -> int:
    freq = 440.0 * 2 ** ((midi - 69) / 12.0)
    return freq_to_tp(freq, clock_hz)


# Pitch above which the brightness tilt starts attenuating (MIDI 60 = middle C).
_BRIGHTNESS_PIVOT = 60.0


def _brightness_tilt(midi_pitch: float, brightness: float) -> float:
    """Per-voice gain that darkens high notes to tame square-wave harshness.

    ``brightness`` is a per-octave factor in ``(0, 1]``: 1.0 leaves everything
    untouched, while e.g. 0.85 multiplies a note one octave above middle C by
    0.85, two octaves by 0.85**2, etc. Notes at or below the pivot are unchanged.
    """
    if brightness >= 1.0 or midi_pitch <= _BRIGHTNESS_PIVOT:
        return 1.0
    octaves_above = (midi_pitch - _BRIGHTNESS_PIVOT) / 12.0
    return float(brightness ** octaves_above)


# Detune spread (in cents) used to fill idle channels with unison copies. The
# alternating ± pattern places the clones either side of the real note so two or
# three slightly mistuned square waves beat against each other — a classic
# chiptune trick that fattens an otherwise bare single tone.
_DETUNE_SPREAD: tuple[float, ...] = (1.0, -1.0, 2.0, -2.0)


def _enrich_voices(
    voices: list[VoiceState | None],
    detune_cents: float,
    volume_scale: float,
    reserve_for_noise: bool,
    extra_seeds: list[VoiceState] | None = None,
) -> list[VoiceState | None]:
    """Fill idle channels with slightly detuned copies of sounding notes.

    Only empty channels are touched, so genuinely polyphonic frames (all 3
    channels busy with real notes) are left untouched. When a drum hits this
    frame one idle channel is kept free so the noise gate still has somewhere
    clean to land.

    ``extra_seeds`` lets one chip enrich from notes that are sounding on the
    *other* chip. This is what keeps the second AY's channels from ever sitting
    idle: even when chip B has no notes of its own, its free channels double the
    chord playing on chip A (slightly detuned) for extra harmonic body.

    Idle channels cycle through the distinct sounding pitches (own notes first,
    then the other chip's), so a chord gets each of its notes doubled rather
    than three copies of a single note.
    """
    own_active = [v for v in voices if v is not None and v.velocity > 0.0]
    empties = [i for i, v in enumerate(voices) if v is None or v.velocity <= 0.0]
    if not empties:
        return voices
    if reserve_for_noise:
        # Keep the highest-index idle channel (noise routing prefers C→B→A).
        empties = empties[:-1]
        if not empties:
            return voices

    # Seed pool: this chip's own notes take priority, then the other chip's.
    # De-duplicate by (rounded) pitch so we double distinct chord tones.
    seeds: list[VoiceState] = []
    seen: set[float] = set()
    for v in sorted(own_active, key=lambda v: -v.velocity) + list(extra_seeds or []):
        key = round(v.midi_pitch, 1)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(v)
    if not seeds:
        return voices

    result = list(voices)
    for k, ch in enumerate(empties):
        src = seeds[k % len(seeds)]
        cents = _DETUNE_SPREAD[k % len(_DETUNE_SPREAD)] * detune_cents
        result[ch] = VoiceState(
            midi_pitch=src.midi_pitch + cents / 100.0,
            velocity=src.velocity * volume_scale,
            source=src.source,
            age_frames=src.age_frames,
        )
    return result


def build_frame(
    assignment: ChannelAssignment,
    envelope: EnvelopeState,
    *,
    clock_hz: int = AY_CLOCK_HZ,
    enrich_unison: bool = False,
    enrich_detune_cents: float = 9.0,
    enrich_volume_scale: float = 0.75,
    enrich_seeds: list[VoiceState] | None = None,
    master_gain: float = 1.0,
    brightness: float = 1.0,
) -> RegisterFrame:
    f = RegisterFrame()
    voices: list[VoiceState | None] = assignment.to_list()
    if enrich_unison:
        reserve = assignment.drum_period > 0 and assignment.drum_intensity > 0.0
        voices = _enrich_voices(
            voices,
            enrich_detune_cents,
            enrich_volume_scale,
            reserve,
            extra_seeds=enrich_seeds,
        )
    tone_on = [False, False, False]
    vols = [0, 0, 0]
    use_env = [False, False, False]

    for ch, v in enumerate(voices):
        if v is None or v.velocity <= 0.0:
            continue
        tp = _midi_to_tp(v.midi_pitch, clock_hz)
        fine, coarse = tp_split(tp)
        f[ch * 2] = fine
        f[ch * 2 + 1] = coarse
        tone_on[ch] = True
        if ch == 0 and envelope.enabled:
            # Channel A follows the hardware envelope: its 4-bit level is ignored.
            vols[ch] = 15
            use_env[ch] = True
        else:
            # Software amplitude envelope: the per-frame velocity already tracks
            # the source note's attack/decay; voice_gain only de-clicks onset.
            # master_gain folds in the song's macro-loudness; brightness tilts
            # high voices down so the square-wave render isn't over-bright.
            amp = v.velocity * voice_gain(v.age_frames) * master_gain
            amp *= _brightness_tilt(v.midi_pitch, brightness)
            vols[ch] = amp_to_vol(amp)

    # Drums → noise on a channel that isn't carrying the melody if possible.
    noise_on = [False, False, False]
    if assignment.drum_period > 0 and assignment.drum_intensity > 0.0:
        f[6] = assignment.drum_period & 0x1F
        # Prefer a free channel (no active tone), searching C → B → A so the
        # bass on channel A is masked last. If every channel has a tone, fall
        # back to the quietest non-bass channel.
        target = next((ch for ch in (2, 1, 0) if not tone_on[ch]), None)
        if target is None:
            target = 2 if vols[2] <= vols[1] else 1
        noise_on[target] = True
        # Tone and noise share one volume DAC per channel; make sure the gate is
        # at least as loud as the drum hit. The drum also rides the song's
        # macro-loudness so it doesn't punch through quiet passages.
        drum_vol = amp_to_vol(assignment.drum_intensity * master_gain)
        if vols[target] < drum_vol:
            vols[target] = drum_vol

    f[7] = mixer_byte(
        tone_on[0], tone_on[1], tone_on[2], noise_on[0], noise_on[1], noise_on[2]
    )
    f[8] = envelope_bit(vols[0], use_env[0])
    f[9] = envelope_bit(vols[1], use_env[1])
    f[10] = envelope_bit(vols[2], use_env[2])

    # Envelope period + shape latch.
    f[11] = envelope.period & 0xFF
    f[12] = (envelope.period >> 8) & 0xFF
    if envelope.shape_byte == ENV_SHAPE_NONE:
        f[13] = 0xFF  # treated by our emulator and conventional players as "no change"
    else:
        f[13] = envelope.shape_byte & 0x0F

    return f
