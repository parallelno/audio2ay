"""End-to-end pipeline: instrumental audio → YM5 register stream."""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import AY_CLOCK_HZ, FRAME_RATE_HZ
from .analysis.amplitude import AmplitudeFollower, loudness_envelope
from .analysis.audio_in import load_audio_stereo
from .analysis.drums import detect_drums
from .analysis.multipitch import transcribe
from .analysis.separation import separate
from .analysis.timeline import build_timeline
from .io.ym_writer import YmSong, write_ym5
from .synth.frame_builder import build_frame
from .synth.hw_envelope import EnvelopeController
from .synth.noise_mapper import drum_to_noise
from .synth.registers import RegisterFrame
from .synth.voice_scheduler import DualVoiceScheduler, VoiceScheduler

log = logging.getLogger(__name__)


@dataclass
class ConvertOptions:
    clock_hz: int = AY_CLOCK_HZ
    frame_rate_hz: int = FRAME_RATE_HZ
    # Run the hardware envelope at audio frequencies so its sawtooth/triangle
    # shape adds harmonic richness to the bass channel (the "fast-envelope bass"
    # trick).  Software amplitude envelopes (volume_quantizer.voice_gain) give
    # better macro-dynamics on their own, so this is opt-in via --envelope.
    use_envelope: bool = False
    demucs_model: str = "htdemucs"

    # Fill idle channels with slightly detuned unison copies of the active note
    # so sparse (mono/duo) material uses all 3 channels for a fuller tone.
    enrich_unison: bool = True
    enrich_detune_cents: float = 9.0
    # Loudness of detuned unison copies relative to a full-scale real voice.
    # Applies to idle channels on chip A (single-chip) and all of chip B
    # (dual-chip). At 0.75 two copies at that level produce more total power
    # than the one real melody note, which drowns the melody in a background hum.
    # 0.4 keeps copies audible for harmonic richness while leaving the real
    # note clearly louder. 1.0 = as loud as a real voice, 0.0 = silent.
    enrich_volume: float = 0.4
    # Demucs can hallucinate faint percussion from tonal onsets. Drop the drum
    # track when the drums stem RMS is below this fraction of the mix RMS.
    drum_energy_floor: float = 0.06
    # Drop phantom drum hits when the drums stem is too tonal (leaked piano /
    # keys rather than real percussion). Tests the 90th-percentile of spectral
    # flatness across active frames: real drums spike high at hit moments so
    # the top decile stays elevated even when most frames are low (tonal bleed);
    # a pure tonal stem stays flat across all frames. Threshold 0.04 reliably
    # separates piano-only stems (p90 ≈ 0.02) from real drum stems (p90 > 0.05).
    # Set to 0.0 to disable.
    drum_tonality_floor: float = 0.04
    # Emit a second AY chip (TurboSound-style): 6 tone channels total. The extra
    # polyphony goes on chip B; chip A is unchanged. Optional, off by default.
    dual_chip: bool = False
    # Track the original mix's overall loudness so the render keeps the song's
    # macro-dynamics (quiet intros / loud climaxes) instead of sitting at a flat,
    # too-loud level. loudness_floor keeps quiet passages audible.
    match_loudness: bool = True
    loudness_floor: float = 0.4
    # AY square waves are harsher/brighter than most acoustic sources. brightness
    # < 1.0 gently attenuates high voices (per octave above middle C) to pull the
    # spectral centroid back down toward the original; 1.0 disables the tilt.
    brightness: float = 0.85
    # Downward expander gate for the amplitude-follower: a note whose onset
    # energy is below (stem_peak * abs_onset_gate) has its velocity attenuated
    # by sqrt(onset / threshold). Suppresses quiet residual piano resonance
    # without affecting loudly-struck notes. 0.0 = disable.
    abs_onset_gate: float = 0.06


def convert_audio_to_ym(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    options: ConvertOptions | None = None,
) -> YmSong:
    """Convert an audio file to an uncompressed YM5 file. Returns the in-memory song.

    When ``out_path`` is ``None`` the YM file is not written; the in-memory
    :class:`YmSong` is just returned (useful for rendering straight to WAV).
    """
    opts = options or ConvertOptions()
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path is not None else None
    _t0 = _time.perf_counter()

    def _tick(label: str, t_prev: float) -> float:
        t = _time.perf_counter()
        log.info("  [%.1f s] %s", t - t_prev, label)
        return t

    log.info("Loading %s", in_path)
    audio, sr = load_audio_stereo(in_path)
    duration_sec = audio.shape[-1] / sr
    _t = _tick(f"load audio  ({duration_sec:.1f}s source, sr={sr})", _t0)

    # ------------------------------------------------------------ separation
    log.info("Running source separation (Demucs %s)", opts.demucs_model)
    stems = separate(audio, sr, model_name=opts.demucs_model)
    _t = _tick("separate (Demucs)", _t)

    # ------------------------------------------------------------ analysis
    log.info("Transcribing bass")
    bass_events = transcribe(stems.bass, stems.sample_rate, min_freq_hz=30.0, max_freq_hz=350.0)
    _t = _tick(f"transcribe bass  ({len(bass_events)} notes)", _t)
    log.info("Transcribing other")
    other_events = transcribe(stems.other, stems.sample_rate, min_freq_hz=80.0, max_freq_hz=4000.0)
    _t = _tick(f"transcribe other  ({len(other_events)} notes)", _t)
    log.info("Detecting drums")
    drum_hits = detect_drums(stems.drums, stems.sample_rate)
    _t = _tick(f"detect drums  ({len(drum_hits)} hits)", _t)
    # Demucs leaks tonal onsets into the drums stem on percussion-free material,
    # producing phantom snares. Drop the drum track when the drums stem is
    # negligible relative to the full mix.
    if drum_hits:
        drums_rms = float(np.sqrt(np.mean(stems.drums.astype(np.float32) ** 2) + 1e-12))
        mix = stems.drums + stems.bass + stems.other
        mix_rms = float(np.sqrt(np.mean(mix.astype(np.float32) ** 2) + 1e-12))
        rel = drums_rms / mix_rms
        if rel < opts.drum_energy_floor:
            log.info(
                "Drum stem negligible (%.1f%% of mix); dropping %d phantom hits",
                rel * 100.0,
                len(drum_hits),
            )
            drum_hits = []
    # Second gate: if the drums stem is predominantly tonal (piano/keys leakage)
    # rather than broadband percussion, treat all detected hits as phantom.
    #
    # Key insight: real drum stems have LOW flatness most of the time (tonal
    # bleed between hits) but spike HIGH at actual hit frames. A pure tonal
    # stem (piano resonance) stays low across ALL frames. So we test the
    # 90th-percentile of flatness values rather than the mean: real drums lift
    # the top decile well above the threshold even when the mean is low.
    if drum_hits and opts.drum_tonality_floor > 0.0:
        import librosa as _librosa

        drums_audio = stems.drums.astype(np.float32)
        hop = 512
        sf = _librosa.feature.spectral_flatness(y=drums_audio, hop_length=hop)[0]
        rms_frames = _librosa.feature.rms(y=drums_audio, hop_length=hop)[0]
        # Restrict to active frames so the silence floor doesn't inflate the
        # percentile with near-zero flatness values.
        above_floor = rms_frames > float(np.percentile(rms_frames, 10))
        sf_active = sf[above_floor] if above_floor.any() else sf
        peak_flatness = float(np.percentile(sf_active, 90))
        if peak_flatness < opts.drum_tonality_floor:
            log.info(
                "Drum stem is too tonal (flatness p90=%.4f < %.4f); "
                "dropping %d phantom hits",
                peak_flatness,
                opts.drum_tonality_floor,
                len(drum_hits),
            )
            drum_hits = []
        else:
            log.debug("Drum stem tonality OK (flatness p90=%.4f)", peak_flatness)
    log.info(
        "Bass notes=%d other notes=%d drum hits=%d",
        len(bass_events),
        len(other_events),
        len(drum_hits),
    )

    # --------------------------------------------------------- timeline (50/100 Hz)
    # Amplitude followers give each note real attack/decay by tracking the
    # source stem's energy around its pitch (instead of a flat onset velocity).
    bass_follower = AmplitudeFollower(
        stems.bass, stems.sample_rate, frame_rate_hz=opts.frame_rate_hz
    )
    other_follower = AmplitudeFollower(
        stems.other, stems.sample_rate, frame_rate_hz=opts.frame_rate_hz
    )

    timeline = build_timeline(
        bass_events,
        other_events,
        drum_hits,
        duration_sec=duration_sec,
        frame_rate_hz=opts.frame_rate_hz,
        bass_follower=bass_follower,
        other_follower=other_follower,
        abs_onset_gate=opts.abs_onset_gate,
    )
    _t = _tick(f"build timeline  ({len(timeline)} frames)", _t)

    # Log per-frame polyphony distribution so we can tell whether limiting
    # bass to one channel is a real constraint or a theoretical one.
    if log.isEnabledFor(logging.INFO):
        from collections import Counter
        bass_counts = Counter(len(f.bass_notes) for f in timeline)
        other_counts = Counter(len(f.other_notes) for f in timeline)
        n = len(timeline) or 1
        bass_dist = "  ".join(
            f"{k}:{v}({100*v//n}%)" for k, v in sorted(bass_counts.items())
        )
        other_dist = "  ".join(
            f"{k}:{v}({100*v//n}%)" for k, v in sorted(other_counts.items()) if k <= 8
        )
        log.info("Bass notes/frame distribution:  %s", bass_dist)
        log.info("Other notes/frame distribution: %s", other_dist)

    # --------------------------------------------------------- synth
    env_ctrl = EnvelopeController(clock_hz=opts.clock_hz)

    # Master loudness contour (one gain per frame) from the full mix, so the
    # render breathes with the original instead of running at a flat level.
    if opts.match_loudness:
        mix_mono = audio.mean(axis=0).astype(np.float32) if audio.ndim > 1 else audio
        master = loudness_envelope(
            mix_mono,
            sr,
            frame_rate_hz=opts.frame_rate_hz,
            n_frames=len(timeline),
            floor=opts.loudness_floor,
        )
    else:
        master = np.ones(len(timeline), dtype=np.float32)

    def _make_frame(
        assignment,
        env_ctrl_local: EnvelopeController,
        drum,
        enrich_seeds=None,
        enrich_volume_scale: float = 0.75,
        master_gain: float = 1.0,
    ) -> RegisterFrame:
        # Drum → noise channel state (chip A only).
        noise_period, noise_intensity = drum_to_noise(drum)
        assignment.drum_period = noise_period
        assignment.drum_intensity = noise_intensity
        envelope = (
            env_ctrl_local.step(assignment.a)
            if opts.use_envelope
            else env_ctrl_local.step(None)
        )
        return build_frame(
            assignment,
            envelope,
            clock_hz=opts.clock_hz,
            enrich_unison=opts.enrich_unison,
            enrich_detune_cents=opts.enrich_detune_cents,
            enrich_volume_scale=enrich_volume_scale,
            enrich_seeds=enrich_seeds,
            master_gain=master_gain,
            brightness=opts.brightness,
        )

    frames: list[RegisterFrame] = []
    frames_b: list[RegisterFrame] | None = None

    if opts.dual_chip:
        scheduler = DualVoiceScheduler()
        env_ctrl_b = EnvelopeController(clock_hz=opts.clock_hz)
        frames_b = []
        for fi, tf in enumerate(timeline):
            assign_a, assign_b = scheduler.assign(tf)
            frames.append(_make_frame(assign_a, env_ctrl, tf.drum, master_gain=master[fi]))
            seeds_b = [v for v in assign_a.to_list() if v is not None and v.velocity > 0.0]
            frames_b.append(
                _make_frame(
                    assign_b,
                    env_ctrl_b,
                    None,
                    enrich_seeds=seeds_b,
                    enrich_volume_scale=opts.enrich_volume,
                    master_gain=master[fi],
                )
            )
    else:
        scheduler = VoiceScheduler()
        for fi, tf in enumerate(timeline):
            assignment = scheduler.assign(tf)
            frames.append(
                _make_frame(
                    assignment,
                    env_ctrl,
                    tf.drum,
                    enrich_volume_scale=opts.enrich_volume,
                    master_gain=master[fi],
                )
            )
    _t = _tick(f"synth frames  ({len(frames)} frames)", _t)

    song = YmSong(
        frames=frames,
        song_name=in_path.stem,
        author="audio2ay",
        comment=f"Generated from {in_path.name}",
        clock_hz=opts.clock_hz,
        frame_rate_hz=opts.frame_rate_hz,
        frames_b=frames_b,
    )
    if out_path is not None:
        write_ym5(out_path, song, interleaved=True)
        log.info(
            "Wrote %s (%d frames%s)",
            out_path,
            len(frames),
            ", dual-AY" if opts.dual_chip else "",
        )
    return song
