"""Hardware-envelope fast-bass for channel A.

The AY's single envelope generator produces a periodic waveform shape at audio
frequencies, giving the bass channel a richer harmonic profile (sawtooth-like)
compared to a plain square wave.

Technique: set the envelope period so its cycle frequency matches the bass
note's pitch.  Shape 0x08 (repeated decay ``\\|\\|``) produces a sawtooth-down
timbre.  Both the tone generator and the envelope run at the same frequency, so
the channel output is a square wave amplitude-modulated by the sawtooth — warm
and bass-like.

Period formula::

    EP = round(clock_hz / (256 * note_freq))

R13 (the shape/retrigger register) is written only on note onset to avoid the
audible click that a mid-cycle retrigger would produce.  R11+R12 (the period
registers) are updated every frame so pitch slides stay in tune.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..synth.voice_scheduler import VoiceState

ENV_SHAPE_NONE = 0xFF  # sentinel: do not write R13 this frame


@dataclass
class EnvelopeState:
    enabled: bool = False
    shape_byte: int = ENV_SHAPE_NONE   # 0xFF = no write this frame
    period: int = 0                    # 16-bit envelope period

    def reset(self) -> None:
        self.enabled = False
        self.shape_byte = ENV_SHAPE_NONE
        self.period = 0


class EnvelopeController:
    """Decide whether channel A should follow the hardware envelope this frame.

    The envelope period is computed each frame from the live MIDI pitch so the
    sawtooth cycle tracks pitch slides automatically.  R13 is written only on
    note onset (``age_frames == 1``) to avoid mid-note retrigger clicks.

    *shape* choices:

    * ``0x08`` — repeated sawtooth-down ``\\|\\|\\|`` (default, warmest bass)
    * ``0x0A`` — triangle ``\\/\\/\\/``
    * ``0x0C`` — repeated sawtooth-up ``/|/|/|``
    """

    def __init__(self, clock_hz: int, shape: int = 0x08) -> None:
        self.clock_hz = clock_hz
        self.shape = shape
        self._was_enabled = False

    def step(self, voice_a: VoiceState | None) -> EnvelopeState:
        """Return the envelope state to apply to channel A this frame."""
        if voice_a is None:
            self._was_enabled = False
            return EnvelopeState(enabled=False, shape_byte=ENV_SHAPE_NONE, period=0)

        freq = 440.0 * 2.0 ** ((voice_a.midi_pitch - 69.0) / 12.0)
        ep = max(1, min(0xFFFF, round(self.clock_hz / (256.0 * freq))))

        # Write R13 only on note onset to avoid the click a mid-cycle retrigger
        # would produce.  The voice scheduler resets age_frames to 1 whenever a
        # new note starts on channel A.
        is_onset = voice_a.age_frames == 1 or not self._was_enabled
        shape_byte = self.shape if is_onset else ENV_SHAPE_NONE

        self._was_enabled = True
        return EnvelopeState(enabled=True, shape_byte=shape_byte, period=ep)
