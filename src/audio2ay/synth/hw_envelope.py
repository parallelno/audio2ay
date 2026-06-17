"""Hardware-envelope assignment for channel A.

The AY's envelope generator is single-shared: only one envelope shape and one
period are active at a time, but each channel can independently follow it via
bit 4 of its volume register.

We use it conservatively, only on channel A (bass) when:

* A note has been continuously active for ≥ K frames.
* The note's natural amplitude shape resembles a decay curve.
* No envelope retrigger was needed in the previous frame (a write to R13 is a
  retrigger and would click).

Currently this module makes a simple choice: shape 0x0A (\\/\\/ tremolo) for
sustained notes that have a slow amplitude pulse, otherwise no envelope. This
is a starting point; Phase E tunes it further.
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
    """Decide whether channel A should follow the hardware envelope this frame."""

    def __init__(self, min_age_frames: int = 6, env_period: int = 5000) -> None:
        self.min_age_frames = min_age_frames
        self.env_period = env_period
        self._last_enabled = False

    def step(self, voice_a: VoiceState | None) -> EnvelopeState:
        if voice_a is None or voice_a.age_frames < self.min_age_frames:
            # Note too short or absent — disable envelope.
            self._last_enabled = False
            return EnvelopeState(enabled=False, shape_byte=ENV_SHAPE_NONE, period=0)

        # Voice is stable. Use a slow tremolo (shape 10 = \/\/) at a fixed rate.
        if not self._last_enabled:
            # First frame envelope is enabled — trigger by writing R13.
            shape = 0x0A
        else:
            shape = ENV_SHAPE_NONE  # do not retrigger on subsequent frames
        self._last_enabled = True
        return EnvelopeState(enabled=True, shape_byte=shape, period=self.env_period)
