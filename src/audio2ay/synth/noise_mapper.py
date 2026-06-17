"""Map drum classes to AY noise periods + intensities.

The AY-3-8910 has a single shared 5-bit noise period and a separate noise gate
per channel via R7. The convention here is:

* hi-hat / cymbal → low noise period (1..4) → bright, fast pseudo-random
* snare           → mid noise period (8..14)
* kick            → high noise period (24..31) → slower, lower-pitched

Drum intensity drives the noise channel volume on whichever AY channel is
allocated to the noise gate (typically channel C).
"""

from __future__ import annotations

from ..analysis.drums import DrumClass
from ..analysis.timeline import FrameDrum

# Empirical defaults, tunable in Phase E.
_DRUM_PERIOD: dict[DrumClass, int] = {
    DrumClass.KICK: 28,
    DrumClass.SNARE: 11,
    DrumClass.HAT: 3,
}


def drum_to_noise(drum: FrameDrum | None) -> tuple[int, float]:
    """Return (noise_period [0..31], intensity [0..1]) for the given drum hit."""
    if drum is None or drum.intensity <= 0.01:
        return 0, 0.0
    period = _DRUM_PERIOD.get(drum.drum, 8)
    return period, drum.intensity
