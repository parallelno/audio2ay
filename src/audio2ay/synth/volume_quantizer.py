"""Linear amplitude → 4-bit AY volume index.

The AY-3-8910 has a 4-bit logarithmic DAC per channel. Steps are roughly
`-3 dB` apart, with index 0 being silence. We use the well-documented
non-linear table from the YM2149 datasheet (which closely matches the
AY-3-8910), normalised so step 15 corresponds to a linear amplitude of 1.0.

`amp_to_vol` accepts a linear amplitude in [0, 1] and returns the index in
[0, 15] whose DAC level is closest. Because the DAC is logarithmic, this is
equivalent to mapping a perceptual loudness curve.
"""

from __future__ import annotations

# YM2149 16-step DAC, normalised to 1.0 at step 15.  Source: Hatari/Ayumi.
_DAC_LIN: tuple[float, ...] = (
    0.0,
    0.00999465,
    0.01445739,
    0.02105745,
    0.03070078,
    0.04554234,
    0.06729960,
    0.10331248,
    0.12658843,
    0.20498286,
    0.29221822,
    0.37283515,
    0.49253231,
    0.63532483,
    0.80558477,
    1.00000000,
)


def dac_level(vol: int) -> float:
    """Linear amplitude (0..1) of a given 4-bit AY volume index."""
    vol = max(0, min(15, int(vol)))
    return _DAC_LIN[vol]


def amp_to_vol(amp: float) -> int:
    """Map a linear amplitude in [0, 1] to the closest 4-bit AY volume index."""
    if amp <= 0.0:
        return 0
    if amp >= _DAC_LIN[-1]:
        return 15
    # Pick the level with smallest absolute distance.
    best_i = 0
    best_d = abs(_DAC_LIN[0] - amp)
    for i in range(1, 16):
        d = abs(_DAC_LIN[i] - amp)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def db_to_vol(db: float) -> int:
    """Map a dBFS value (≤0) to a 4-bit AY volume index."""
    if db <= -60:
        return 0
    amp = 10 ** (db / 20.0)
    return amp_to_vol(amp)


def envelope_bit(vol_index: int, use_envelope: bool) -> int:
    """Compose the per-channel volume register byte (low 4 bits = level, bit 4 = envelope)."""
    if use_envelope:
        return 0x10
    return max(0, min(15, int(vol_index))) & 0x0F


def voice_gain(age_frames: int) -> float:
    """Short attack ramp applied to a freshly-onset note to avoid click-on.

    The note's real attack/decay now comes from the per-frame source-energy
    follower (see :class:`audio2ay.analysis.amplitude.AmplitudeFollower`), so
    this only softens the first few frames: switching a square wave on at full
    amplitude in one 20 ms frame produces an audible click on the AY.
    A 3-frame ramp (60 ms at 50 Hz) gives a perceptibly smoother onset while
    remaining fast enough for staccato passages.
    """
    if age_frames <= 0:
        return 0.0
    if age_frames == 1:
        return 0.3
    if age_frames == 2:
        return 0.65
    return 1.0
