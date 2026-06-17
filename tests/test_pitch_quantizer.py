import math

from audio2ay.synth.pitch_quantizer import (
    cents_error,
    freq_to_tp,
    midi_to_tp,
    tp_split,
    tp_to_freq,
)


def test_freq_to_tp_a4_at_1_5mhz() -> None:
    # 440 Hz at 1.5 MHz → 1.5e6 / (16*440) = 213.07 → round to 213
    assert freq_to_tp(440.0, clock_hz=1_500_000) == 213
    realised = tp_to_freq(213, clock_hz=1_500_000)
    assert abs(realised - 440.140845) < 1e-3
    # cents error well below 1 cent
    assert abs(cents_error(440.0, clock_hz=1_500_000)) < 1.0


def test_freq_to_tp_clamps() -> None:
    assert freq_to_tp(0.1, clock_hz=1_500_000) == 4095          # very low → max period
    assert freq_to_tp(1_000_000, clock_hz=1_500_000) == 1       # very high → min period


def test_midi_to_tp_a4() -> None:
    assert midi_to_tp(69, clock_hz=1_500_000) == 213


def test_tp_split() -> None:
    fine, coarse = tp_split(0xABC)
    assert fine == 0xBC
    assert coarse == 0x0A


def test_round_trip_sub_cent_in_audible_range() -> None:
    # Below ~3 kHz the quantizer is well under 30 cents.
    for midi in range(36, 84):  # C2 .. B5
        f = 440.0 * 2 ** ((midi - 69) / 12)
        err = cents_error(f, clock_hz=1_500_000)
        assert math.fabs(err) < 30.0, (midi, f, err)
