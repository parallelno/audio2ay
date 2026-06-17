"""Pitch ↔ AY tone-period quantizer.

The AY-3-8910 generates a square wave on each tone channel from a 12-bit
period register `TP` (1..4095). With master clock `f_clk` (Hz):

    f_out = f_clk / (16 * TP)

For Vector-06C this is f_clk = 1.5 MHz, so:

* TP =    1 →  93.75 kHz
* TP =  213 → 440.14 Hz   (A4 ≈ 0.14 cents sharp)
* TP = 4095 →  22.89 Hz

Quantization error is sub-cent in the bass register and grows with pitch:
near TP=10 a unit step is ~170 cents, so very high notes can be off by
tens of cents. The functions in this module map between Hz and TP and
report the resulting cent error.
"""

from __future__ import annotations

import math

from .. import AY_CLOCK_HZ

TP_MIN = 1
TP_MAX = 4095


def freq_to_tp(freq_hz: float, clock_hz: int = AY_CLOCK_HZ) -> int:
    """Quantize a frequency to the nearest valid 12-bit AY tone period."""
    if freq_hz <= 0:
        return TP_MAX
    raw = clock_hz / (16.0 * freq_hz)
    tp = int(round(raw))
    if tp < TP_MIN:
        tp = TP_MIN
    elif tp > TP_MAX:
        tp = TP_MAX
    return tp


def tp_to_freq(tp: int, clock_hz: int = AY_CLOCK_HZ) -> float:
    """Inverse: realised frequency for a given tone period."""
    tp = max(TP_MIN, min(TP_MAX, int(tp)))
    return clock_hz / (16.0 * tp)


def cents_error(freq_hz: float, clock_hz: int = AY_CLOCK_HZ) -> float:
    """Cent error after rounding `freq_hz` to the nearest AY tone period."""
    if freq_hz <= 0:
        return 0.0
    tp = freq_to_tp(freq_hz, clock_hz)
    realised = tp_to_freq(tp, clock_hz)
    return 1200.0 * math.log2(realised / freq_hz)


def midi_to_tp(midi_note: float, clock_hz: int = AY_CLOCK_HZ) -> int:
    """Quantize a (possibly fractional) MIDI note to an AY tone period."""
    return freq_to_tp(440.0 * 2 ** ((midi_note - 69) / 12.0), clock_hz)


def tp_split(tp: int) -> tuple[int, int]:
    """Split TP into the AY's coarse/fine register pair (R0/R1, R2/R3, R4/R5)."""
    tp = max(TP_MIN, min(TP_MAX, int(tp)))
    return tp & 0xFF, (tp >> 8) & 0x0F
