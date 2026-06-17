"""Pure-Python AY-3-8910 / YM2149 emulator.

This is a numpy-vectorised emulator focused on **rendering YM register dumps
to PCM audio for validation listening**. It is not cycle-accurate; it models:

* 3 tone channels as 50%-duty square waves driven from 12-bit period regs.
* A 17-bit LFSR noise generator clocked at `clock / (16 * NP)`.
* An envelope generator with all 8 distinct shapes, 32 internal steps mapped
  to the 16-step log DAC.
* The mixer (R7) with active-low tone/noise enables.
* Per-channel 4-bit log volume (R8/R9/R10) with the bit-4 envelope follow.

For more accurate emulation, swap in `ayumi` (https://github.com/true-grue/ayumi)
via ctypes — the public interface here (`AYRenderer.render_song`) is the
same shape Ayumi exposes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..synth.volume_quantizer import _DAC_LIN
from .ym_writer import YmSong, read_ym5

# Per-shape lookup of the AY envelope. Each entry encodes
# (start_dir, alternates, hold_high_after_first_cycle, hold_low_after_first_cycle).
# Shapes 0..3 == 9 (decay then off); 4..7 == 15 (attack then off).
# Continuous shapes (CONT=1) are 8..15 with sub-variants determined by ALT/HOLD.
_DAC = np.asarray(_DAC_LIN, dtype=np.float32)


def _envelope_level(shape: int, step: int) -> int:
    """Return the 0..15 envelope level for the given shape at the given 0..31 step.

    `step` keeps incrementing across cycles; this function handles the
    "hold" / "alternate" semantics internally.
    """
    shape &= 0x0F
    # Normalise CONT=0 shapes to canonical equivalents.
    if shape < 8:
        shape = 9 if shape < 4 else 15

    if shape == 8:                     # \\\\  (decay, repeat)
        return 15 - (step & 0x1F) // 2
    if shape == 12:                    # ////  (attack, repeat)
        return (step & 0x1F) // 2
    if shape == 10:                    # \/\/
        cycle = step // 32
        s = step & 0x1F
        return (15 - s // 2) if (cycle & 1) == 0 else (s // 2)
    if shape == 14:                    # /\/\
        cycle = step // 32
        s = step & 0x1F
        return (s // 2) if (cycle & 1) == 0 else (15 - s // 2)
    if shape == 9:                     # \___  (decay, hold low)
        return 0 if step >= 32 else 15 - step // 2
    if shape == 15:                    # /___  (attack, hold low)
        return 0 if step >= 32 else step // 2
    if shape == 11:                    # \¯¯¯  (decay, hold high)
        return 15 if step >= 32 else 15 - step // 2
    if shape == 13:                    # /¯¯¯  (attack, hold high)
        return 15 if step >= 32 else step // 2
    return 0


# Pre-compute envelope curves. We only need 64 steps to capture the longest
# transient before any continuous shape becomes periodic.
_ENV_LUT_LEN = 64
_ENV_LUT = np.zeros((16, _ENV_LUT_LEN), dtype=np.int8)
for sh in range(16):
    for st in range(_ENV_LUT_LEN):
        _ENV_LUT[sh, st] = _envelope_level(sh, st)


class AYRenderer:
    """Stateful renderer that converts YM register frames into PCM samples."""

    def __init__(self, clock_hz: int, sample_rate: int = 44100, pulse_width: float = 0.7) -> None:
        self.clock_hz = int(clock_hz)
        self.sr = int(sample_rate)
        self.pulse_width = float(np.clip(pulse_width, 0.1, 0.9))  # Clamp to avoid degenerate waves
        # Phase accumulators (cycles, fractional).
        self.phase = np.zeros(3, dtype=np.float64)
        # Noise: 17-bit LFSR seeded to all-ones.
        self.lfsr = 0x1FFFF
        self.noise_out = 1  # current bit output by the LFSR (high/low square)
        self.noise_acc = 0.0
        # Envelope state.
        self.env_step = 0
        self.env_acc = 0.0
        self.env_shape = 0

    # ------------------------------------------------------------------ utils
    def _step_noise(self) -> int:
        # 17-bit Galois LFSR, taps at bits 0 and 3 (Ayumi/ST-Sound convention).
        # Be defensive: if corrupted runtime state ever makes `self.lfsr`
        # non-integral, recover to a valid seed instead of crashing mid-render.
        try:
            lfsr = int(self.lfsr)
        except (TypeError, ValueError):
            lfsr = 0x1FFFF
        bit = (lfsr ^ (lfsr >> 3)) & 1
        lfsr = ((lfsr >> 1) | (bit << 16)) & 0x1FFFF
        self.lfsr = lfsr
        self.noise_out = lfsr & 1
        return self.noise_out

    # ---------------------------------------------------------------- render
    def render_frame(self, regs: list[int], n_samples: int) -> np.ndarray:
        """Render `n_samples` of audio for the given 14-byte register snapshot.

        Returns a float32 array in roughly [-1, 1].
        """
        sr = self.sr
        clk = self.clock_hz

        # Coerce to plain Python ints so the bitwise ops below never trip on
        # numpy scalars (their `<<` semantics differ).
        regs = [int(r) & 0xFF for r in regs]

        # Period registers.
        tp = [
            ((regs[1] & 0x0F) << 8) | regs[0],
            ((regs[3] & 0x0F) << 8) | regs[2],
            ((regs[5] & 0x0F) << 8) | regs[4],
        ]
        np_period = regs[6] & 0x1F
        if np_period == 0:
            np_period = 1
        ep = (regs[12] << 8) | regs[11]
        if ep == 0:
            ep = 1
        mixer = regs[7]
        vols = [regs[8], regs[9], regs[10]]

        # New envelope shape latched whenever R13 is written. The YM dump format
        # represents "no write this frame" as 0xFF; if any other byte appears,
        # treat it as a shape latch (matches ST-Sound/Ayumi behaviour).
        shape_byte = regs[13]
        if shape_byte != 0xFF:
            new_shape = shape_byte & 0x0F
            # In real hardware *any* write to R13 retriggers; we mimic by
            # resetting on shape change.
            if new_shape != self.env_shape:
                self.env_shape = new_shape
                self.env_step = 0
                self.env_acc = 0.0

        # ------------------------------------------------------------ tones
        # Square waves at f = clk / (16 * tp). dphase per sample = f / sr.
        # We treat tp == 0 as silence (real hw produces a DC level — close enough).
        out = np.zeros(n_samples, dtype=np.float32)
        squares = []
        for ch in range(3):
            if tp[ch] == 0:
                squares.append(np.ones(n_samples, dtype=np.float32))
                continue
            f = clk / (16.0 * tp[ch])
            dphi = f / sr
            phases = self.phase[ch] + dphi * np.arange(n_samples, dtype=np.float64)
            self.phase[ch] = float((phases[-1] + dphi) % 1.0)
            phases = phases % 1.0
            # Variable duty-cycle pulse wave: high when phase < pulse_width.
            # pulse_width=0.5 → 50% square, 0.75 → 75% high (fewer harmonics).
            sq = (phases < self.pulse_width).astype(np.float32)
            squares.append(sq)

        # ------------------------------------------------------------ noise
        # Step the LFSR every (16 * np_period) clock ticks. In samples, that's
        # `samples_per_step = 16 * np_period * sr / clk`. For clk=1.5e6, sr=44100,
        # np=1 → ~0.47 samples/step; np=31 → ~14.6 samples/step.
        samples_per_step = 16.0 * np_period * sr / clk
        if samples_per_step < 1.0:
            # Faster than one step per sample — pre-toggle randomness, but most
            # information will be aliased. Approximate as white noise.
            noise = np.where(
                np.random.random(n_samples) < 0.5,
                np.float32(1.0),
                np.float32(0.0),
            ).astype(np.float32)
            # Keep LFSR moving roughly.
            for _ in range(min(n_samples, 64)):
                self._step_noise()
        else:
            noise = np.empty(n_samples, dtype=np.float32)
            acc = self.noise_acc
            cur = self.noise_out  # already 0 or 1; numpy will up-cast on store
            noise_step = self._step_noise  # bind once to skip attribute lookup
            for i in range(n_samples):
                acc += 1.0
                while acc >= samples_per_step:
                    acc -= samples_per_step
                    noise_step()
                    cur = self.noise_out
                noise[i] = cur
            self.noise_acc = acc

        # -------------------------------------------------------- envelope
        # 32 steps per cycle, step rate = clk / (256 * ep). Samples per step:
        env_samples_per_step = 256.0 * ep * sr / clk
        if env_samples_per_step < 1.0:
            env_samples_per_step = 1.0
        env_levels = np.empty(n_samples, dtype=np.float32)
        try:
            acc = float(self.env_acc)
        except (TypeError, ValueError):
            acc = 0.0
        try:
            env_step = int(self.env_step)
        except (TypeError, ValueError):
            env_step = 0
        env_step = max(0, env_step)
        try:
            sh = int(self.env_shape) & 0x0F
        except (TypeError, ValueError):
            sh = 0
        for i in range(n_samples):
            # Guard against rare state corruption during long dual-chip runs.
            if not isinstance(acc, (int, float, np.floating)):
                try:
                    acc = float(acc)
                except (TypeError, ValueError):
                    acc = 0.0
            if not isinstance(env_step, (int, np.integer)):
                try:
                    env_step = int(env_step)
                except (TypeError, ValueError):
                    env_step = 0
            env_step = max(0, int(env_step))
            acc += 1.0
            while acc >= env_samples_per_step:
                acc -= env_samples_per_step
                if env_step + 1 < _ENV_LUT_LEN:
                    env_step += 1
                else:
                    # Wrap into the steady-state half of the LUT for continuous shapes.
                    env_step = 32 + ((env_step + 1 - 32) & 0x1F)
            env_levels[i] = _DAC[_ENV_LUT[sh, min(env_step, _ENV_LUT_LEN - 1)]]
        self.env_acc = float(acc)
        self.env_step = int(env_step)

        # --------------------------------------------------- mixer + volume
        # Mixer: bit i tone-off, bit i+3 noise-off (active-low). For each
        # channel the gate (0/1) is centred to (-0.5, +0.5) so the resulting
        # signal has no DC component (real AY hardware is AC-coupled).
        for ch in range(3):
            tone_off = (mixer >> ch) & 1
            noise_off = (mixer >> (ch + 3)) & 1
            tone_sig = squares[ch] if not tone_off else np.float32(1.0)
            noise_sig = noise if not noise_off else np.float32(1.0)
            gate = tone_sig * noise_sig  # both must be high to pass DAC level

            v = vols[ch]
            if v & 0x10:
                level = env_levels
            else:
                level = np.float32(_DAC[v & 0x0F])
            # Centre about zero so DC ≈ 0; keeps perceived loudness while
            # avoiding huge offsets that confuse FFT-based validation.
            out += (gate - np.float32(0.5)) * level

        # 3 channels mixed; scale to keep peak within [-1, 1].
        return (out / np.float32(1.5)).astype(np.float32)


def render_song_to_array(song: YmSong, sample_rate: int = 44100, pulse_width: float = 0.7) -> np.ndarray:
    """Render an in-memory YM song to a float32 mono numpy array.

    Dual-AY songs (``song.frames_b`` set) render each chip with its own stateful
    renderer and sum the two, scaled to stay within range.
    
    Args:
        song: YM song to render.
        sample_rate: Output sample rate in Hz.
        pulse_width: Pulse duty cycle (default 0.7; reduces square-wave harshness).
            0.5=square (harsh), 0.7=wider pulse (fewer harmonics, smoother).
    """
    samples_per_frame = sample_rate // song.frame_rate_hz
    n_total = len(song.frames) * samples_per_frame

    rend_a = AYRenderer(clock_hz=song.clock_hz, sample_rate=sample_rate, pulse_width=pulse_width)
    out = np.empty(n_total, dtype=np.float32)
    for i, frame in enumerate(song.frames):
        chunk = rend_a.render_frame(list(frame.regs), samples_per_frame)
        out[i * samples_per_frame : (i + 1) * samples_per_frame] = chunk

    if song.frames_b is not None:
        rend_b = AYRenderer(clock_hz=song.clock_hz, sample_rate=sample_rate, pulse_width=pulse_width)
        for i, frame in enumerate(song.frames_b):
            chunk = rend_b.render_frame(list(frame.regs), samples_per_frame)
            sl = slice(i * samples_per_frame, (i + 1) * samples_per_frame)
            out[sl] += chunk
        # Two chips summed; scale back toward unity headroom.
        out *= np.float32(0.6)

    return out


def render_ym_file(ym_path: str | Path, wav_path: str | Path, sample_rate: int = 44100) -> None:
    """Convenience: read a .ym, render via the bundled emulator, write a wav."""
    song = read_ym5(ym_path)
    audio = render_song_to_array(song, sample_rate=sample_rate)
    sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
