"""Per-note amplitude following.

Basic Pitch gives a single onset *velocity* per note. Holding that flat for the
note's whole duration is what makes the AY render sound like a sustained organ
with hard cut-offs. To give notes a real attack/decay we sample the source
stem's spectral energy *around the note's pitch*, per 50 Hz frame, and let the
AY volume track it.

``AmplitudeFollower`` computes one STFT of a stem (hop aligned to the frame
grid) and exposes :meth:`band`, the energy in a few harmonic bands around a
given pitch on a given frame. The timeline normalises each note against its own
onset energy, so a struck/plucked note decays toward silence while a bowed/held
note stays up — independent of absolute signal scale.
"""

from __future__ import annotations

import numpy as np


def loudness_envelope(
    audio: np.ndarray | None,
    sr: int,
    *,
    frame_rate_hz: int,
    n_frames: int,
    floor: float = 0.35,
    percentile: float = 95.0,
) -> np.ndarray:
    """Per-frame master-gain contour tracking the mix's overall loudness.

    The per-note :class:`AmplitudeFollower` normalises *each note* against its
    own onset energy, which deliberately discards the song's macro-dynamics: a
    note in a quiet intro and the same note in a loud climax end up at similar
    AY volumes. That flattening is what makes the render sit too loud with a
    compressed dynamic range. This returns one gain per 50 Hz frame, derived
    from the RMS of the full mix, so callers can scale every channel's volume
    and restore the quiet/loud shape of the original.

    Returns an array of length ``n_frames`` with values in ``[floor, 1.0]``.
    ``floor`` keeps quiet passages audible instead of dropping to silence.
    """
    out = np.full(int(n_frames), 1.0, dtype=np.float32)
    if audio is None or np.size(audio) == 0 or n_frames <= 0:
        return out

    y = np.asarray(audio, dtype=np.float32)
    hop = max(1, int(round(sr / frame_rate_hz)))
    win = hop  # one analysis window per frame, no overlap
    # Per-frame RMS over the whole mix.
    rms = np.empty(int(n_frames), dtype=np.float32)
    for i in range(int(n_frames)):
        a = i * hop
        b = min(a + win, y.shape[0])
        seg = y[a:b]
        rms[i] = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0

    ref = float(np.percentile(rms, percentile))
    if ref <= 1e-9:
        return out
    g = np.clip(rms / ref, 0.0, 1.0)
    # Soften with a perceptual (square-root) curve so the contour follows
    # loudness rather than raw power, then lift onto the [floor, 1] range.
    g = np.sqrt(g)
    f = float(np.clip(floor, 0.0, 1.0))
    return (f + (1.0 - f) * g).astype(np.float32)


class AmplitudeFollower:
    """Sample per-frame source energy around a note's pitch.

    Parameters
    ----------
    audio:
        Mono stem samples (float32). Empty/None yields a follower that always
        returns 0.0 (callers fall back to the flat onset velocity).
    sr:
        Sample rate of ``audio``.
    frame_rate_hz:
        Target frame grid (50 Hz). The STFT hop is ``sr / frame_rate_hz`` so
        STFT column ``i`` corresponds to timeline frame ``i``.
    """

    def __init__(
        self,
        audio: np.ndarray | None,
        sr: int,
        *,
        frame_rate_hz: int,
        n_fft: int = 2048,
        n_harmonics: int = 3,
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.n_harmonics = max(1, n_harmonics)
        self.hop = max(1, int(round(sr / frame_rate_hz)))
        if audio is None or np.size(audio) == 0:
            self._S = np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
        else:
            import librosa

            self._S = np.abs(
                librosa.stft(
                    np.asarray(audio, dtype=np.float32),
                    n_fft=n_fft,
                    hop_length=self.hop,
                )
            ).astype(np.float32)
        self._ncols = self._S.shape[1]
        self._df = sr / n_fft  # Hz per FFT bin
        self._nyquist = sr / 2.0

    @property
    def available(self) -> bool:
        """True if the follower carries real signal energy."""
        return self._ncols > 1

    def band(self, frame_index: int, midi_pitch: float) -> float:
        """Harmonic-band energy around ``midi_pitch`` on ``frame_index``.

        Sums the peak magnitude in a ±1-bin window around the fundamental and
        the first few harmonics (harmonics weighted ``1/h``). Returns 0.0 when
        out of range.
        """
        if midi_pitch <= 0:
            return 0.0
        c = min(max(int(frame_index), 0), self._ncols - 1)
        col = self._S[:, c]
        nb = col.shape[0]
        f0 = 440.0 * 2.0 ** ((midi_pitch - 69.0) / 12.0)
        total = 0.0
        for h in range(1, self.n_harmonics + 1):
            fh = f0 * h
            if fh >= self._nyquist:
                break
            center = fh / self._df
            lo = max(int(np.floor(center)) - 1, 0)
            hi = min(int(np.ceil(center)) + 1, nb - 1)
            if hi < lo:
                continue
            total += float(col[lo : hi + 1].max()) / h
        return total
