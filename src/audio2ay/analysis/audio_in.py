"""Audio decoding + normalisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 22050  # Basic Pitch's required input rate; Demucs accepts arbitrary.


def load_audio_mono(path: str | Path, target_sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load an audio file, downmix to mono float32, optionally resample to ``target_sr``."""
    # We rely on librosa for resampling but read with soundfile for fewer codec
    # surprises. librosa.load is also fine; either path works.
    import librosa

    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32, copy=False), int(sr)


def load_audio_stereo(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio at native rate as float32 of shape (channels, samples)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T.astype(np.float32, copy=False), int(sr)
