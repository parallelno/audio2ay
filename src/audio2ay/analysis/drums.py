"""Drum onset detection + crude classification (kick / snare / hat).

Uses librosa's onset_strength + peak picking. Each onset is classified by the
short-time spectral centroid + low-band energy ratio:

* low energy dominant + low centroid → kick
* mid centroid + broad spectrum     → snare
* high centroid                     → hi-hat / cymbal
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DrumClass(Enum):
    KICK = "kick"
    SNARE = "snare"
    HAT = "hat"


@dataclass
class DrumHit:
    time_sec: float
    drum: DrumClass
    intensity: float  # 0..1


def detect_drums(audio: np.ndarray, sr: int) -> list[DrumHit]:
    if audio.size == 0:
        return []
    import librosa

    hop = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    if onset_env.max() <= 1e-6:
        return []
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False
    )
    if len(onset_frames) == 0:
        return []
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    # Spectral analysis on short windows around each onset.
    n_fft = 2048
    win = n_fft
    hits: list[DrumHit] = []
    max_intensity = float(onset_env.max())
    for t, f in zip(onset_times, onset_frames):
        i0 = max(0, int(t * sr) - win // 4)
        i1 = min(len(audio), i0 + win)
        seg = audio[i0:i1]
        if len(seg) < 256:
            continue
        # Zero-pad if needed.
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        e_low = float(np.sum(spec[freqs < 200] ** 2))
        e_mid = float(np.sum(spec[(freqs >= 200) & (freqs < 2000)] ** 2))
        e_high = float(np.sum(spec[freqs >= 2000] ** 2))
        total = e_low + e_mid + e_high + 1e-12
        # Spectral centroid (Hz).
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))

        if e_low / total > 0.55 and centroid < 250:
            cls = DrumClass.KICK
        elif e_high / total > 0.5 and centroid > 4000:
            cls = DrumClass.HAT
        else:
            cls = DrumClass.SNARE

        intensity = float(min(1.0, onset_env[f] / max_intensity))
        hits.append(DrumHit(time_sec=float(t), drum=cls, intensity=intensity))
    return hits
