"""Quick A/B spectral comparison of an original mix vs the AY render."""

from __future__ import annotations

import sys

import librosa
import numpy as np


def feat(y: np.ndarray, sr: int) -> dict[str, float]:
    S = np.abs(librosa.stft(y, n_fft=2048))
    return dict(
        centroid=float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        bandwidth=float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()),
        rolloff=float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean()),
        flatness=float(librosa.feature.spectral_flatness(S=S).mean()),
        rms=float(librosa.feature.rms(y=y).mean()),
        zcr=float(librosa.feature.zero_crossing_rate(y).mean()),
    )


def band_energy(y: np.ndarray, sr: int) -> dict[str, float]:
    S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = [(0, 120), (120, 500), (500, 2000), (2000, 6000), (6000, 11025)]
    tot = S.sum() + 1e-12
    out = {}
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out[f"{lo}-{hi}Hz"] = float(S[m].sum() / tot)
    return out


def main(orig_path: str, conv_path: str) -> None:
    orig, sr = librosa.load(orig_path, sr=22050, mono=True)
    conv, _ = librosa.load(conv_path, sr=22050, mono=True)
    n = min(len(orig), len(conv))
    orig, conv = orig[:n], conv[:n]

    fo, fc = feat(orig, sr), feat(conv, sr)
    print("feature        original      converted     delta%")
    for k in fo:
        d = (fc[k] - fo[k]) / (abs(fo[k]) + 1e-9) * 100
        print(f"{k:12s} {fo[k]:12.4f} {fc[k]:12.4f} {d:+8.1f}")

    print("\nband energy fraction   original  converted")
    bo, bc = band_energy(orig, sr), band_energy(conv, sr)
    for k in bo:
        print(f"  {k:14s} {bo[k]:8.3f}  {bc[k]:8.3f}")

    print(f"\nduration orig {len(orig)/sr:.1f}s conv {len(conv)/sr:.1f}s")
    to = float(librosa.beat.beat_track(y=orig, sr=sr)[0])
    tc = float(librosa.beat.beat_track(y=conv, sr=sr)[0])
    print(f"tempo orig {to:.1f} conv {tc:.1f}")

    # Onset density (note/event rate) in events per second.
    oo = librosa.onset.onset_detect(y=orig, sr=sr, units="time")
    oc = librosa.onset.onset_detect(y=conv, sr=sr, units="time")
    print(f"onset rate orig {len(oo)/(n/sr):.2f}/s conv {len(oc)/(n/sr):.2f}/s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
