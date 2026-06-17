"""Source separation via Demucs.

Demucs `htdemucs` separates a mix into 4 stems: drums, bass, vocals, other.
The optional `htdemucs_6s` adds piano + guitar but is slower.

We defensively drop the vocals stem (the input is declared instrumental) and
warn if the vocal energy is non-trivial.

The Demucs API at `demucs.apply.apply_model` is preferred over the CLI for
in-process use. Falls back to a degraded-but-functional analysis if Demucs is
not installed (single tonal stem = the original mix, no drums isolation).

**Why separation matters:**
- Drums are isolated → accurate onset detection and noise-channel routing.
- Bass/other are isolated → clean pitch transcription (no bass/melody interference).
- Better multi-voice part separation → more accurate AY channel assignment.

**Without separation (--no-separation flag):**
- Entire mix transcribed as one signal → blurred pitch, drums lost, poor part separation.
- Only viable for single clean instruments (solo synth, trumpet, etc.).
- For mixed music, measurably degrades quality. The flag is rarely useful in practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Stems:
    """Separated stems at the original sample rate (mono float32 each)."""

    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    vocals: np.ndarray
    sample_rate: int

    @property
    def has_vocals_warning(self) -> bool:
        rms_vocals = float(np.sqrt(np.mean(self.vocals**2) + 1e-12))
        rms_total = float(np.sqrt(np.mean((self.drums + self.bass + self.other) ** 2) + 1e-12))
        return rms_vocals > 0.1 * rms_total


def separate(audio_stereo: np.ndarray, sr: int, *, model_name: str = "htdemucs") -> Stems:
    """Run Demucs on a stereo (or mono-broadcast) waveform and return mono stems.

    `audio_stereo` is shape (channels, samples). If only 1 channel is provided,
    it is duplicated to stereo for Demucs.
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:  # pragma: no cover - optional dep path
        log.warning(
            "Demucs not available (%s); falling back to no-separation. "
            "Install with `pip install audio2ay[ml]` for proper stem splitting.",
            exc,
        )
        mono = audio_stereo.mean(axis=0).astype(np.float32)
        zero = np.zeros_like(mono)
        return Stems(drums=zero, bass=zero, other=mono, vocals=zero, sample_rate=sr)

    if audio_stereo.ndim == 1:
        audio_stereo = np.stack([audio_stereo, audio_stereo], axis=0)
    elif audio_stereo.shape[0] == 1:
        audio_stereo = np.concatenate([audio_stereo, audio_stereo], axis=0)

    model = get_model(model_name)
    model.eval()
    # Demucs expects (batch, channels, samples), float32, sr matching model.samplerate.
    target_sr = int(model.samplerate)
    if sr != target_sr:
        import librosa

        # Resample each channel.
        resampled = np.stack(
            [librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in audio_stereo],
            axis=0,
        )
        audio_stereo = resampled
        sr = target_sr

    wav = torch.from_numpy(audio_stereo).unsqueeze(0).float()
    with torch.no_grad():
        sources = apply_model(model, wav, split=True, overlap=0.25, progress=False)
    sources = sources.squeeze(0).cpu().numpy()  # (n_sources, ch, samples)

    # `model.sources` lists names in order.
    name_to_idx = {name: i for i, name in enumerate(model.sources)}

    def _mono(name: str) -> np.ndarray:
        if name in name_to_idx:
            return sources[name_to_idx[name]].mean(axis=0).astype(np.float32)
        return np.zeros(sources.shape[-1], dtype=np.float32)

    drums = _mono("drums")
    bass = _mono("bass")
    vocals = _mono("vocals")
    if "other" in name_to_idx:
        other = _mono("other")
    else:
        # 6-stem model splits "other" further; recombine guitar + piano + leftover.
        other_parts = [
            _mono(n) for n in name_to_idx if n not in {"drums", "bass", "vocals"}
        ]
        other = np.sum(other_parts, axis=0).astype(np.float32) if other_parts else np.zeros_like(bass)

    stems = Stems(drums=drums, bass=bass, other=other, vocals=vocals, sample_rate=sr)
    if stems.has_vocals_warning:
        log.warning(
            "Significant vocal energy detected in input — converter is tuned for instrumental "
            "music; results may be poor."
        )
    return stems
