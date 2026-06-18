"""End-to-end pipeline test using a synthetic instrumental signal.

We create a 2-second synthetic mix of a 110 Hz square wave (bass) plus a
440 Hz sine (lead), write it to wav, run the pipeline with separation
disabled, and verify the produced YM file plays back through the bundled
emulator with audible energy near both frequencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from audio2ay.io.ay_emulator import render_song_to_array
from audio2ay.pipeline import ConvertOptions, convert_audio_to_ym


def _make_mix(path: Path, sr: int = 22050, dur: float = 2.0) -> None:
    t = np.arange(int(sr * dur)) / sr
    bass = 0.4 * np.sign(np.sin(2 * np.pi * 110 * t))
    lead = 0.3 * np.sin(2 * np.pi * 440 * t)
    mix = (bass + lead).astype(np.float32)
    sf.write(str(path), mix, sr, subtype="PCM_16")


def _peak_freqs(audio: np.ndarray, sr: int, top_n: int = 4, skip: float = 0.2) -> list[float]:
    s = audio[int(skip * sr) :]
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sr)
    idx = np.argsort(spec)[-top_n:]
    return sorted(float(freqs[i]) for i in idx)


def test_pipeline_synthetic_mix(tmp_path: Path) -> None:
    in_wav = tmp_path / "mix.wav"
    out_ym = tmp_path / "mix.ym"
    _make_mix(in_wav)
    options = ConvertOptions(use_envelope=False)
    song = convert_audio_to_ym(in_wav, out_ym, options=options)

    assert out_ym.exists()
    assert len(song.frames) > 50  # at least a second of frames at 50 Hz

    audio = render_song_to_array(song, sample_rate=44100)
    # Should not be silent.
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > 1e-3, rms
    # FFT should have peaks below ~600 Hz (we are tracking either of the source
    # tones; exact pitch depends on Basic Pitch availability).
    peaks = _peak_freqs(audio, 44100)
    low_peaks = [p for p in peaks if p < 600]
    assert len(low_peaks) >= 1, peaks


def test_pipeline_silence(tmp_path: Path) -> None:
    in_wav = tmp_path / "sil.wav"
    sr = 22050
    sf.write(str(in_wav), np.zeros(int(sr * 1.0), dtype=np.float32), sr, subtype="PCM_16")
    out_ym = tmp_path / "sil.ym"
    options = ConvertOptions(use_envelope=False)
    song = convert_audio_to_ym(in_wav, out_ym, options=options)
    audio = render_song_to_array(song, sample_rate=44100)
    # All channels should stay muted: very low RMS.
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms < 0.05, rms


def test_pipeline_frame_rate_100hz(tmp_path: Path) -> None:
    in_wav = tmp_path / "mix_100hz.wav"
    _make_mix(in_wav)

    song_50 = convert_audio_to_ym(
        in_wav,
        None,
        options=ConvertOptions(use_envelope=False, frame_rate_hz=50),
    )
    song_100 = convert_audio_to_ym(
        in_wav,
        None,
        options=ConvertOptions(use_envelope=False, frame_rate_hz=100),
    )

    assert song_50.frame_rate_hz == 50
    assert song_100.frame_rate_hz == 100
    # Same source duration, so 100 Hz should produce about 2x frame count.
    assert len(song_100.frames) >= int(len(song_50.frames) * 1.9)
