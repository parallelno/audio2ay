"""Round-trip smoke test: hand-author 50 frames of A4, render via the AY
emulator, FFT the output, verify the dominant peak is at 440 Hz."""


import numpy as np

from audio2ay.io.ay_emulator import render_song_to_array
from audio2ay.io.ym_writer import YmSong
from audio2ay.synth.pitch_quantizer import freq_to_tp, tp_split
from audio2ay.synth.registers import RegisterFrame, mixer_byte


def _make_a4_song(seconds: float = 1.0) -> YmSong:
    fine, coarse = tp_split(freq_to_tp(440.0, clock_hz=1_500_000))
    frames = []
    n = int(round(50 * seconds))
    for _ in range(n):
        f = RegisterFrame()
        f[0], f[1] = fine, coarse
        f[7] = mixer_byte(True, False, False, False, False, False)
        f[8] = 12
        f[13] = 0xFF
        frames.append(f)
    return YmSong(frames=frames, clock_hz=1_500_000, frame_rate_hz=50)


def test_smoke_a4_dominant_peak_440() -> None:
    song = _make_a4_song(1.0)
    sr = 44100
    audio = render_song_to_array(song, sample_rate=sr)
    # Skip the first 100 ms to avoid envelope/transient effects.
    skip = sr // 10
    spec = np.fft.rfft(audio[skip:] * np.hanning(len(audio) - skip))
    freqs = np.fft.rfftfreq(len(audio) - skip, d=1.0 / sr)
    mag = np.abs(spec)
    peak_freq = freqs[int(np.argmax(mag))]
    assert abs(peak_freq - 440.0) < 5.0, peak_freq


def test_smoke_silence_when_muted() -> None:
    # All bits high in mixer → tones and noise off; volumes 0 → output is silent.
    frames = []
    for _ in range(10):
        f = RegisterFrame()
        f[7] = 0x3F
        f[8] = f[9] = f[10] = 0
        f[13] = 0xFF
        frames.append(f)
    song = YmSong(frames=frames, clock_hz=1_500_000, frame_rate_hz=50)
    audio = render_song_to_array(song, sample_rate=44100)
    # With centred output and zero volume, RMS should be effectively zero.
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms < 1e-6, rms
