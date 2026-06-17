from pathlib import Path

from audio2ay.io.ym_writer import YmSong, read_ym5, write_ym5
from audio2ay.synth.pitch_quantizer import freq_to_tp, tp_split
from audio2ay.synth.registers import RegisterFrame, mixer_byte


def _make_song(tmp_path: Path) -> Path:
    frames: list[RegisterFrame] = []
    # 50 frames (= 1 second @ 50 Hz) of an A4 tone on channel A only.
    fine, coarse = tp_split(freq_to_tp(440.0, clock_hz=1_500_000))
    for _ in range(50):
        f = RegisterFrame()
        f[0], f[1] = fine, coarse
        f[7] = mixer_byte(True, False, False, False, False, False)
        f[8] = 12
        f[13] = 0xFF  # no envelope retrigger
        frames.append(f)
    song = YmSong(
        frames=frames,
        song_name="Smoke",
        author="audio2ay",
        comment="A4 1s",
        clock_hz=1_500_000,
        frame_rate_hz=50,
    )
    out = tmp_path / "a4.ym"
    write_ym5(out, song, interleaved=True)
    return out


def test_ym5_roundtrip(tmp_path: Path) -> None:
    p = _make_song(tmp_path)
    song = read_ym5(p)
    assert len(song.frames) == 50
    assert song.clock_hz == 1_500_000
    assert song.frame_rate_hz == 50
    assert song.song_name == "Smoke"
    assert song.author == "audio2ay"
    # Tone register A on every frame should be the A4 fine byte.
    fine, coarse = tp_split(freq_to_tp(440.0, clock_hz=1_500_000))
    for f in song.frames:
        assert f[0] == fine
        assert f[1] == coarse
        assert f[8] == 12


def test_ym5_header_magic(tmp_path: Path) -> None:
    p = _make_song(tmp_path)
    data = p.read_bytes()
    assert data[:4] == b"YM5!"
    assert data[4:12] == b"LeOnArD!"
    assert data.endswith(b"End!")


def _dual_song() -> YmSong:
    fine_a, coarse_a = tp_split(freq_to_tp(440.0, clock_hz=1_500_000))
    fine_b, coarse_b = tp_split(freq_to_tp(660.0, clock_hz=1_500_000))
    frames_a: list[RegisterFrame] = []
    frames_b: list[RegisterFrame] = []
    for _ in range(20):
        fa = RegisterFrame()
        fa[0], fa[1] = fine_a, coarse_a
        fa[7] = mixer_byte(True, False, False, False, False, False)
        fa[8] = 12
        fa[13] = 0xFF
        frames_a.append(fa)
        fb = RegisterFrame()
        fb[0], fb[1] = fine_b, coarse_b
        fb[7] = mixer_byte(True, False, False, False, False, False)
        fb[8] = 10
        fb[13] = 0xFF
        frames_b.append(fb)
    return YmSong(
        frames=frames_a,
        frames_b=frames_b,
        song_name="Dual",
        clock_hz=1_500_000,
        frame_rate_hz=50,
    )


def test_ym5_dual_chip_roundtrip(tmp_path: Path) -> None:
    song = _dual_song()
    assert song.is_dual_chip
    out = tmp_path / "dual.ym"
    write_ym5(out, song, interleaved=True)

    loaded = read_ym5(out)
    assert loaded.is_dual_chip
    assert loaded.frames_b is not None
    assert len(loaded.frames) == 20
    assert len(loaded.frames_b) == 20

    fine_a, coarse_a = tp_split(freq_to_tp(440.0, clock_hz=1_500_000))
    fine_b, coarse_b = tp_split(freq_to_tp(660.0, clock_hz=1_500_000))
    for fa, fb in zip(loaded.frames, loaded.frames_b):
        assert (fa[0], fa[1], fa[8]) == (fine_a, coarse_a, 12)
        assert (fb[0], fb[1], fb[8]) == (fine_b, coarse_b, 10)


def test_ym5_dual_chip_non_interleaved_roundtrip(tmp_path: Path) -> None:
    song = _dual_song()
    out = tmp_path / "dual_seq.ym"
    write_ym5(out, song, interleaved=False)
    loaded = read_ym5(out)
    assert loaded.is_dual_chip
    assert loaded.frames_b is not None and len(loaded.frames_b) == 20
    fine_b, coarse_b = tp_split(freq_to_tp(660.0, clock_hz=1_500_000))
    assert (loaded.frames_b[0][0], loaded.frames_b[0][1]) == (fine_b, coarse_b)

