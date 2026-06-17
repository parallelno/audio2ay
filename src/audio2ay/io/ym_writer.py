"""YM5 file I/O.

YM5 is the most common register-dump container for AY-3-8910 / YM2149 music.
Spec by Arnaud Carré (Leonard / OXG):
    http://leonard.oxg.free.fr/ymformat.html

Header layout (big-endian unless noted):

    char[4]   "YM5!"
    char[8]   "LeOnArD!"           (sanity check)
    u32       NbFrames             (frame count)
    u32       SongAttributes       (bit 0 = interleaved register dump)
    u16       NbDigiDrums          (0 for vanilla AY)
    u32       MasterClockHz        (1_500_000 for Vector-06C, 2_000_000 for Atari ST, …)
    u16       FrameRateHz          (50 here)
    u32       LoopFrame            (0 = play once)
    u16       AdditionalDataSize   (0 unless digi-drums)
    NTString  SongName
    NTString  Author
    NTString  Comment
    --- frames ---
    if interleaved: R0[0..N-1] R1[0..N-1] ... R15[0..N-1]
    else          : per-frame records of 16 bytes
    char[4]   "End!"

The YM5 frame is 16 bytes; only R0..R13 are musical, R14/R15 stay zero
unless Atari ST timer-driven digidrums/SID-effects are used (out of scope).

This module produces and consumes plain (uncompressed) YM5. The on-disk
`.ym` files distributed online are usually LHA-packed; that is not
implemented here. The bundled emulator and unit tests work on the raw form.
A "YM5!" magic at byte 0 of a real .ym file means it is *already*
unpacked — many tooling chains accept either form. Players can normally
unpack LHA themselves.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .. import AY_CLOCK_HZ, FRAME_RATE_HZ
from ..synth.registers import YM_FRAME_BYTES, RegisterFrame

YM5_MAGIC = b"YM5!"
YM5_CHECK = b"LeOnArD!"
YM5_END = b"End!"
YM5_INTERLEAVED_FLAG = 0x00000001
# Non-standard extension: a second AY chip's registers follow the first
# (TurboSound-style). Stored as 32 interleaved register rows per frame instead
# of 16. Only this project's writer/reader and emulator understand it; vanilla
# YM players ignore the flag and play chip A only.
YM5_DUAL_CHIP_FLAG = 0x00000002


@dataclass
class YmSong:
    """An in-memory YM song: header metadata + a list of register snapshots.

    When ``frames_b`` is set the song is a dual-AY (TurboSound) song: ``frames``
    are chip A and ``frames_b`` are chip B, played in lock-step.
    """

    frames: list[RegisterFrame] = field(default_factory=list)
    song_name: str = ""
    author: str = ""
    comment: str = ""
    clock_hz: int = AY_CLOCK_HZ
    frame_rate_hz: int = FRAME_RATE_HZ
    loop_frame: int = 0
    frames_b: list[RegisterFrame] | None = None

    @property
    def is_dual_chip(self) -> bool:
        return self.frames_b is not None


def _write_nt_string(buf: bytearray, s: str) -> None:
    buf.extend(s.encode("latin-1", errors="replace"))
    buf.append(0)


def _read_nt_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\x00", offset)
    if end < 0:
        raise ValueError("Unterminated string in YM header")
    return data[offset:end].decode("latin-1", errors="replace"), end + 1


def write_ym5(path: str | Path, song: YmSong, *, interleaved: bool = True) -> None:
    """Serialise an in-memory song to an uncompressed `.ym` file (YM5)."""
    n = len(song.frames)
    flags = YM5_INTERLEAVED_FLAG if interleaved else 0
    dual = song.is_dual_chip
    if dual:
        flags |= YM5_DUAL_CHIP_FLAG
        frames_b = song.frames_b or []
        if len(frames_b) != n:
            raise ValueError("Dual-chip song must have equal frame counts on both chips")

    buf = bytearray()
    buf.extend(YM5_MAGIC)
    buf.extend(YM5_CHECK)
    buf.extend(struct.pack(">I", n))           # NbFrames
    buf.extend(struct.pack(">I", flags))       # SongAttributes
    buf.extend(struct.pack(">H", 0))           # NbDigiDrums
    buf.extend(struct.pack(">I", song.clock_hz))
    buf.extend(struct.pack(">H", song.frame_rate_hz))
    buf.extend(struct.pack(">I", song.loop_frame))
    buf.extend(struct.pack(">H", 0))           # AdditionalDataSize
    _write_nt_string(buf, song.song_name)
    _write_nt_string(buf, song.author)
    _write_nt_string(buf, song.comment)

    # Per-frame register width: 16 bytes (chip A) or 32 bytes (chip A + B).
    rows = YM_FRAME_BYTES * (2 if dual else 1)

    def _reg(frame_list: list[RegisterFrame], fi: int, r: int) -> int:
        f = frame_list[fi]
        return f.regs[r] & 0xFF if r < len(f.regs) else 0

    # Frames.
    if interleaved:
        for r in range(rows):
            if r < YM_FRAME_BYTES:
                for fi in range(n):
                    buf.append(_reg(song.frames, fi, r))
            else:
                rb = r - YM_FRAME_BYTES
                for fi in range(n):
                    buf.append(_reg(song.frames_b, fi, rb))
    else:
        for fi in range(n):
            buf.extend(song.frames[fi].to_bytes16())
            if dual:
                buf.extend(song.frames_b[fi].to_bytes16())

    buf.extend(YM5_END)
    Path(path).write_bytes(bytes(buf))


def read_ym5(path: str | Path) -> YmSong:
    """Parse an uncompressed YM5 file. (LHA-packed files are not supported here.)"""
    data = Path(path).read_bytes()
    if data[:4] != YM5_MAGIC:
        raise ValueError(
            f"Not an uncompressed YM5 file (magic={data[:4]!r}). "
            "If your file is LHA-packed, decompress it first (e.g. with 7-Zip)."
        )
    if data[4:12] != YM5_CHECK:
        raise ValueError("YM5 sanity check string missing")

    o = 12
    n_frames = struct.unpack(">I", data[o : o + 4])[0]; o += 4
    flags = struct.unpack(">I", data[o : o + 4])[0]; o += 4
    _n_drums = struct.unpack(">H", data[o : o + 2])[0]; o += 2
    clock_hz = struct.unpack(">I", data[o : o + 4])[0]; o += 4
    frame_rate = struct.unpack(">H", data[o : o + 2])[0]; o += 2
    loop_frame = struct.unpack(">I", data[o : o + 4])[0]; o += 4
    extra_size = struct.unpack(">H", data[o : o + 2])[0]; o += 2
    o += extra_size  # skip digidrum / extra block

    song_name, o = _read_nt_string(data, o)
    author, o = _read_nt_string(data, o)
    comment, o = _read_nt_string(data, o)

    interleaved = bool(flags & YM5_INTERLEAVED_FLAG)
    dual = bool(flags & YM5_DUAL_CHIP_FLAG)
    rows = YM_FRAME_BYTES * (2 if dual else 1)
    body_len = n_frames * rows
    body = data[o : o + body_len]
    if len(body) < body_len:
        raise ValueError("Truncated YM5 register block")

    frames: list[RegisterFrame] = []
    frames_b: list[RegisterFrame] | None = [] if dual else None
    if interleaved:
        # body laid out as r=0..rows-1, frame=0..N-1
        for fi in range(n_frames):
            regs = [body[r * n_frames + fi] for r in range(14)]
            frames.append(RegisterFrame(regs))
            if dual:
                regs_b = [body[(YM_FRAME_BYTES + r) * n_frames + fi] for r in range(14)]
                frames_b.append(RegisterFrame(regs_b))
    else:
        for fi in range(n_frames):
            base = fi * rows
            frames.append(RegisterFrame(list(body[base : base + 14])))
            if dual:
                b2 = base + YM_FRAME_BYTES
                frames_b.append(RegisterFrame(list(body[b2 : b2 + 14])))

    o += body_len
    if data[o : o + 4] != YM5_END:
        # Not fatal — many tools omit it. Warn silently for now.
        pass

    return YmSong(
        frames=frames,
        song_name=song_name,
        author=author,
        comment=comment,
        clock_hz=clock_hz,
        frame_rate_hz=frame_rate,
        loop_frame=loop_frame,
        frames_b=frames_b,
    )
