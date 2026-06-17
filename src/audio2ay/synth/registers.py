"""AY-3-8910 register set helpers.

A `RegisterFrame` is a 14-byte snapshot covering R0..R13 (the 14 user-visible
sound-generation registers — R14/R15 are GPIO and unused for music).
The YM5 file format pads each frame to 16 bytes, with R14/R15 = 0 unless the
file uses Atari ST-specific digidrum effects (out of scope here).

Register layout:

    R0  Tone A fine
    R1  Tone A coarse  (4 bits)
    R2  Tone B fine
    R3  Tone B coarse
    R4  Tone C fine
    R5  Tone C coarse
    R6  Noise period   (5 bits)
    R7  Mixer/IO ctrl  (bits 0-2 = tone enables, 3-5 = noise enables; LOW = ON)
    R8  Vol A          (bits 0-3 = level, bit 4 = follow envelope)
    R9  Vol B
    R10 Vol C
    R11 Envelope period fine
    R12 Envelope period coarse
    R13 Envelope shape (low 4 bits)
"""

from __future__ import annotations

from dataclasses import dataclass, field

NUM_REGISTERS = 14  # R0..R13 (musical), R14/R15 unused
YM_FRAME_BYTES = 16  # YM5 file stores 16 bytes per frame


# Mixer convention: bits LOW = enabled.
def mixer_byte(
    tone_a: bool, tone_b: bool, tone_c: bool, noise_a: bool, noise_b: bool, noise_c: bool
) -> int:
    """Compose R7 from per-channel tone/noise enables (True == audible)."""
    b = 0
    if not tone_a:
        b |= 0x01
    if not tone_b:
        b |= 0x02
    if not tone_c:
        b |= 0x04
    if not noise_a:
        b |= 0x08
    if not noise_b:
        b |= 0x10
    if not noise_c:
        b |= 0x20
    # bits 6,7 = IO direction, set to input (0).
    return b


@dataclass
class RegisterFrame:
    """One AY snapshot (R0..R13). Values default to "everything muted, nothing playing"."""

    regs: list[int] = field(default_factory=lambda: [0] * NUM_REGISTERS)

    def __post_init__(self) -> None:
        if len(self.regs) != NUM_REGISTERS:
            raise ValueError(f"RegisterFrame must have {NUM_REGISTERS} bytes")
        # Default mixer: everything OFF (all bits high).
        if self.regs[7] == 0:
            self.regs[7] = 0x3F

    def __getitem__(self, i: int) -> int:
        return self.regs[i]

    def __setitem__(self, i: int, v: int) -> None:
        self.regs[i] = int(v) & 0xFF

    def to_bytes16(self) -> bytes:
        """16-byte form (YM5 frame size); R14/R15 zeroed."""
        return bytes(self.regs) + b"\x00\x00"

    def copy(self) -> RegisterFrame:
        return RegisterFrame(list(self.regs))
