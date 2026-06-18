# audio2ay

Convert instrumental audio (mp3/wav/flac/ogg/m4a) into an **AY-3-8910 register stream** (`.ym`)
targeted at a Vector-06C-style chip clock of **1.5 MHz**.

Pipeline:

```
audio → Demucs (stem split) → Basic Pitch (multi-F0) + drum onsets
      → 50/100 Hz event timeline (per-note amplitude following)
      → 3-voice scheduler (voice-leading) [×2 for dual-AY]
      → per-frame quantizer (12-bit tone period, 4-bit log volume,
        idle-channel unison enrichment, optional hardware envelope,
        noise on a free channel)
      → YM5 writer (.ym, optional dual-chip TurboSound extension)
```

A pure-Python AY-3-8910 emulator lets you render `.ym` files back to audio.

## Install

```pwsh
# Foundations only (YM I/O + AY emulator, no ML)
pip install -e .

# Add ML pipeline — required for conversion (Demucs + Basic Pitch via ONNX).
# First run downloads model weights (~80 MB).
pip install -e .[ml]
```

## Quick start

```pwsh
# Convert audio → .ym register stream
audio2ay convert input.wav out.ym

# Render .ym → wav
audio2ay render out.ym out_ay.wav

# Preview: convert + render in one step (outputs MP3 by default)
audio2ay preview input.wav

# Full A/B: convert + render + save both wavs side-by-side
audio2ay validate input.wav --outdir build/
```

## Documentation

- [doc/cli.md](doc/cli.md) — all commands and flags with examples
- [doc/fidelity.md](doc/fidelity.md) — how the pipeline preserves fidelity
- [doc/contributing.md](doc/contributing.md) — dev setup, tests, code style
- [design/plan.md](design/plan.md) — research plan and architecture notes
