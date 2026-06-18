# Contributing

## Setup

```pwsh
git clone <repo>
cd audio2ay

# Full dev install: ML pipeline + test/lint tools
pip install -e .[all]
```

`.[all]` installs:
- `.[ml]` — Demucs, Basic Pitch (ONNX), onnxruntime, torch (required for conversion)
- `pytest>=7` — test runner
- `ruff>=0.1` — linter / formatter

## Generating test samples

The test suite needs audio samples. A helper script creates synthetic ones and
optionally fetches small public-domain clips:

```pwsh
# Synthetic + downloads (recommended)
python scripts/make_samples.py

# Corporate network with HTTPS MITM? Bypass cert checks:
python scripts/make_samples.py --insecure

# No network at all:
python scripts/make_samples.py --no-online
```

## Running tests

```pwsh
python -m pytest
```

The core test suite is pure-Python — no ML deps required. ML-dependent paths
are exercised by the smoke tests when the `.[ml]` extras are installed.

Sample files used by the tests:

| File | What it exercises |
| --- | --- |
| `samples/01_arpeggio_mono.wav` | monophonic pitch quantisation |
| `samples/02_bass_and_lead.wav` | 2-voice polyphony + bass channel |
| `samples/03_drum_loop.wav` | drums → noise channel routing |
| `samples/04_chord_progression.wav` | 3-voice scheduler stress |
| `samples/trumpet.ogg` | real solo instrument |
| `samples/brahms.ogg` | small string-orchestra excerpt |
| `samples/nutcracker.ogg` | full orchestral excerpt |

## Linting

```pwsh
ruff check src tests
ruff format src tests
```

## Project layout

```
src/audio2ay/
    cli.py              # argparse entry point
    pipeline.py         # end-to-end convert_audio_to_ym()
    analysis/           # audio analysis (separation, transcription, drums, timeline)
    synth/              # AY register synthesis (scheduler, quantizers, frame builder)
    io/                 # YM5 writer + AY emulator binding
tests/                  # pytest suite
scripts/                # helper scripts (make_samples, debug utilities)
doc/                    # documentation (you are here)
design/plan.md          # research plan and architecture notes
```

## Architecture notes

See [design/plan.md](../design/plan.md) for the full research plan, phase
breakdown, and design decisions.
