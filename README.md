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

A pure-Python AY-3-8910 emulator and a small PySide6 GUI let you A/B the
original audio against the AY rendering.

## Install

```pwsh
# Foundations only (YM I/O + pure-Python AY emulator + tests)
pip install -e .

# Add ML pipeline (Demucs + Basic Pitch). First run downloads model weights.
pip install -e .[ml]

# Add GUI validator
pip install -e .[gui]

# Everything
pip install -e .[all]
```

## CLI

```pwsh
# Convert audio to a .ym register stream
audio2ay convert input.wav out.ym

# Render a .ym back to wav using the bundled emulator
audio2ay render out.ym out_ay.wav

# Preview (convert + render directly to audio, no .ym written)
audio2ay preview input.wav out.wav

# Or omit output to get default MP3: build/<input-stem>.mp3
audio2ay preview input.wav

# One-shot: convert + render + write side-by-side wavs
audio2ay validate input.wav --outdir build/

# Launch the A/B GUI
audio2ay gui
```

### Convert options

`audio2ay convert`, `preview`, and `validate` accept flags to tune fidelity:

| Flag | Default | Scope | Effect |
| --- | --- | --- | --- |
| `--no-separation` | off | convert | Skip Demucs; treat the input as a single tonal stem. **Loses drums isolation and degrades pitch accuracy on mixed signals.** |
| `--demucs-model NAME` | `htdemucs` | convert | Demucs model to use for stem separation. |
| `--dual-chip` | off | convert, preview, validate | Emit **two** AY chips (TurboSound-style) for 6 tone channels. |
| `--no-enrich` | off | convert, preview, validate | Disable filling idle channels with detuned unison copies. |
| `--detune-cents N` | `9.0` | convert, preview, validate | Unison detune spread (cents) for idle-channel enrichment. |
| `--enrich-volume-b N` | `0.75` | convert, preview, validate | Chip B's doubling volume (0–1) when using `--dual-chip`. |
| `--no-loudness-match` | off | convert, preview, validate | Skip tracking original's loudness contour (keep flat level). |
| `--brightness N` | `0.85` | convert, preview, validate | Per-octave high-voice attenuation (1.0=off, lower=darker). |
| `--frame-rate N` | `50` | convert, preview, validate | Register update rate (50 or 100 Hz). |
| `--100hz` | off | convert, preview, validate | Shortcut to force 100 Hz register updates. |
| `--pulse-width N` | `0.7` | preview, render, validate | Pulse duty cycle: 0.5=square/harsh, 0.7=default, 0.75=wider/darker. Reduces harmonics vs. pure square wave. |
| `--envelope` | off | convert, preview, validate | Enable the experimental hardware envelope on channel A. |
| `-v`, `--verbose` | off | Show full third-party logs (TensorFlow / Basic Pitch / absl). |

`validate` accepts the same flags as `convert` and `preview` except `--demucs-model`. 
`render` accepts only `--pulse-width` and `--sample-rate` (the conversion flags are handled during `convert`).

```pwsh
# Fuller sound on sparse material via two chips (6 channels)
audio2ay convert input.wav out.ym --dual-chip
audio2ay preview input.wav out.wav --dual-chip --pulse-width 0.7

# Higher register update rate for finer timing on Vector-06C
audio2ay convert input.wav out_100hz.ym --100hz
audio2ay preview input.wav out_100hz.wav --frame-rate 100

# Wider unison shimmer; or turn enrichment off entirely
audio2ay convert input.wav out.ym --detune-cents 14
audio2ay convert input.wav out.ym --no-enrich

# Adjust pulse width to control harshness (narrower = brighter, wider = darker)
audio2ay preview input.wav out.wav --pulse-width 0.65   # Brighter
audio2ay preview input.wav out.wav --pulse-width 0.75   # Darker
```

## How the conversion preserves fidelity

A few stages beyond the basic transcription keep the AY rendering close to
the source:

- **Per-note amplitude following** — each note's per-frame volume tracks the
  source stem's energy around its pitch (a hop-aligned STFT), so notes get
  real attack/decay instead of a flat, organ-like sustain.
- **Fine note transcription** — Basic Pitch's minimum note length is lowered
  from ~128 ms to ~46 ms (and onset threshold relaxed) so fast passages are
  not silently dropped.
- **Pulse-width modulation** — by default, the square-wave oscillators use a
  70% duty cycle instead of the standard 50%, naturally suppressing harsh
  high harmonics while preserving the fundamental. Adjust with `--pulse-width`
  (0.5=bright/harsh, 0.7=default, 0.75=dark/smooth).
- **Idle-channel unison enrichment** — on sparse (mono/duo) material, unused
  tone channels are filled with slightly **detuned unison copies** of the
  sounding notes. Two or three square waves a few cents apart beat against
  each other, fattening an otherwise bare tone. Disable with `--no-enrich`.
- **Phantom-drum gating** — Demucs can hallucinate faint percussion from
  tonal onsets; the drum track is dropped when the drums stem is negligible
  relative to the mix, avoiding spurious noise on clean tonal inputs.
- **Dual-AY (TurboSound)** — with `--dual-chip`, a second AY adds three more
  tone channels. The extra polyphony goes on chip B, and any of chip B's
  channels not carrying their own note double chip A's chord (slightly
  detuned) so they are never left idle while there is sound to thicken.

> **Note:** `--dual-chip` writes a project-specific YM5 extension (32 register
> rows per frame instead of 16, flagged in the song attributes). The bundled
> emulator/`render` understand it; a vanilla single-AY YM player will read
> only chip A.

## Testing

A helper script populates a `samples/` folder with **synthetic** clips
(always created) plus a few small **public-domain** instrumental clips
(fetched on best-effort).

```pwsh
# Generate everything (synthetic + downloads)
python scripts/make_samples.py

# Corporate network MITMs HTTPS? Bypass cert checks for the download step:
python scripts/make_samples.py --insecure

# No network at all? Synthetic-only:
python scripts/make_samples.py --no-online
```

Then run the unit tests and a quick conversion:

```pwsh
# Run the test suite (pure-Python, no ML deps required)
python -m pytest

# Convert one sample → .ym → wav (only use --no-separation for single clean instruments)
audio2ay convert samples/02_bass_and_lead.wav build/bass_lead.ym --no-separation
audio2ay render  build/bass_lead.ym build/bass_lead_ay.wav

# Or do everything + dump the original next to the AY rendering for A/B
audio2ay validate samples/03_drum_loop.wav --outdir build/

# Launch the GUI player (requires `pip install -e .[gui]`)
audio2ay gui
```

What each sample exercises:

| File | Tests |
| --- | --- |
| `samples/01_arpeggio_mono.wav` | monophonic pitch quantisation |
| `samples/02_bass_and_lead.wav` | 2-voice polyphony + bass channel |
| `samples/03_drum_loop.wav` | drums → noise channel routing |
| `samples/04_chord_progression.wav` | 3-voice scheduler stress |
| `samples/trumpet.ogg` | real solo instrument |
| `samples/brahms.ogg` | small string-orchestra excerpt |
| `samples/nutcracker.ogg` | full orchestral excerpt (Demucs separation works best here) |

### `--no-separation` flag

By default, Demucs separates your mix into **drums**, **bass**, **other** (melody/harmony), and **vocals**.
Each stem is transcribed independently, giving clean pitch detection and proper drum → noise-channel routing.

With `--no-separation`, the entire mix is treated as one signal:
- ❌ **No drum channel** — drums become silence (0 hits).
- ❌ **Blurred pitch detection** — bass and melody interfere with each other; low notes especially degrade.
- ❌ **Lost part separation** — both transcribers see the full mix instead of isolated stems.

**When is it useful?** Only for **single clean instruments** (solo synth, solo trumpet, clean arpeggio).
For anything else (orchestras, bands, full mixes), use separation. It measurably improves quality.

Once you `pip install -e .[ml]` (Demucs + Basic Pitch), keep the flag off for real recordings.

## Status

See [design/plan.md](design/plan.md) for the research plan.
