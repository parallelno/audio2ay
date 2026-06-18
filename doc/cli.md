# CLI reference

## Commands

```pwsh
# Convert audio to a .ym register stream
audio2ay convert input.wav out.ym

# Render a .ym back to wav using the bundled emulator
audio2ay render out.ym out_ay.wav

# Preview: convert + render directly to audio (no .ym written)
audio2ay preview input.wav out.wav

# Omit output to get default MP3: build/<input-stem>.mp3
audio2ay preview input.wav

# One-shot: convert + render + write side-by-side wavs for A/B comparison
audio2ay validate input.wav --outdir build/
```

## Flags

`audio2ay convert`, `preview`, and `validate` accept flags to tune fidelity:

| Flag | Default | Scope | Effect |
| --- | --- | --- | --- |
| `--demucs-model NAME` | `htdemucs` | convert | Demucs model to use for stem separation. |
| `--dual-chip` | off | convert, preview, validate | Emit **two** AY chips (TurboSound-style) for 6 tone channels. |
| `--no-enrich` | off | convert, preview, validate | Disable filling idle channels with detuned unison copies. |
| `--detune-cents N` | `9.0` | convert, preview, validate | Unison detune spread (cents) for idle-channel enrichment. |
| `--enrich-volume-b N` | `0.75` | convert, preview, validate | Chip B's doubling volume (0–1) when using `--dual-chip`. |
| `--no-loudness-match` | off | convert, preview, validate | Skip tracking original's loudness contour (keep flat level). |
| `--brightness N` | `0.85` | convert, preview, validate | Per-octave high-voice attenuation (1.0=off, lower=darker). |
| `--frame-rate N` | `50` | convert, preview, validate | Register update rate (50 or 100 Hz). |
| `--100hz` | off | convert, preview, validate | Shortcut for `--frame-rate 100`. |
| `--envelope` | off | convert, preview, validate | Run the hardware envelope at the bass note's pitch on channel A, giving a sawtooth timbre instead of a plain square wave. |
| `-v`, `--verbose` | off | all | Show full third-party logs (Basic Pitch / absl). |

`validate` accepts the same flags as `convert` and `preview` except `--demucs-model`.
`render` accepts only `--sample-rate`.

## Examples

```pwsh
# Fuller sound via two chips (6 tone channels)
audio2ay convert input.wav out.ym --dual-chip
audio2ay preview input.wav out.wav --dual-chip

# Higher register update rate for finer timing on Vector-06C
audio2ay convert input.wav out_100hz.ym --100hz
audio2ay preview input.wav out_100hz.wav --frame-rate 100

# Wider unison shimmer; or turn enrichment off entirely
audio2ay convert input.wav out.ym --detune-cents 14
audio2ay convert input.wav out.ym --no-enrich
```
