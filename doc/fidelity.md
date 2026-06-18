# How the conversion preserves fidelity

A few stages beyond the basic transcription keep the AY rendering close to
the source.

## Per-note amplitude following

Each note's per-frame volume tracks the source stem's energy around its pitch
(a hop-aligned STFT), so notes get real attack/decay instead of a flat,
organ-like sustain.

## Fine note transcription

Basic Pitch's minimum note length is lowered from ~128 ms to ~46 ms (and onset
threshold relaxed) so fast passages are not silently dropped.

## Pulse-width modulation

By default, the square-wave oscillators use a 70% duty cycle instead of the
standard 50%, naturally suppressing harsh high harmonics while preserving the
fundamental. Adjust with `--pulse-width` (0.5=bright/harsh, 0.7=default,
0.75=dark/smooth).

## Idle-channel unison enrichment

On sparse (mono/duo) material, unused tone channels are filled with slightly
**detuned unison copies** of the sounding notes. Two or three square waves a few
cents apart beat against each other, fattening an otherwise bare tone. Disable
with `--no-enrich`.

## Phantom-drum gating

Demucs can hallucinate faint percussion from tonal onsets; the drum track is
dropped when the drums stem is negligible relative to the mix, avoiding spurious
noise on clean tonal inputs.

## Dual-AY (TurboSound)

With `--dual-chip`, a second AY adds three more tone channels. The extra
polyphony goes on chip B, and any of chip B's channels not carrying their own
note double chip A's chord (slightly detuned) so they are never left idle while
there is sound to thicken.

> **Note:** `--dual-chip` writes a project-specific YM5 extension (32 register
> rows per frame instead of 16, flagged in the song attributes). The bundled
> emulator/`render` understand it; a vanilla single-AY YM player will read only
> chip A.
