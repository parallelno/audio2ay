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

## High-frequency tilt (brightness)

AY square waves are spectrally brighter than most acoustic sources. The
`--brightness` flag (default 0.85) applies a per-octave gain tilt above middle
C: a note one octave above C4 is multiplied by 0.85, two octaves by 0.85², and
so on. This pulls the spectral centroid back toward the source without
completely muting high notes. Set to 1.0 to disable.

## Idle-channel unison enrichment

On sparse (mono/duo) material, unused tone channels are filled with slightly
**detuned unison copies** of the sounding notes. Two or three square waves a few
cents apart beat against each other, fattening an otherwise bare tone. The
volume of these copies is controlled by `--enrich-volume` (default 0.4 — loud
enough to add harmonic body, quiet enough that the real melody note stays
clearly above them). Disable entirely with `--no-enrich`.

## Phantom-drum gating

Demucs can hallucinate faint percussion from tonal onsets. Two independent
gates defend against this:

1. **Energy floor** — the drum track is dropped when the drums stem's RMS is
   below 6 % of the mix RMS.
2. **Tonality gate** — if the drums stem's active-frame spectral flatness falls
   below 0.04 (real drums score ~0.05–0.15; leaked piano < 0.03), the entire
   drum track is treated as phantom and discarded. This reliably silences the
   burst of noise that would otherwise appear on piano-only sources.

## Harmonic-overtone suppression

Basic Pitch sometimes transcribes strong piano overtones as independent note
events (e.g. the 2nd partial of A3 appearing as a spurious A4). Notes whose
pitch falls within 50 cents of a harmonic-series interval above a
simultaneously active louder note are removed, provided they are noticeably
weaker (velocity ≤ 80 % of the fundamental). An intentional octave chord where
both notes are struck at similar strength is preserved.

## Residual-resonance gate

After a piano key is released, its string continues to resonate quietly.
Basic Pitch can transcribe this as a new (very soft) note event, which the
per-note amplitude follower then normalises to its own quiet onset — playing it
at near-full AY volume for its whole duration. This creates a background wash
that drowns the melody.

The `--abs-onset-gate` flag (default 0.06) applies a downward expander: a
note whose onset energy is below `stem_peak × threshold` has its velocity
scaled by `√(onset / threshold)`. At 6 % of peak, a note at 1.5 % peak energy
is already attenuated to 50 %. Loudly-struck notes are unaffected. Increase the
threshold (e.g. `--abs-onset-gate 0.12`) for sources with heavy resonance.

With `--dual-chip`, a second AY adds three more tone channels. The extra
polyphony goes on chip B, and any of chip B's channels not carrying their own
note double chip A's chord (slightly detuned) so they are never left idle while
there is sound to thicken. `--enrich-volume` controls the loudness of all
unison copies on both chips.

> **Note:** `--dual-chip` writes a project-specific YM5 extension (32 register
> rows per frame instead of 16, flagged in the song attributes). The bundled
> emulator/`render` understand it; a vanilla single-AY YM player will read only
> chip A.
