# Plan: High-Fidelity Instrumental Audio → AY-3-8910 Register Stream Converter

## TL;DR
Build a Python tool that converts instrumental audio into a 50 Hz `.ym` register stream for a Vector-06C-style AY-3-8910 (clock = 1.5 MHz), plus a small GUI validator that A/Bs the original against the AY rendering. The pipeline is **Demucs (stem separation) → Basic Pitch (multi-F0 transcription, per stem) + drum onset classifier → 50 Hz event timeline → 3-voice scheduler with voice-leading → per-frame quantizer (tone period @ 1.5 MHz, 4-bit log volume, optional hardware envelope, shared noise) → YM5 writer**. Validation renders the YM through the Ayumi emulator and plays both signals back with level meters.

---

## Hardware target (fixed inputs)
- **Chip**: AY-3-8910 on Vector-06C
- **Master clock**: 1.5 MHz (user-confirmed)
- **Frame rate**: 50 Hz (20 ms per register snapshot)
- **Tone period**: 12-bit (1..4095), `f = clock / (16 * TP)` → range ≈ 22.9 Hz to 93.75 kHz; pitch quantization is sub-cent below ~C6 and grows to ~30 cents near C8
- **Noise period**: 5-bit (0..31), shared across channels
- **Volume**: 4-bit log-DAC per channel (≈3 dB/step) OR follow hardware envelope
- **Envelope generator**: 16-bit period, 1 of 8 useful shapes, single-shared

## Scope decisions (locked)
- **Input**: instrumental only (mp3/wav/flac/ogg/m4a). Vocal content is out of scope; the pipeline will defensively drop the Demucs vocals stem and warn if its energy is non-trivial.
- **Output**: `.ym` (YM5 variant — vanilla register dump, no Atari-specific digidrum/SID effects).
- **Stack**: Python.
- **ML**: allowed.
- **AY features used**: 3 tone channels, noise channel, hardware envelope.
- **Excluded**: per-frame volume PCM tricks, channel-stealing for drums, custom Vector-06C-only register tricks.
- **Validator**: simple GUI — load file, convert, play original vs AY, level meters.

---

## Phases

### Phase A — Foundations (no ML yet)
Goal: prove the round-trip plumbing works end-to-end with a hand-authored register stream.

A1. Project scaffolding (`pyproject.toml`, `src/audio2ay/` layout described below, `pytest` + `ruff`).
A2. **Pitch & volume quantizers** (`synth/pitch_quantizer.py`, `synth/volume_quantizer.py`) with unit tests for known frequencies/dB references at 1.5 MHz.
A3. **YM5 writer** (`io/ym_writer.py`): header magic `YM5!`, `LeOnArD!` check, frame count, flags (interleaved/non-loop), chip clock = 1 500 000, frame rate = 50, loop frame, song name/author/comment fields, **interleaved register storage** (R0[0..N-1], …, R15[0..N-1]), `End!` trailer. Emit raw uncompressed first; LHA packing is a deferred polish step.
A4. **Ayumi emulator binding** (`io/ay_emulator.py`): `ctypes` wrapper around `ayumi.c` (`ayumi_configure(is_ym=0, clock_rate=1500000, sr=44100)`, `set_tone/noise/mixer/volume/envelope/process`). Build `ayumi.c` to a shared library at install time (or vendored prebuilt). Pure-Python fallback acceptable if ctypes is fragile on Windows.
A5. **YM player** that streams a `.ym` through Ayumi to a WAV. End-to-end smoke test: hand-author a 50-frame A-major arpeggio, write YM, render WAV, verify pitches by FFT.

### Phase B — Audio analysis frontend
Goal: turn arbitrary instrumental audio into a clean, time-aligned event stream.

B1. **Decode & normalize** (`analysis/audio_in.py`): librosa/soundfile → mono float32, target SR 22 050 Hz (matches Basic Pitch and Demucs internals where applicable).
B2. **Source separation** (`analysis/separation.py`): Demucs `htdemucs` (4 stems: drums, bass, vocals, other). Drop vocals stem. Optionally evaluate `htdemucs_6s` (adds piano + guitar) for better tonal disambiguation — gated by a config flag.
B3. **Multi-pitch transcription per tonal stem** (`analysis/multipitch.py`): Spotify Basic Pitch (`from basic_pitch.inference import predict`) on `bass` and `other` separately. Configure `minimum_frequency`/`maximum_frequency` per stem (e.g. bass capped at ~250 Hz). Output: list of note events (start, end, pitch, velocity, optional pitch-bend curve).
B4. **Drum onset & classification** (`analysis/drums.py`): librosa onset detection on the drums stem + spectral-centroid-based classifier into {kick, snare, hat/cymbal}. Output: timestamped percussion events with intensity. Optional upgrade: madmom `RNNDrumProcessor`.
B5. **Event-stream resampling to 50 Hz grid** (`analysis/timeline.py`): merge note + drum events into a unified 20 ms-quantized sequence, preserving onsets, sustain frames, and velocities. Round onsets to nearest frame; trim sub-frame note offs.

### Phase C — Synthesis (audio events → AY registers)
Goal: solve the 3-tone budget, fold in noise and envelope, and produce 14 register bytes per frame.

C1. **Voice scheduler** (`synth/voice_scheduler.py`): assign at most 3 simultaneous notes to channels A/B/C per frame.
   - Hard rule: bass stem notes are pinned to channel A (lowest-frequency, hardware envelope candidate).
   - Channels B, C: from `other` stem. Pick top-2 by salience (note velocity × spectral mask energy). Resolve frame-to-frame assignment via Hungarian matching with a voice-leading cost = `α·|Δsemitones| + β·new_voice_penalty + γ·dropped_voice_salience`.
   - Smoothing: hysteresis so a note must persist ≥2 frames to claim a channel; trailing release frames if no successor.
C2. **Pitch & volume quantization per active voice**: TP from `synth/pitch_quantizer.py`; volume from log-mapped per-frame RMS of the source spectrogram bin around the note (perceptually weighted, mapped to 0..15).
C3. **Hardware envelope assignment** (`synth/hw_envelope.py`):
   - Detect frames where channel A holds a sustained note ≥ K frames with a decay-like amplitude envelope.
   - Fit best AY shape (9 = decay, 10/14 = tremolo, 11 = decay-then-hold) and envelope period (16-bit, derived from desired cycle length); enable bit 4 in channel A volume register.
   - Skip if shape/period would have to change mid-note (would retrigger), or if amplitude is well-approximated by per-frame static 4-bit volume (cheaper, no risk of clicks).
C4. **Noise mapping** (`synth/noise_mapper.py`):
   - Map drum class → noise period (kick ≈ 28–31, snare ≈ 8–14, hi-hat ≈ 1–4).
   - Enable noise mixer bit on channel C (or whichever has lowest tonal demand that frame); volume from drum hit intensity; auto-decay over 2–4 frames.
   - When all 3 channels are busy with tones, allow simultaneous tone+noise on channel C (mixer permits it) rather than dropping a tone.
C5. **Frame builder** (`synth/frame_builder.py`): assemble R0..R13 from voice + envelope + noise state. Apply post-processing:
   - Click suppression: avoid sudden TP jumps > N at low volume — rebrief envelope retrigger order.
   - Mixer-bit hygiene: explicit tone-off when volume = 0 to silence inactive channels.
   - Per-frame register diff log for debugging.
C6. **YM5 emission**: stream of frames → `io/ym_writer.py`.

### Phase D — Validation tool
Goal: enable the researcher to hear and measure result quality.

D1. **GUI** (`validate/gui.py`, PySide6): file picker, “Convert” button, dual play buttons (Original / AY), A/B toggle, live level meters (per-source RMS), conversion progress bar.
D2. **Audio playback**: `sounddevice` for cross-platform low-friction output. AY rendering goes through the Phase A4 Ayumi binding to a 44.1 kHz buffer.
D3. **Metrics panel** (optional but cheap): log-mel spectrogram MSE between original and AY render; `mir_eval` multi-F0 metrics by re-transcribing the AY render with Basic Pitch and comparing to the original transcription.
D4. **CLI parity** (`cli.py`): `audio2ay convert IN.wav OUT.ym`, `audio2ay render IN.ym OUT.wav`, `audio2ay validate IN.wav` (one-shot pipeline + WAV outputs without GUI).

### Phase E — Tuning & evaluation
E1. Curate a small reference set (5–10 instrumental clips, varied genres: solo piano, jazz trio, rock instrumental, electronic, orchestral excerpt).
E2. Sweep scheduler weights (α, β, γ), hysteresis thresholds, envelope-assignment thresholds against the metrics above.
E3. Document final defaults in `docs/research.md`.

---

## Project layout
```
src/audio2ay/
  cli.py
  pipeline.py
  io/
    audio_in.py        # decode + resample
    ym_writer.py       # YM5 (+ optional LHA later)
    ay_emulator.py     # ayumi ctypes wrapper, render YM → WAV
  analysis/
    separation.py      # Demucs htdemucs
    multipitch.py      # Basic Pitch (per stem)
    drums.py           # onset + class
    timeline.py        # 50 Hz grid merge
  synth/
    pitch_quantizer.py
    volume_quantizer.py
    voice_scheduler.py
    hw_envelope.py
    noise_mapper.py
    frame_builder.py
  validate/
    gui.py             # PySide6 GUI
    metrics.py         # spectral + mir_eval
  vendor/
    ayumi.c, ayumi.h   # MIT, vendored
tests/
docs/research.md       # long-form findings + decisions
pyproject.toml
```

## Key external dependencies
- `librosa`, `soundfile`, `numpy`, `scipy`
- `demucs` (htdemucs / htdemucs_6s)
- `basic-pitch` (Spotify, Apache-2.0)
- `madmom` (optional, drum RNN)
- `mir_eval`
- `PySide6`, `sounddevice`
- `ayumi` (vendored C source, MIT) compiled via `cffi` or built by a small `setup.py`/`scikit-build` shim

## Verification plan
1. **Unit tests** — pitch quantizer round-trips A4=440 Hz → TP=213 → 440.14 Hz (<1 cent). Volume quantizer monotonic and 0 ↔ silence. YM5 header byte-identical to a known-good reference file.
2. **Round-trip smoke test** — author 50-frame arpeggio → write YM → render via Ayumi → FFT peak per frame matches expected pitch ±1 bin.
3. **Stem-isolation sanity** — feed a known multi-track instrumental (drums + bass + lead) where stems are independently available; check Demucs separation SNR > some threshold and Basic Pitch note F-measure on isolated bass > 0.7.
4. **End-to-end metric** — log-mel MSE between original and AY render should drop monotonically across phases; track in a CSV per commit.
5. **Listening test** — manual A/B in the GUI on the 5–10-clip reference set after Phase E.

## Decisions
- Target Vector-06C with **AY clock = 1.5 MHz** (per user). Documented but easy to change in one config constant.
- **YM5** chosen over YM6 (no need for digidrum/SID effects) and over `.psg`/`.vtx` (better emulator support per user pick).
- ML stack centered on **Demucs htdemucs + Basic Pitch**: best-in-class instrument-agnostic polyphonic transcription with reasonable size.
- **Hardware envelope used only on channel A (bass)** to avoid the contention from a single-shared envelope generator.
- **No per-frame volume PCM tricks** and **no channel stealing for drums** — kept out of scope.
- **Vocals defensively dropped** even though the input is declared instrumental.
- **LHA packing of YM5 deferred** to a polish step; uncompressed YM5 is accepted by ST-Sound and aylet-style players.

## Further considerations (suggestions; not blockers)
1. **Demucs variant**: `htdemucs` (4 stems) vs `htdemucs_6s` (6 stems incl. piano/guitar). Recommend default to `htdemucs` (faster, more reliable bass+drums) and expose `--6s` flag for tonal-rich inputs. Option A: 4-stem default / Option B: 6-stem default / Option C: auto-pick by genre tag (out of scope).
2. **Preferred AY emulator binding**: (A) ctypes against vendored `ayumi.c` (best fidelity, build complexity), (B) pure-Python port of Ayumi (easier install, slower), (C) call out to `ayumi_render` CLI as a subprocess. Recommend A with B as fallback.
3. **GUI framework**: PySide6 (LGPL, modern) vs PyQt6 vs Tkinter. Recommend PySide6 — least friction with `sounddevice` and good level-meter widgets.

---

## Implemented fidelity enhancements (post-plan)

Tuning work layered on top of the Phase A–D plumbing, all reflected in the CLI:

- **Per-note amplitude following** (`analysis/amplitude.py`): a hop-aligned STFT
  per stem samples the source energy around each note's pitch, so per-frame
  volume tracks real attack/decay instead of a flat onset velocity. Replaces the
  earlier static-sustain behaviour that sounded organ-like.
- **Finer note transcription** (`analysis/multipitch.py`): Basic Pitch's
  `minimum_note_length` lowered ~128 ms → ~46 ms and `onset_threshold` relaxed
  to 0.4 so fast passages survive into the timeline. Exposed as `transcribe`
  parameters.
- **Idle-channel unison enrichment** (`synth/frame_builder.py`): empty tone
  channels are filled with slightly detuned (`±cents`) copies of the sounding
  notes for a fuller tone on sparse material. CLI: `--no-enrich`,
  `--detune-cents`.
- **Phantom-drum gating** (`pipeline.py`): the drum track is dropped when the
  Demucs drums stem RMS is below `drum_energy_floor` (default 6 %) of the mix,
  avoiding hallucinated percussion on clean tonal inputs.
- **Hardware envelope made opt-in**: the single-shared envelope generator only
  produced a constant tremolo on channel A that overrode real dynamics; software
  amplitude following is preferred, so it is now off unless `--envelope`.
- **Dual-AY (TurboSound) mode** (`synth/voice_scheduler.py:DualVoiceScheduler`,
  `io/ym_writer.py`, `io/ay_emulator.py`): optional second AY chip for 6 tone
  channels. `YmSong.frames_b` + a custom YM5 attribute bit (32 register rows per
  frame); the bundled emulator sums both chips. Chip A is unchanged; chip B
  carries the extra polyphony and doubles chip A's chord on any otherwise-idle
  channel so it is never silent while there is sound. CLI: `--dual-chip`.
  Project-specific extension — vanilla single-AY YM players read chip A only.
