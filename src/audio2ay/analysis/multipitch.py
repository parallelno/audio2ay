"""Multi-pitch transcription for tonal stems via Spotify Basic Pitch.

Returns a list of note events with start/end times (seconds), MIDI pitch, and
a per-note normalised velocity in [0, 1].

Requires ``onnxruntime`` and the ``basic-pitch`` package (ONNX model included).
Install via ``pip install -e .[ml]``.
"""

from __future__ import annotations

import importlib.util
import importlib.machinery
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class NoteEvent:
    """One transcribed note (from Basic Pitch, polyphonic, pitch bends collapsed to median)."""

    start_sec: float
    end_sec: float
    midi_pitch: float           # may be fractional if pitch bends are tracked
    velocity: float             # 0..1
    salience: float = 1.0       # extra weight used by the voice scheduler

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


# Harmonic-series intervals in semitones for 2nd through 8th partials.
# Piano inharmonicity adds at most ~30 cents for high partials, so 50-cent
# tolerance (0.5 st) covers all cases.
_HARMONIC_ST: tuple[float, ...] = (12.0, 19.019, 24.0, 27.863, 31.174, 33.688, 36.0)


def transcribe(
    audio: np.ndarray,
    sr: int,
    *,
    min_freq_hz: float = 30.0,
    max_freq_hz: float = 4000.0,
    min_note_length_ms: float = 46.0,
    onset_threshold: float = 0.4,
    frame_threshold: float = 0.3,
    suppress_harmonics: bool = True,
) -> list[NoteEvent]:
    """Run Basic Pitch on a mono audio buffer via ONNX runtime.

    ``min_note_length_ms`` overrides Basic Pitch's 127.7 ms default, which is
    ~6 frames at 50 Hz and silently discards fast passages. We lower it to
    ~2 frames so quick notes survive into the timeline. ``onset_threshold`` is
    also relaxed slightly so soft fast onsets are not missed.

    When ``suppress_harmonics`` is True (default), notes whose pitch is at a
    harmonic-series interval above a simultaneously active louder note are
    removed.  This prevents strong piano overtones from being transcribed as
    spurious high-register notes.
    """
    note_events = _predict_note_events_onnx(
        audio,
        sr,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        min_note_length_ms=min_note_length_ms,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    # `note_events` is a list of (start, end, midi_pitch, velocity, pitch_bends_or_None).
    out: list[NoteEvent] = []
    for ev in note_events:
        start, end, pitch, velocity, *_ = ev
        out.append(
            NoteEvent(
                start_sec=float(start),
                end_sec=float(end),
                midi_pitch=float(pitch),
                velocity=float(velocity) / 127.0 if velocity > 1.0 else float(velocity),
            )
        )
    if suppress_harmonics:
        n_before = len(out)
        out = _suppress_harmonics(out)
        n_removed = n_before - len(out)
        if n_removed:
            log.debug("suppress_harmonics removed %d overtone events", n_removed)
    return out


def _suppress_harmonics(
    events: list[NoteEvent],
    tolerance_cents: float = 50.0,
    min_velocity_ratio: float = 0.8,
) -> list[NoteEvent]:
    """Remove notes that are likely overtones of a simultaneously active louder note.

    A note *Y* is suppressed when all of the following hold:

    * Another note *X* with lower MIDI pitch overlaps *Y* in time.
    * The pitch difference ``Y.midi_pitch - X.midi_pitch`` is within
      ``tolerance_cents`` of a harmonic-series interval (see ``_HARMONIC_ST``).
    * ``Y.velocity <= X.velocity * min_velocity_ratio`` — *Y* is notably
      weaker than the fundamental, as expected for an overtone artifact.

    The velocity guard prevents suppression of intentional octave chords where
    both notes are played at similar strength.
    """
    if len(events) < 2:
        return events
    tol_st = tolerance_cents / 100.0
    suppress: set[int] = set()
    for i, ev_high in enumerate(events):
        if i in suppress:
            continue
        for j, ev_low in enumerate(events):
            if j == i or j in suppress:
                continue
            diff = ev_high.midi_pitch - ev_low.midi_pitch
            if diff <= 0.0:
                continue
            # Must overlap in time.
            if ev_high.start_sec >= ev_low.end_sec or ev_low.start_sec >= ev_high.end_sec:
                continue
            # Check whether the pitch difference matches a harmonic interval.
            for h_st in _HARMONIC_ST:
                if abs(diff - h_st) <= tol_st:
                    if ev_high.velocity <= ev_low.velocity * min_velocity_ratio:
                        suppress.add(i)
                    break
    return [ev for k, ev in enumerate(events) if k not in suppress]


def _predict_note_events_onnx(
    audio: np.ndarray,
    sr: int,
    *,
    min_freq_hz: float,
    max_freq_hz: float,
    min_note_length_ms: float,
    onset_threshold: float,
    frame_threshold: float,
) -> list[tuple[float, float, int, float, list[int] | None]]:
    """Run Basic Pitch using ONNX runtime only (avoids TensorFlow import path)."""
    import onnxruntime as ort
    constants, infer, onnx_path = _load_basic_pitch_onnx_modules()

    ANNOTATIONS_FPS = int(constants.ANNOTATIONS_FPS)
    AUDIO_N_SAMPLES = int(constants.AUDIO_N_SAMPLES)
    AUDIO_SAMPLE_RATE = int(constants.AUDIO_SAMPLE_RATE)
    FFT_HOP = int(constants.FFT_HOP)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Basic Pitch ONNX model not found: {onnx_path}")

    # Match Basic Pitch behavior: force model sample rate and mono.
    audio32 = np.asarray(audio, dtype=np.float32)
    if sr != AUDIO_SAMPLE_RATE:
        audio32 = librosa.resample(audio32, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
    if audio32.ndim > 1:
        audio32 = np.mean(audio32, axis=-1, dtype=np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    # Prefix with half-overlap just like basic_pitch.inference.get_audio_input.
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio32])
    original_length = int(audio32.shape[0])

    output: dict[str, list[np.ndarray]] = {"note": [], "onset": [], "contour": []}

    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, pad_width=[[0, AUDIO_N_SAMPLES - len(window)]])
        x = np.expand_dims(np.expand_dims(window.astype(np.float32), axis=0), axis=-1)
        note, onset, contour = sess.run(
            [
                "StatefulPartitionedCall:1",
                "StatefulPartitionedCall:2",
                "StatefulPartitionedCall:0",
            ],
            {"serving_default_input_2:0": x},
        )
        output["note"].append(note)
        output["onset"].append(onset)
        output["contour"].append(contour)

    def _unwrap(arr3: np.ndarray) -> np.ndarray:
        n_olap = int(0.5 * n_overlapping_frames)
        if n_olap > 0:
            arr3 = arr3[:, n_olap:-n_olap, :]
        n_frames = int(np.floor(original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
        flat = arr3.reshape(arr3.shape[0] * arr3.shape[1], arr3.shape[2])
        return flat[:n_frames, :]

    model_output = {
        "note": _unwrap(np.concatenate(output["note"], axis=0)),
        "onset": _unwrap(np.concatenate(output["onset"], axis=0)),
        "contour": _unwrap(np.concatenate(output["contour"], axis=0)),
    }

    min_note_len = int(np.round(min_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    _midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        min_freq=min_freq_hz,
        max_freq=max_freq_hz,
        multiple_pitch_bends=False,
        melodia_trick=True,
        midi_tempo=120,
    )
    return note_events


def _load_basic_pitch_onnx_modules() -> tuple[types.ModuleType, types.ModuleType, Path]:
    """Load basic_pitch constants + note_creation without importing package __init__."""
    pkg_dir: Path | None = None
    existing_pkg = sys.modules.get("basic_pitch")
    if existing_pkg is not None:
        paths = getattr(existing_pkg, "__path__", None)
        if paths:
            pkg_dir = Path(next(iter(paths)))

    if pkg_dir is None:
        try:
            spec = importlib.util.find_spec("basic_pitch")
        except ValueError:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("basic_pitch package not found")
        pkg_dir = Path(spec.submodule_search_locations[0])
    constants_path = pkg_dir / "constants.py"
    note_creation_path = pkg_dir / "note_creation.py"
    onnx_path = pkg_dir / "saved_models" / "icassp_2022" / "nmp.onnx"

    # Create a lightweight package stub so `from basic_pitch.constants` resolves
    # without executing basic_pitch/__init__.py (which imports TensorFlow).
    if "basic_pitch" not in sys.modules:
        pkg = types.ModuleType("basic_pitch")
        pkg.__path__ = [str(pkg_dir)]
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            name="basic_pitch",
            loader=None,
            is_package=True,
        )
        if pkg.__spec__ is not None:
            pkg.__spec__.submodule_search_locations = [str(pkg_dir)]
        sys.modules["basic_pitch"] = pkg

    const_name = "basic_pitch.constants"
    note_name = "basic_pitch.note_creation"

    constants = sys.modules.get(const_name)
    if constants is None:
        c_spec = importlib.util.spec_from_file_location(const_name, constants_path)
        if c_spec is None or c_spec.loader is None:
            raise RuntimeError("Failed to load basic_pitch.constants")
        constants = importlib.util.module_from_spec(c_spec)
        sys.modules[const_name] = constants
        c_spec.loader.exec_module(constants)

    note_creation = sys.modules.get(note_name)
    if note_creation is None:
        n_spec = importlib.util.spec_from_file_location(note_name, note_creation_path)
        if n_spec is None or n_spec.loader is None:
            raise RuntimeError("Failed to load basic_pitch.note_creation")
        note_creation = importlib.util.module_from_spec(n_spec)
        sys.modules[note_name] = note_creation
        n_spec.loader.exec_module(note_creation)

    return constants, note_creation, onnx_path
