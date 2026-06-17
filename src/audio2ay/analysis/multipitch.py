"""Multi-pitch transcription for tonal stems via Spotify Basic Pitch.

Returns a list of note events with start/end times (seconds), MIDI pitch, and
a per-note normalised velocity in [0, 1].

Falls back to a librosa-based monophonic-ish fallback (single dominant pitch
per frame) if Basic Pitch isn't installed; that mode is good enough for bass
lines and unit-test scenarios.
"""

from __future__ import annotations

import importlib.util
import importlib.machinery
import logging
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


@dataclass
class NoteEvent:
    """One transcribed note.

    Fields are intentionally minimal so events can come from either Basic Pitch
    (polyphonic, with pitch bends collapsed to the median) or the fallback
    monophonic detector.
    """

    start_sec: float
    end_sec: float
    midi_pitch: float           # may be fractional if pitch bends are tracked
    velocity: float             # 0..1
    salience: float = 1.0       # extra weight used by the voice scheduler

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def transcribe(
    audio: np.ndarray,
    sr: int,
    *,
    min_freq_hz: float = 30.0,
    max_freq_hz: float = 4000.0,
    min_note_length_ms: float = 46.0,
    onset_threshold: float = 0.4,
    frame_threshold: float = 0.3,
) -> list[NoteEvent]:
    """Run Basic Pitch (or fallback) on a mono audio buffer.

    ``min_note_length_ms`` overrides Basic Pitch's 127.7 ms default, which is
    ~6 frames at 50 Hz and silently discards fast passages. We lower it to
    ~2 frames so quick notes survive into the timeline. ``onset_threshold`` is
    also relaxed slightly so soft fast onsets are not missed.
    """
    try:
        note_events = _predict_note_events_onnx(
            audio,
            sr,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
            min_note_length_ms=min_note_length_ms,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )
    except Exception as onnx_exc:  # pragma: no cover - optional dep path
        log.warning(
            "ONNX transcription path unavailable (%s); using librosa monophonic fallback.",
            onnx_exc,
        )
        return _fallback_monophonic(audio, sr, min_freq_hz, max_freq_hz)
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
    return out


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


def _fallback_monophonic(
    audio: np.ndarray, sr: int, min_freq: float, max_freq: float
) -> list[NoteEvent]:
    """Fallback when Basic Pitch is unavailable: single-pitch tracker via librosa.pyin."""
    import librosa

    if audio.size == 0:
        return []
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=max(min_freq, 30.0),
        fmax=min(max_freq, 1500.0),
        sr=sr,
        frame_length=2048,
    )
    # Convert frame-by-frame to NoteEvents by grouping consecutive voiced frames
    # with a similar pitch.
    times = librosa.times_like(f0, sr=sr)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    # Trim/pad rms to match f0 length.
    if len(rms) > len(f0):
        rms = rms[: len(f0)]
    elif len(rms) < len(f0):
        rms = np.pad(rms, (0, len(f0) - len(rms)))

    events: list[NoteEvent] = []
    current_pitch = None
    current_start = None
    current_rms_sum = 0.0
    current_count = 0

    for i, (t, f, voiced) in enumerate(zip(times, f0, voiced_flag)):
        if not voiced or f is None or np.isnan(f) or f <= 0:
            if current_pitch is not None:
                events.append(
                    NoteEvent(
                        start_sec=float(current_start),
                        end_sec=float(t),
                        midi_pitch=float(current_pitch),
                        velocity=float(min(1.0, current_rms_sum / max(current_count, 1) * 4)),
                    )
                )
                current_pitch = None
            continue
        midi = librosa.hz_to_midi(f)
        if current_pitch is None:
            current_pitch = midi
            current_start = t
            current_rms_sum = rms[i]
            current_count = 1
        elif abs(midi - current_pitch) > 0.5:
            events.append(
                NoteEvent(
                    start_sec=float(current_start),
                    end_sec=float(t),
                    midi_pitch=float(current_pitch),
                    velocity=float(min(1.0, current_rms_sum / max(current_count, 1) * 4)),
                )
            )
            current_pitch = midi
            current_start = t
            current_rms_sum = rms[i]
            current_count = 1
        else:
            # Smooth update
            current_pitch = 0.7 * current_pitch + 0.3 * midi
            current_rms_sum += rms[i]
            current_count += 1

    if current_pitch is not None:
        events.append(
            NoteEvent(
                start_sec=float(current_start),
                end_sec=float(times[-1]),
                midi_pitch=float(current_pitch),
                velocity=float(min(1.0, current_rms_sum / max(current_count, 1) * 4)),
            )
        )
    return events
