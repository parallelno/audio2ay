"""Merge tonal note events + drum hits onto the 50 Hz frame grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import FRAME_RATE_HZ
from .amplitude import AmplitudeFollower
from .drums import DrumClass, DrumHit
from .multipitch import NoteEvent


@dataclass
class FrameNote:
    """A tonal note that is sounding (or starting) on a particular frame."""

    midi_pitch: float
    velocity: float
    is_onset: bool
    salience: float
    source: str  # "bass" | "other"


@dataclass
class FrameDrum:
    drum: DrumClass
    intensity: float
    is_onset: bool


@dataclass
class Frame:
    """All sound events active on a single 20 ms frame."""

    bass_notes: list[FrameNote] = field(default_factory=list)
    other_notes: list[FrameNote] = field(default_factory=list)
    drum: FrameDrum | None = None


def build_timeline(
    bass_events: list[NoteEvent],
    other_events: list[NoteEvent],
    drum_hits: list[DrumHit],
    duration_sec: float,
    *,
    frame_rate_hz: int = FRAME_RATE_HZ,
    drum_decay_frames: int = 4,
    bass_follower: AmplitudeFollower | None = None,
    other_follower: AmplitudeFollower | None = None,
) -> list[Frame]:
    """Merge transcribed events onto the 50 Hz frame grid for the given duration.

    Notes that span multiple frames remain "active" on each covered frame; the
    first frame they touch is marked `is_onset=True`.

    If an :class:`AmplitudeFollower` is supplied for a stem, each note's
    per-frame velocity tracks the source energy around its pitch (normalised to
    the note's own onset), giving real attack/decay instead of a flat sustain.

    Drum hits decay across `drum_decay_frames` so the noise channel has natural
    tails on the AY (which has no per-channel envelope reset for noise).
    """
    n_frames = int(round(duration_sec * frame_rate_hz)) + 1
    frames = [Frame() for _ in range(n_frames)]

    def _place_notes(
        events: list[NoteEvent],
        bucket_attr: str,
        source: str,
        follower: AmplitudeFollower | None,
    ) -> None:
        use_follower = follower is not None and follower.available
        for ev in events:
            f0 = max(0, int(round(ev.start_sec * frame_rate_hz)))
            f1 = min(n_frames - 1, int(round(ev.end_sec * frame_rate_hz)))
            if f1 < f0:
                f1 = f0
            # Reference energy at the note's onset for self-normalised dynamics.
            base = follower.band(f0, ev.midi_pitch) if use_follower else 0.0
            track = use_follower and base > 1e-6
            for fi in range(f0, f1 + 1):
                if track:
                    ratio = follower.band(fi, ev.midi_pitch) / base
                    vel = float(np.clip(ev.velocity * ratio, 0.0, 1.0))
                else:
                    vel = ev.velocity
                getattr(frames[fi], bucket_attr).append(
                    FrameNote(
                        midi_pitch=ev.midi_pitch,
                        velocity=vel,
                        is_onset=(fi == f0),
                        salience=ev.salience * vel,
                        source=source,
                    )
                )

    _place_notes(bass_events, "bass_notes", "bass", bass_follower)
    _place_notes(other_events, "other_notes", "other", other_follower)

    # Drums: place onset on the nearest frame, decay intensity across following frames.
    for hit in drum_hits:
        f0 = int(round(hit.time_sec * frame_rate_hz))
        if f0 < 0 or f0 >= n_frames:
            continue
        for k in range(drum_decay_frames):
            fi = f0 + k
            if fi >= n_frames:
                break
            decay = max(0.0, 1.0 - k / max(drum_decay_frames - 1, 1))
            new_intensity = hit.intensity * decay
            existing = frames[fi].drum
            if existing is None or new_intensity > existing.intensity:
                frames[fi].drum = FrameDrum(
                    drum=hit.drum,
                    intensity=new_intensity,
                    is_onset=(k == 0),
                )

    return frames
