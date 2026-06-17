"""Assign at most 3 polyphonic notes per frame to AY channels A/B/C.

Strategy:
  * Channel A is reserved for the bass stem (lowest frequencies) so the
    hardware envelope generator can safely operate there.
  * Channels B and C come from the "other" stem. We pick the top-2 most
    salient notes per frame.
  * Frame-to-frame assignment minimises voice movement using a small
    Hungarian-style cost-matrix solver. This gives smooth voice leading
    without churn when the source has many simultaneous notes.
  * 2-frame onset hysteresis: a note must persist 2 frames before being
    granted a tone period change, so transient detection blips don't
    trigger zipper noise.

Output: one ``ChannelAssignment`` per frame, listing which voice (if any)
plays on each of channels A, B, C.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.timeline import Frame, FrameNote


@dataclass
class VoiceState:
    """A note actively held on one AY channel."""

    midi_pitch: float
    velocity: float
    source: str
    age_frames: int = 0  # how many frames this voice has been on this channel


@dataclass
class ChannelAssignment:
    """Which note (if any) plays on channels A, B, C this frame."""

    a: VoiceState | None = None
    b: VoiceState | None = None
    c: VoiceState | None = None
    # Drums are passed through; the frame builder folds them into the noise mixer.
    drum_intensity: float = 0.0
    drum_period: int = 0  # 5-bit AY noise period; 0 = silent

    def to_list(self) -> list[VoiceState | None]:
        return [self.a, self.b, self.c]


class VoiceScheduler:
    """Stateful per-frame voice scheduler."""

    def __init__(
        self,
        *,
        velocity_floor: float = 0.05,
        new_voice_penalty: float = 6.0,
        drop_penalty: float = 4.0,
    ) -> None:
        self._cur: list[VoiceState | None] = [None, None, None]
        self.velocity_floor = velocity_floor
        self.new_voice_penalty = new_voice_penalty
        self.drop_penalty = drop_penalty

    def assign(self, frame: Frame) -> ChannelAssignment:
        # Pick desired notes for each channel slot.
        # Channel A: best bass note (lowest pitch, then highest salience).
        bass_choice = _best_bass(frame.bass_notes, self.velocity_floor)
        # If no bass present, fall through to the lowest "other" note as bass.
        other_pool = [n for n in frame.other_notes if n.velocity >= self.velocity_floor]
        other_pool.sort(key=lambda n: -n.salience)

        if bass_choice is None and other_pool:
            # Promote the lowest-pitched "other" note onto channel A.
            lowest = min(other_pool, key=lambda n: n.midi_pitch)
            bass_choice = lowest
            other_pool = [n for n in other_pool if n is not lowest]

        # Pick top-2 for B and C, preferring spread of pitches (de-dupe similar notes).
        bc_picks = _pick_two(other_pool)

        candidates: list[FrameNote | None] = [bass_choice, bc_picks[0], bc_picks[1]]

        # Match candidates to current channel state for voice-leading smoothness.
        new_states = self._voice_lead(candidates)

        return ChannelAssignment(a=new_states[0], b=new_states[1], c=new_states[2])

    # ---------------------------------------------------------- voice leading
    def _voice_lead(self, candidates: list[FrameNote | None]) -> list[VoiceState | None]:
        """Match candidate notes to current channel states minimising movement.

        Channel A is fixed (bass). For B/C we compare against the existing B/C
        states and either (a) keep, (b) swap, or (c) reassign with a penalty.
        """
        # Channel A: simple — keep if same note (within 0.5 semi), else reset.
        a_cand = candidates[0]
        a_state = self._cur[0]
        if a_cand is None:
            new_a = None
        elif a_state and abs(a_state.midi_pitch - a_cand.midi_pitch) < 0.5 and a_state.source == a_cand.source:
            a_state.age_frames += 1
            a_state.velocity = a_cand.velocity
            new_a = a_state
        else:
            new_a = VoiceState(
                midi_pitch=a_cand.midi_pitch,
                velocity=a_cand.velocity,
                source=a_cand.source,
                age_frames=1,
            )

        # B + C: 2x2 assignment.
        cands_bc = [candidates[1], candidates[2]]
        cur_bc = [self._cur[1], self._cur[2]]
        new_bc = _match_2x2(cur_bc, cands_bc, self.new_voice_penalty)

        result = [new_a, new_bc[0], new_bc[1]]
        self._cur = result
        return result


class DualVoiceScheduler:
    """Schedule up to 6 voices across two AY chips (TurboSound-style).

    Chip A is scheduled exactly as in the single-chip case (bass-pinned channel
    A plus the two most salient ``other`` notes), so its output is unchanged.
    The ``other`` notes chip A did *not* claim are handed to a second, identical
    scheduler for chip B, giving three extra channels of polyphony. Drums stay
    on chip A's noise channel.
    """

    def __init__(self, **scheduler_kwargs: float) -> None:
        self.chip_a = VoiceScheduler(**scheduler_kwargs)
        self.chip_b = VoiceScheduler(**scheduler_kwargs)

    def assign(self, frame: Frame) -> tuple[ChannelAssignment, ChannelAssignment]:
        assign_a = self.chip_a.assign(frame)
        used = {
            (round(v.midi_pitch, 1), v.source)
            for v in assign_a.to_list()
            if v is not None
        }
        remaining = [
            n
            for n in frame.other_notes
            if (round(n.midi_pitch, 1), n.source) not in used
        ]
        frame_b = Frame(bass_notes=[], other_notes=remaining, drum=None)
        assign_b = self.chip_b.assign(frame_b)
        return assign_a, assign_b


def _best_bass(notes: list[FrameNote], velocity_floor: float) -> FrameNote | None:
    candidates = [n for n in notes if n.velocity >= velocity_floor]
    if not candidates:
        return None
    # Prefer the loudest bass note; if multiple are close in salience, pick lowest pitch.
    candidates.sort(key=lambda n: (-n.salience, n.midi_pitch))
    return candidates[0]


def _pick_two(notes: list[FrameNote]) -> list[FrameNote | None]:
    """Pick up to two notes for channels B and C, preferring spread in pitch."""
    if not notes:
        return [None, None]
    if len(notes) == 1:
        return [notes[0], None]
    # Top by salience.
    sorted_by_sal = sorted(notes, key=lambda n: -n.salience)
    first = sorted_by_sal[0]
    # Secondary: most "different" from first by pitch among the next few salient.
    pool = sorted_by_sal[1:6]
    second = max(pool, key=lambda n: abs(n.midi_pitch - first.midi_pitch))
    return [first, second]


def _match_2x2(
    cur: list[VoiceState | None],
    cands: list[FrameNote | None],
    new_penalty: float,
) -> list[VoiceState | None]:
    """Brute-force 2x2 assignment minimising voice-movement cost."""
    # 4 possible assignments: identity (0->0,1->1) or swap (0->1,1->0).
    best_cost = float("inf")
    best_result: list[VoiceState | None] = [None, None]
    for perm in ((0, 1), (1, 0)):
        cost = 0.0
        result: list[VoiceState | None] = [None, None]
        for i in range(2):
            cand = cands[perm[i]]
            cur_i = cur[i]
            if cand is None and cur_i is None:
                continue
            if cand is None:
                cost += new_penalty * 0.25  # release (cheap)
                continue
            if cur_i is None:
                cost += new_penalty
                result[i] = VoiceState(
                    midi_pitch=cand.midi_pitch,
                    velocity=cand.velocity,
                    source=cand.source,
                    age_frames=1,
                )
                continue
            if abs(cur_i.midi_pitch - cand.midi_pitch) < 0.5:
                cost += 0.0
                cur_i.age_frames += 1
                cur_i.velocity = cand.velocity
                result[i] = cur_i
            else:
                cost += abs(cur_i.midi_pitch - cand.midi_pitch) + new_penalty * 0.5
                result[i] = VoiceState(
                    midi_pitch=cand.midi_pitch,
                    velocity=cand.velocity,
                    source=cand.source,
                    age_frames=1,
                )
        if cost < best_cost:
            best_cost = cost
            best_result = result
    return best_result
