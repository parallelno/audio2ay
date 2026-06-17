"""End-to-end pipeline: instrumental audio → YM5 register stream."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import AY_CLOCK_HZ, FRAME_RATE_HZ
from .analysis.amplitude import AmplitudeFollower, loudness_envelope
from .analysis.audio_in import load_audio_stereo
from .analysis.drums import detect_drums
from .analysis.multipitch import transcribe
from .analysis.separation import separate
from .analysis.timeline import build_timeline
from .io.ym_writer import YmSong, write_ym5
from .synth.frame_builder import build_frame
from .synth.hw_envelope import EnvelopeController
from .synth.noise_mapper import drum_to_noise
from .synth.registers import RegisterFrame
from .synth.voice_scheduler import DualVoiceScheduler, VoiceScheduler

log = logging.getLogger(__name__)


@dataclass
class ConvertOptions:
    clock_hz: int = AY_CLOCK_HZ
    frame_rate_hz: int = FRAME_RATE_HZ
    # The hardware envelope generator is single-shared and currently only emits
    # a continuous tremolo on channel A, which overrides the bass note's real
    # dynamics. Software per-voice amplitude envelopes (see synth.volume_quantizer
    # .voice_gain) give better results, so the hardware envelope is opt-in.
    use_envelope: bool = False
    demucs_model: str = "htdemucs"
    skip_separation: bool = False  # Useful for tests / when input is a single stem.
    # Fill idle channels with slightly detuned unison copies of the active note
    # so sparse (mono/duo) material uses all 3 channels for a fuller tone.
    enrich_unison: bool = True
    enrich_detune_cents: float = 9.0
    # Loudness of the detuned unison copies on the *second* chip relative to a
    # full-scale voice. Chip B doubles chip A for harmonic richness, but at full
    # volume that doubling can dominate; lower this to push it into the
    # background. 1.0 = as loud as a real voice, 0.0 = silent (no chip-B doubling).
    enrich_volume_b: float = 0.75
    # Demucs can hallucinate faint percussion from tonal onsets. Drop the drum
    # track when the drums stem RMS is below this fraction of the mix RMS.
    drum_energy_floor: float = 0.06
    # Emit a second AY chip (TurboSound-style): 6 tone channels total. The extra
    # polyphony goes on chip B; chip A is unchanged. Optional, off by default.
    dual_chip: bool = False
    # Track the original mix's overall loudness so the render keeps the song's
    # macro-dynamics (quiet intros / loud climaxes) instead of sitting at a flat,
    # too-loud level. loudness_floor keeps quiet passages audible.
    match_loudness: bool = True
    loudness_floor: float = 0.4
    # AY square waves are harsher/brighter than most acoustic sources. brightness
    # < 1.0 gently attenuates high voices (per octave above middle C) to pull the
    # spectral centroid back down toward the original; 1.0 disables the tilt.
    brightness: float = 0.85


def convert_audio_to_ym(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    options: ConvertOptions | None = None,
) -> YmSong:
    """Convert an audio file to an uncompressed YM5 file. Returns the in-memory song.

    When ``out_path`` is ``None`` the YM file is not written; the in-memory
    :class:`YmSong` is just returned (useful for rendering straight to WAV).
    """
    opts = options or ConvertOptions()
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path is not None else None
    log.info("Loading %s", in_path)
    audio, sr = load_audio_stereo(in_path)
    duration_sec = audio.shape[-1] / sr

    # ------------------------------------------------------------ separation
    if opts.skip_separation:
        mono = audio.mean(axis=0).astype(np.float32) if audio.ndim > 1 else audio
        from .analysis.separation import Stems
        stems = Stems(
            drums=np.zeros_like(mono),
            bass=mono,
            other=mono,
            vocals=np.zeros_like(mono),
            sample_rate=sr,
        )
    else:
        log.info("Running source separation (Demucs %s)", opts.demucs_model)
        stems = separate(audio, sr, model_name=opts.demucs_model)

    # ------------------------------------------------------------ analysis
    log.info("Transcribing bass")
    bass_events = transcribe(stems.bass, stems.sample_rate, min_freq_hz=30.0, max_freq_hz=350.0)
    log.info("Transcribing other")
    other_events = transcribe(stems.other, stems.sample_rate, min_freq_hz=80.0, max_freq_hz=4000.0)
    log.info("Detecting drums")
    drum_hits = detect_drums(stems.drums, stems.sample_rate)
    # Demucs leaks tonal onsets into the drums stem on percussion-free material,
    # producing phantom snares. Drop the drum track when the drums stem is
    # negligible relative to the full mix.
    if drum_hits:
        drums_rms = float(np.sqrt(np.mean(stems.drums.astype(np.float32) ** 2) + 1e-12))
        mix = stems.drums + stems.bass + stems.other
        mix_rms = float(np.sqrt(np.mean(mix.astype(np.float32) ** 2) + 1e-12))
        rel = drums_rms / mix_rms
        if rel < opts.drum_energy_floor:
            log.info(
                "Drum stem negligible (%.1f%% of mix); dropping %d phantom hits",
                rel * 100.0,
                len(drum_hits),
            )
            drum_hits = []
    log.info(
        "Bass notes=%d other notes=%d drum hits=%d",
        len(bass_events),
        len(other_events),
        len(drum_hits),
    )

    # --------------------------------------------------------- timeline (50/100 Hz)
    # Amplitude followers give each note real attack/decay by tracking the
    # source stem's energy around its pitch (instead of a flat onset velocity).
    bass_follower = AmplitudeFollower(
        stems.bass, stems.sample_rate, frame_rate_hz=opts.frame_rate_hz
    )
    other_follower = AmplitudeFollower(
        stems.other, stems.sample_rate, frame_rate_hz=opts.frame_rate_hz
    )

    timeline = build_timeline(
        bass_events,
        other_events,
        drum_hits,
        duration_sec=duration_sec,
        frame_rate_hz=opts.frame_rate_hz,
        bass_follower=bass_follower,
        other_follower=other_follower,
    )

    # --------------------------------------------------------- synth
    env_ctrl = EnvelopeController()

    # Master loudness contour (one gain per frame) from the full mix, so the
    # render breathes with the original instead of running at a flat level.
    if opts.match_loudness:
        mix_mono = audio.mean(axis=0).astype(np.float32) if audio.ndim > 1 else audio
        master = loudness_envelope(
            mix_mono,
            sr,
            frame_rate_hz=opts.frame_rate_hz,
            n_frames=len(timeline),
            floor=opts.loudness_floor,
        )
    else:
        master = np.ones(len(timeline), dtype=np.float32)

    def _make_frame(
        assignment,
        env_ctrl_local: EnvelopeController,
        drum,
        enrich_seeds=None,
        enrich_volume_scale: float = 0.75,
        master_gain: float = 1.0,
    ) -> RegisterFrame:
        # Drum → noise channel state (chip A only).
        noise_period, noise_intensity = drum_to_noise(drum)
        assignment.drum_period = noise_period
        assignment.drum_intensity = noise_intensity
        envelope = (
            env_ctrl_local.step(assignment.a)
            if opts.use_envelope
            else env_ctrl_local.step(None)
        )
        return build_frame(
            assignment,
            envelope,
            clock_hz=opts.clock_hz,
            enrich_unison=opts.enrich_unison,
            enrich_detune_cents=opts.enrich_detune_cents,
            enrich_volume_scale=enrich_volume_scale,
            enrich_seeds=enrich_seeds,
            master_gain=master_gain,
            brightness=opts.brightness,
        )

    frames: list[RegisterFrame] = []
    frames_b: list[RegisterFrame] | None = None

    if opts.dual_chip:
        scheduler = DualVoiceScheduler()
        env_ctrl_b = EnvelopeController()
        frames_b = []
        for fi, tf in enumerate(timeline):
            assign_a, assign_b = scheduler.assign(tf)
            frames.append(_make_frame(assign_a, env_ctrl, tf.drum, master_gain=master[fi]))
            # Chip B doubles whatever chip A is voicing (slightly detuned) on any
            # channel it isn't already using, so its three channels never sit
            # idle while there is sound to thicken.
            seeds_b = [v for v in assign_a.to_list() if v is not None and v.velocity > 0.0]
            frames_b.append(
                _make_frame(
                    assign_b,
                    env_ctrl_b,
                    None,
                    enrich_seeds=seeds_b,
                    enrich_volume_scale=opts.enrich_volume_b,
                    master_gain=master[fi],
                )
            )
    else:
        scheduler = VoiceScheduler()
        for fi, tf in enumerate(timeline):
            assignment = scheduler.assign(tf)
            frames.append(_make_frame(assignment, env_ctrl, tf.drum, master_gain=master[fi]))

    song = YmSong(
        frames=frames,
        song_name=in_path.stem,
        author="audio2ay",
        comment=f"Generated from {in_path.name}",
        clock_hz=opts.clock_hz,
        frame_rate_hz=opts.frame_rate_hz,
        frames_b=frames_b,
    )
    if out_path is not None:
        write_ym5(out_path, song, interleaved=True)
        log.info(
            "Wrote %s (%d frames%s)",
            out_path,
            len(frames),
            ", dual-AY" if opts.dual_chip else "",
        )
    return song
