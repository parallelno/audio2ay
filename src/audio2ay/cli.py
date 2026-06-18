"""Command-line interface for audio2ay."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Always show our own pipeline progress at the chosen level.
    logging.getLogger("audio2ay").setLevel(level)

    if verbose:
        return

    # Quiet noisy third-party output (TensorFlow / Basic Pitch / absl) so the
    # default convert log stays readable. Use --verbose to see all of it.
    # TF_CPP_MIN_LOG_LEVEL must be set before TensorFlow is imported (Basic
    # Pitch imports it lazily, after this runs).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "3")
    # Raise the root logger so Basic Pitch's root-level "Coremltools/tflite/
    # onnxruntime is not installed" warnings are suppressed. Our records use the
    # "audio2ay" logger and still propagate to the root handler.
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("tensorflow", "absl", "h5py", "numba"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _write_preview_audio(path: Path, audio, sample_rate: int) -> None:
    import soundfile as sf

    ext = path.suffix.lower()
    if ext == ".mp3":
        # libsndfile MP3 encoding can be unstable on some Windows setups.
        # Encode via ffmpeg from a temporary WAV for reliability.
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            sf.write(str(tmp_path), audio, sample_rate, subtype="PCM_16")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_path),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is required for MP3 output but was not found in PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg failed while encoding MP3: {exc.stderr.strip()}"
            ) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return

    subtype = "PCM_16" if ext in (".wav", ".wave") else None
    if subtype is None:
        sf.write(str(path), audio, sample_rate)
    else:
        sf.write(str(path), audio, sample_rate, subtype=subtype)


def cmd_convert(args: argparse.Namespace) -> int:
    from .pipeline import ConvertOptions, convert_audio_to_ym

    frame_rate_hz = 100 if args.hz100 else getattr(args, "frame_rate", 50)

    options = ConvertOptions(
        demucs_model=args.demucs_model,
        frame_rate_hz=frame_rate_hz,
        use_envelope=args.envelope,
        enrich_unison=not args.no_enrich,
        enrich_detune_cents=args.detune_cents,
        enrich_volume_b=args.enrich_volume_b,
        match_loudness=not args.no_loudness_match,
        brightness=args.brightness,
        dual_chip=args.dual_chip,
    )
    convert_audio_to_ym(args.input, args.output, options=options)
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Convert audio straight to an audio file (via emulator), skipping .ym."""

    # Native crashes in third-party ML stacks (Demucs / TensorFlow deps) are
    # process-fatal on Windows. Run preview logic in a worker subprocess so the
    # parent command can retry instead of dying immediately.
    if os.environ.get("A2A_PREVIEW_WORKER") != "1":
        cmd = [sys.executable, "-m", "audio2ay.cli", "preview", args.input]
        if args.output:
            cmd.append(args.output)
        cmd.extend(["--sample-rate", str(args.sample_rate)])
        cmd.extend(["--demucs-model", str(args.demucs_model)])
        if args.envelope:
            cmd.append("--envelope")
        if args.no_enrich:
            cmd.append("--no-enrich")
        cmd.extend(["--detune-cents", str(args.detune_cents)])
        cmd.extend(["--enrich-volume-b", str(args.enrich_volume_b)])
        if args.no_loudness_match:
            cmd.append("--no-loudness-match")
        cmd.extend(["--brightness", str(args.brightness)])
        if args.dual_chip:
            cmd.append("--dual-chip")
        cmd.extend(["--frame-rate", str(getattr(args, "frame_rate", 50))])
        if args.hz100:
            cmd.append("--100hz")
        cmd.extend(["--pulse-width", str(args.pulse_width)])
        env = os.environ.copy()
        env["A2A_PREVIEW_WORKER"] = "1"

        last_code = 1
        for attempt in range(1, 4):
            try:
                proc = subprocess.run(cmd, env=env, timeout=180)
                rc = int(proc.returncode)
            except subprocess.TimeoutExpired:
                rc = 124
            if rc == 0:
                return 0
            last_code = int(rc)
            logging.getLogger("audio2ay").warning(
                "Preview worker exited with code %s (attempt %d/3), retrying...",
                rc,
                attempt,
            )

        return last_code

    from .io.ay_emulator import render_song_to_array
    from .pipeline import ConvertOptions, convert_audio_to_ym

    frame_rate_hz = 100 if args.hz100 else getattr(args, "frame_rate", 50)

    options = ConvertOptions(
        demucs_model=args.demucs_model,
        frame_rate_hz=frame_rate_hz,
        use_envelope=args.envelope,
        enrich_unison=not args.no_enrich,
        enrich_detune_cents=args.detune_cents,
        enrich_volume_b=args.enrich_volume_b,
        match_loudness=not args.no_loudness_match,
        brightness=args.brightness,
        dual_chip=args.dual_chip,
    )
    song = convert_audio_to_ym(args.input, None, options=options)
    audio = render_song_to_array(song, sample_rate=args.sample_rate,
                                 pulse_width=args.pulse_width)
    # Default preview output to MP3 when not explicitly provided.
    out_path = Path(args.output) if args.output else Path("build") / f"{Path(args.input).stem}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_preview_audio(out_path, audio, args.sample_rate)
    print(f"Wrote {out_path}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .io.ay_emulator import render_song_to_array, read_ym5
    import soundfile as sf

    song = read_ym5(args.input)
    audio = render_song_to_array(song, sample_rate=args.sample_rate,
                                 pulse_width=args.pulse_width)
    # Create parent directories if needed
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, args.sample_rate, subtype="PCM_16")
    print(f"Rendered {args.input} -> {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """One-shot: convert + render side-by-side to a directory."""

    import librosa
    import soundfile as sf
    from .io.ay_emulator import render_song_to_array, read_ym5
    from .pipeline import ConvertOptions, convert_audio_to_ym

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ym_path = out_dir / (in_path.stem + ".ym")
    ay_wav = out_dir / (in_path.stem + ".ay.wav")
    orig_wav = out_dir / (in_path.stem + ".orig.wav")

    frame_rate_hz = 100 if args.hz100 else getattr(args, "frame_rate", 50)

    options = ConvertOptions(
        frame_rate_hz=frame_rate_hz,
        use_envelope=args.envelope,
        enrich_unison=not args.no_enrich,
        enrich_detune_cents=args.detune_cents,
        enrich_volume_b=args.enrich_volume_b,
        match_loudness=not args.no_loudness_match,
        brightness=args.brightness,
        dual_chip=args.dual_chip,
    )
    convert_audio_to_ym(in_path, ym_path, options=options)

    # Render with custom pulse width if specified
    song = read_ym5(ym_path)
    audio = render_song_to_array(song, sample_rate=44100,
                                 pulse_width=args.pulse_width)
    sf.write(str(ay_wav), audio, 44100, subtype="PCM_16")

    # Convenience: also drop a wav of the original at 44.1k mono for A/B-ing.
    orig, sr = librosa.load(str(in_path), sr=44100, mono=True)
    sf.write(str(orig_wav), orig, 44100, subtype="PCM_16")

    print(f"Wrote: {ym_path}")
    print(f"       {ay_wav}")
    print(f"       {orig_wav}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audio2ay", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Audio file → .ym register stream")
    p_conv.add_argument("input")
    p_conv.add_argument("output")
    p_conv.add_argument("--demucs-model", default="htdemucs")
    p_conv.add_argument("--envelope", action="store_true",
                        help="Run hardware envelope at note pitch for sawtooth bass timbre on channel A")
    p_conv.add_argument("--no-enrich", action="store_true",
                        help="Disable filling idle channels with detuned unison copies")
    p_conv.add_argument("--detune-cents", type=float, default=9.0,
                        help="Unison detune spread for idle-channel enrichment (cents)")
    p_conv.add_argument("--enrich-volume-b", type=float, default=0.75,
                        help="Loudness of chip B's unison doubling (0=silent, 1=full); "
                             "lower it if the second chip is too prominent")
    p_conv.add_argument("--no-loudness-match", action="store_true",
                        help="Don't track the original mix's loudness contour "
                             "(keep a flat output level)")
    p_conv.add_argument("--brightness", type=float, default=0.85,
                        help="Per-octave high-voice attenuation to tame square-wave "
                             "harshness (1.0=off, lower=darker)")
    p_conv.add_argument("--dual-chip", action="store_true",
                        help="Emit two AY chips (TurboSound): 6 tone channels total")
    p_conv.add_argument("--frame-rate", type=int, choices=(50, 100), default=50,
                        help="YM register update rate in Hz (50=classic, 100=finer timing)")
    p_conv.add_argument("--100hz", dest="hz100", action="store_true",
                        help="Shortcut for --frame-rate 100")
    p_conv.set_defaults(func=cmd_convert)

    p_prev = sub.add_parser(
        "preview",
        help="Audio file → audio directly (convert + render, no .ym written)",
    )
    p_prev.add_argument("input")
    p_prev.add_argument("output", nargs="?",
                        help="Output path (default: build/<input-stem>.mp3)")
    p_prev.add_argument("--sample-rate", type=int, default=44100)
    p_prev.add_argument("--demucs-model", default="htdemucs")
    p_prev.add_argument("--envelope", action="store_true",
                        help="Run hardware envelope at note pitch for sawtooth bass timbre on channel A")
    p_prev.add_argument("--no-enrich", action="store_true",
                        help="Disable filling idle channels with detuned unison copies")
    p_prev.add_argument("--detune-cents", type=float, default=9.0,
                        help="Unison detune spread for idle-channel enrichment (cents)")
    p_prev.add_argument("--enrich-volume-b", type=float, default=0.75,
                        help="Loudness of chip B's unison doubling (0=silent, 1=full); "
                             "lower it if the second chip is too prominent")
    p_prev.add_argument("--no-loudness-match", action="store_true",
                        help="Don't track the original mix's loudness contour "
                             "(keep a flat output level)")
    p_prev.add_argument("--brightness", type=float, default=0.85,
                        help="Per-octave high-voice attenuation to tame square-wave "
                             "harshness (1.0=off, lower=darker)")
    p_prev.add_argument("--dual-chip", action="store_true",
                        help="Emit two AY chips (TurboSound): 6 tone channels total")
    p_prev.add_argument("--frame-rate", type=int, choices=(50, 100), default=50,
                        help="YM register update rate in Hz (50=classic, 100=finer timing)")
    p_prev.add_argument("--100hz", dest="hz100", action="store_true",
                        help="Shortcut for --frame-rate 100")
    p_prev.add_argument("--pulse-width", type=float, default=0.7,
                        help="Pulse duty cycle (0.5=square/harsh, 0.7=default, 0.75=wider/darker)")
    p_prev.set_defaults(func=cmd_preview)

    p_rend = sub.add_parser("render", help=".ym → .wav via the bundled emulator")
    p_rend.add_argument("input")
    p_rend.add_argument("output")
    p_rend.add_argument("--sample-rate", type=int, default=44100)
    p_rend.add_argument("--pulse-width", type=float, default=0.7,
                        help="Pulse duty cycle (0.5=square/harsh, 0.7=default, 0.75=wider/darker)")
    p_rend.set_defaults(func=cmd_render)

    p_val = sub.add_parser("validate", help="Convert + render + dump side-by-side WAVs")
    p_val.add_argument("input")
    p_val.add_argument("--outdir", default="build")
    p_val.add_argument("--envelope", action="store_true",
                       help="Run hardware envelope at note pitch for sawtooth bass timbre on channel A")
    p_val.add_argument("--no-enrich", action="store_true",
                       help="Disable filling idle channels with detuned unison copies")
    p_val.add_argument("--detune-cents", type=float, default=9.0,
                       help="Unison detune spread for idle-channel enrichment (cents)")
    p_val.add_argument("--enrich-volume-b", type=float, default=0.75,
                       help="Loudness of chip B's unison doubling (0=silent, 1=full); "
                            "lower it if the second chip is too prominent")
    p_val.add_argument("--no-loudness-match", action="store_true",
                       help="Don't track the original mix's loudness contour "
                            "(keep a flat output level)")
    p_val.add_argument("--brightness", type=float, default=0.85,
                       help="Per-octave high-voice attenuation to tame square-wave "
                            "harshness (1.0=off, lower=darker)")
    p_val.add_argument("--dual-chip", action="store_true",
                       help="Emit two AY chips (TurboSound): 6 tone channels total")
    p_val.add_argument("--frame-rate", type=int, choices=(50, 100), default=50,
                       help="YM register update rate in Hz (50=classic, 100=finer timing)")
    p_val.add_argument("--100hz", dest="hz100", action="store_true",
                       help="Shortcut for --frame-rate 100")
    p_val.add_argument("--pulse-width", type=float, default=0.7,
                       help="Pulse duty cycle (0.5=square/harsh, 0.7=default, 0.75=wider/darker)")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
