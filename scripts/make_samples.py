"""Generate / fetch a small set of instrumental audio clips for testing.

Produces a `samples/` folder next to the project root with a mix of:

* **synthetic** clips (always created, no network needed) covering the cases
  the converter cares about: monophonic, polyphonic, with drums.
* **public-domain** clips from Wikimedia Commons, fetched on best-effort and
  silently skipped if the network is unreachable.

Run from the project root:

    python scripts/make_samples.py
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


# ---------------------------------------------------------------- synthesis
def _adsr(n: int, attack: float = 0.01, release: float = 0.05) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(SR * attack))
    r = max(1, int(SR * release))
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    env[-r:] = np.linspace(1, 0, r, dtype=np.float32)
    return env


def _sine(freq: float, dur: float, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(dur * SR), dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t).astype(np.float32)) * _adsr(len(t))


def _square(freq: float, dur: float, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(dur * SR), dtype=np.float32) / SR
    return (amp * np.sign(np.sin(2 * np.pi * freq * t)).astype(np.float32)) * _adsr(len(t))


def _kick(dur: float = 0.15) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    # Quick pitch sweep 120 → 40 Hz with exponential decay.
    f = 120 * np.exp(-t * 12) + 40
    phase = 2 * np.pi * np.cumsum(f) / SR
    sig = np.sin(phase) * np.exp(-t * 8)
    return (0.7 * sig).astype(np.float32)


def _snare(dur: float = 0.18) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 18)
    body = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32) * np.exp(-t * 25)
    return (0.5 * noise + body).astype(np.float32)


def _hihat(dur: float = 0.06) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(7)
    return (0.3 * rng.standard_normal(n).astype(np.float32) * np.exp(-t * 60)).astype(np.float32)


# Note name (MIDI) helper.
def _midi_hz(midi: float) -> float:
    return float(440.0 * 2 ** ((midi - 69) / 12.0))


# ------------------------------------------------------------------ scenes
def make_arpeggio() -> np.ndarray:
    """2 s, monophonic C-major arpeggio (C4 E4 G4 C5)."""
    notes = [60, 64, 67, 72, 67, 64, 60, 64]
    out = np.zeros(int(2.0 * SR), dtype=np.float32)
    step = len(out) // len(notes)
    for i, m in enumerate(notes):
        clip = _sine(_midi_hz(m), step / SR, amp=0.4)
        if len(clip) > step:
            clip = clip[:step]
        out[i * step : i * step + len(clip)] += clip
    return out


def make_bass_and_lead() -> np.ndarray:
    """3 s, polyphonic: A2 square bass under an A4 sine lead."""
    bass = _square(_midi_hz(45), 3.0, amp=0.3)  # A2
    lead = _sine(_midi_hz(69), 3.0, amp=0.3)    # A4
    return (bass + lead).astype(np.float32)


def make_drum_loop() -> np.ndarray:
    """4 s, simple kick/snare/hat groove with a square bass + sine lead."""
    out = np.zeros(int(4.0 * SR), dtype=np.float32)
    bpm = 120
    beat = SR * 60 // bpm  # samples per quarter note

    # Drums: kick on 1 & 3, snare on 2 & 4, hat on every 8th.
    for bar in range(2):
        b0 = bar * beat * 4
        for k in (0, 2):
            i = b0 + k * beat
            out[i : i + len(_kick())] += _kick()[: max(0, len(out) - i)]
        for s in (1, 3):
            i = b0 + s * beat
            seg = _snare()
            end = min(len(out), i + len(seg))
            out[i:end] += seg[: end - i]
        for h in range(0, 8):
            i = b0 + h * (beat // 2)
            seg = _hihat()
            end = min(len(out), i + len(seg))
            out[i:end] += 0.6 * seg[: end - i]

    # Bass line: A2, A2, E2, A2 every quarter note (root + V).
    bass_notes = [45, 45, 40, 45, 45, 45, 40, 45]
    for i, m in enumerate(bass_notes):
        seg = _square(_midi_hz(m), beat / SR, amp=0.25)
        s = i * beat
        end = min(len(out), s + len(seg))
        out[s:end] += seg[: end - s]

    # Lead: an arpeggio above
    lead_notes = [69, 73, 76, 81, 76, 73, 69, 73]  # A4 C#5 E5 A5 ...
    for i, m in enumerate(lead_notes):
        seg = _sine(_midi_hz(m), beat / SR, amp=0.2)
        s = i * beat
        end = min(len(out), s + len(seg))
        out[s:end] += seg[: end - s]

    # Normalise.
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.9).astype(np.float32)


def make_chord_progression() -> np.ndarray:
    """4 s, I-V-vi-IV in C major as 3-voice sustained chords (polyphony stress)."""
    chords = [
        (60, 64, 67),  # C
        (55, 59, 62),  # G
        (57, 60, 64),  # Am
        (53, 57, 60),  # F
    ]
    out = np.zeros(int(4.0 * SR), dtype=np.float32)
    step = len(out) // len(chords)
    for i, (a, b, c) in enumerate(chords):
        clip = (
            _sine(_midi_hz(a), step / SR, amp=0.25)
            + _sine(_midi_hz(b), step / SR, amp=0.2)
            + _sine(_midi_hz(c), step / SR, amp=0.2)
        )
        if len(clip) > step:
            clip = clip[:step]
        out[i * step : i * step + len(clip)] += clip
    return out


# ----------------------------------------------------------------- network
# Stable, permissively-licensed instrumental clips hosted on librosa.org's CDN.
# Original sources: see librosa example registry. Files are CC-BY / CC0.
_DOWNLOADS: list[tuple[str, str]] = [
    (
        "trumpet.ogg",
        "https://librosa.org/data/audio/sorohanro_-_solo-trumpet-06.ogg",
    ),
    (
        "nutcracker.ogg",
        "https://librosa.org/data/audio/Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.ogg",
    ),
    (
        "brahms.ogg",
        "https://librosa.org/data/audio/Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).ogg",
    ),
]


def _download(url: str, dest: Path, *, ctx: ssl.SSLContext, timeout: float = 30.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "audio2ay-samples/0.1"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  skip {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Bypass SSL verification when fetching online samples "
        "(needed on corporate networks that MITM HTTPS).",
    )
    parser.add_argument(
        "--no-online",
        action="store_true",
        help="Only generate synthetic samples; do not fetch online clips.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "samples"
    out_dir.mkdir(exist_ok=True)

    synthetic = {
        "01_arpeggio_mono.wav": make_arpeggio(),
        "02_bass_and_lead.wav": make_bass_and_lead(),
        "03_drum_loop.wav": make_drum_loop(),
        "04_chord_progression.wav": make_chord_progression(),
    }
    for name, audio in synthetic.items():
        sf.write(out_dir / name, audio, SR, subtype="PCM_16")
        print(f"wrote {out_dir / name} ({len(audio) / SR:.2f}s)")

    if not args.no_online:
        print("\nFetching public-domain instrumental clips (best-effort)...")
        if args.insecure:
            ctx = ssl._create_unverified_context()
            print("  [!] SSL verification disabled (insecure mode)")
        else:
            ctx = ssl.create_default_context()
        for filename, url in _DOWNLOADS:
            dest = out_dir / filename
            if dest.exists() and dest.stat().st_size > 1024:
                print(f"  exists {dest}")
                continue
            print(f"  fetch {url}")
            _download(url, dest, ctx=ctx)

    print("\nDone. Samples in", out_dir)
    print("Try:")
    print("  audio2ay convert samples/02_bass_and_lead.wav build/02.ym --no-separation")
    print("  audio2ay render  build/02.ym build/02_ay.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
