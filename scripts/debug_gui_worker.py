"""Reproduce the GUI Worker pipeline through a real QThread without showing
any window. Prints any traceback to stderr.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

import librosa

from audio2ay.io.ay_emulator import render_song_to_array
from audio2ay.pipeline import convert_audio_to_ym

SR = 44100


class Worker(QtCore.QObject):
    progress = QtCore.Signal(str)
    done = QtCore.Signal(object, object)
    failed = QtCore.Signal(str)

    def __init__(self, in_path: Path) -> None:
        super().__init__()
        self.in_path = Path(in_path)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            tmp_dir = self.in_path.parent / "audio2ay_build"
            tmp_dir.mkdir(exist_ok=True)
            ym_path = tmp_dir / (self.in_path.stem + ".ym")
            self.progress.emit("Converting audio -> YM ...")
            song = convert_audio_to_ym(self.in_path, ym_path)
            self.progress.emit("Rendering YM via emulator ...")
            ay_audio = render_song_to_array(song, sample_rate=SR)
            self.progress.emit("Loading original ...")
            orig, _ = librosa.load(str(self.in_path), sr=SR, mono=True)
            self.done.emit(orig.astype(np.float32), ay_audio.astype(np.float32))
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{tb}")


def main() -> int:
    in_path = sys.argv[1] if len(sys.argv) > 1 else "samples/nutcracker.ogg"
    if not Path(in_path).exists():
        print(f"missing: {in_path}", file=sys.stderr)
        return 2

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    thread = QtCore.QThread()
    worker = Worker(in_path)
    worker.moveToThread(thread)

    state = {"rc": 1}

    def on_progress(msg: str) -> None:
        print(f"[progress] {msg}", flush=True)

    def on_done(orig, ay) -> None:
        try:
            print(f"[done] orig={orig.shape} ay={ay.shape}", flush=True)
            state["rc"] = 0
        finally:
            thread.quit()

    def on_failed(msg: str) -> None:
        print(f"[FAILED] {msg}", file=sys.stderr, flush=True)
        thread.quit()

    worker.progress.connect(on_progress)
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
    thread.started.connect(worker.run)
    thread.finished.connect(app.quit)

    thread.start()
    app.exec()
    return state["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
