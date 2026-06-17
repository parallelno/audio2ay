"""Simple PySide6 A/B validator GUI.

Loads an audio file, converts it to YM in the background, renders the YM via
the bundled emulator, and lets the user toggle between the original audio and
the AY rendering with a single button. Both signals share a level meter.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def launch_gui() -> None:
    try:
        import sounddevice as sd
        from PySide6 import QtCore, QtWidgets
    except Exception as exc:
        raise SystemExit(
            f"GUI dependencies missing ({exc}). Install with `pip install audio2ay[gui]`."
        ) from exc

    import librosa

    from ..io.ay_emulator import render_song_to_array
    from ..pipeline import convert_audio_to_ym

    SR = 44100

    class Player:
        """A simple stream that plays one of two buffers, with switchable selection."""

        def __init__(self) -> None:
            self.buffers: dict[str, np.ndarray] = {"orig": np.zeros(0, dtype=np.float32),
                                                    "ay": np.zeros(0, dtype=np.float32)}
            self.active = "orig"
            self.pos = 0
            self.stream: sd.OutputStream | None = None
            self._lock = threading.Lock()
            self.last_rms = 0.0

        def set_buffer(self, key: str, arr: np.ndarray) -> None:
            with self._lock:
                self.buffers[key] = arr.astype(np.float32, copy=False)

        def set_active(self, key: str) -> None:
            with self._lock:
                self.active = key

        def play(self) -> None:
            if self.stream is not None:
                return
            self.pos = 0
            self.stream = sd.OutputStream(
                samplerate=SR, channels=1, dtype="float32", callback=self._cb
            )
            self.stream.start()

        def stop(self) -> None:
            if self.stream is not None:
                self.stream.stop(); self.stream.close()
                self.stream = None
                self.pos = 0

        def _cb(self, outdata, frames, _t, _status):
            with self._lock:
                buf = self.buffers[self.active]
            n = min(frames, max(0, len(buf) - self.pos))
            if n > 0:
                chunk = buf[self.pos : self.pos + n]
                outdata[:n, 0] = chunk
                self.pos += n
                self.last_rms = float(np.sqrt(np.mean(chunk**2)))
            else:
                self.last_rms = 0.0
            if n < frames:
                outdata[n:, 0] = 0
                # End of buffer — keep playing silence rather than autostopping
                # (user can press Stop to release the stream).

    class Worker(QtCore.QObject):
        progress = QtCore.Signal(str)
        done = QtCore.Signal(object, object)
        failed = QtCore.Signal(str)

        def __init__(self, in_path: Path) -> None:
            super().__init__()
            self.in_path = in_path

        @QtCore.Slot()
        def run(self) -> None:
            try:
                tmp_dir = self.in_path.parent / "audio2ay_build"
                tmp_dir.mkdir(exist_ok=True)
                ym_path = tmp_dir / (self.in_path.stem + ".ym")
                self.progress.emit("Converting audio → YM …")
                song = convert_audio_to_ym(self.in_path, ym_path)
                self.progress.emit("Rendering YM via emulator …")
                ay_audio = render_song_to_array(song, sample_rate=SR)
                self.progress.emit("Loading original …")
                orig, _ = librosa.load(str(self.in_path), sr=SR, mono=True)
                self.done.emit(orig.astype(np.float32), ay_audio.astype(np.float32))
            except Exception as exc:  # pragma: no cover - GUI path
                import traceback
                tb = traceback.format_exc()
                # Print to stderr so it also shows up in the launching terminal.
                print(tb, file=sys.stderr)
                self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{tb}")

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("audio2ay — A/B validator")
            self.resize(560, 220)

            self.player = Player()
            self.thread: QtCore.QThread | None = None
            self.worker: Worker | None = None

            central = QtWidgets.QWidget(); self.setCentralWidget(central)
            v = QtWidgets.QVBoxLayout(central)

            row = QtWidgets.QHBoxLayout()
            self.btn_open = QtWidgets.QPushButton("Open audio…")
            self.btn_convert = QtWidgets.QPushButton("Convert + Render")
            self.btn_convert.setEnabled(False)
            self.btn_play = QtWidgets.QPushButton("Play")
            self.btn_play.setEnabled(False)
            self.btn_stop = QtWidgets.QPushButton("Stop")
            self.btn_stop.setEnabled(False)
            row.addWidget(self.btn_open); row.addWidget(self.btn_convert)
            row.addWidget(self.btn_play); row.addWidget(self.btn_stop)
            v.addLayout(row)

            row2 = QtWidgets.QHBoxLayout()
            self.rb_orig = QtWidgets.QRadioButton("Original")
            self.rb_orig.setChecked(True)
            self.rb_ay = QtWidgets.QRadioButton("AY rendering")
            row2.addWidget(self.rb_orig); row2.addWidget(self.rb_ay); row2.addStretch()
            v.addLayout(row2)

            self.status = QtWidgets.QLabel("Open an audio file to begin.")
            v.addWidget(self.status)

            self.meter = QtWidgets.QProgressBar()
            self.meter.setRange(0, 100); self.meter.setTextVisible(False)
            v.addWidget(self.meter)

            self.btn_open.clicked.connect(self._on_open)
            self.btn_convert.clicked.connect(self._on_convert)
            self.btn_play.clicked.connect(self._on_play)
            self.btn_stop.clicked.connect(self._on_stop)
            self.rb_orig.toggled.connect(lambda checked: checked and self.player.set_active("orig"))
            self.rb_ay.toggled.connect(lambda checked: checked and self.player.set_active("ay"))

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(50)

            self.in_path: Path | None = None

        def _on_open(self) -> None:
            f, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open instrumental audio",
                "",
                "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)",
            )
            if not f:
                return
            self.in_path = Path(f)
            self.status.setText(f"Loaded: {self.in_path.name}")
            self.btn_convert.setEnabled(True)

        def _on_convert(self) -> None:
            if self.in_path is None:
                return
            self.btn_convert.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.status.setText("Working…")
            self.thread = QtCore.QThread(self)
            self.worker = Worker(self.in_path)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(self.status.setText)
            self.worker.done.connect(self._on_done)
            self.worker.failed.connect(self._on_failed)
            self.worker.done.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.start()

        def _on_done(self, orig: np.ndarray, ay: np.ndarray) -> None:
            self.player.set_buffer("orig", orig)
            self.player.set_buffer("ay", ay)
            self.btn_play.setEnabled(True)
            self.btn_convert.setEnabled(True)
            dur = len(orig) / SR
            self.status.setText(f"Done. Duration: {dur:.1f}s. Toggle Original/AY to A/B.")

        def _on_failed(self, msg: str) -> None:
            QtWidgets.QMessageBox.critical(self, "Conversion failed", msg)
            self.btn_convert.setEnabled(True)
            self.status.setText("Idle.")

        def _on_play(self) -> None:
            self.player.play()
            self.btn_stop.setEnabled(True)

        def _on_stop(self) -> None:
            self.player.stop()
            self.btn_stop.setEnabled(False)

        def _tick(self) -> None:
            level = min(1.0, self.player.last_rms * 4)
            self.meter.setValue(int(level * 100))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(); w.show()
    app.exec()
