"""Video export dialog: settings, progress/ETA, cancel. Renders one frame per event loop turn
so the application stays responsive."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from earthling.export.frames import parse_resolution
from earthling.export.video import VideoWriter, VideoWriterError, available_presets
from earthling.ui.widgets import double_spin, row


def _resolutions() -> list[tuple[str, str]]:
    """(value, label) of the export resolutions (landscape, portrait, square)."""
    from earthling.render.parameters import EXPORT

    definition = next(d for d in EXPORT if d.id == "export.resolution")
    return list(definition.options)


class ExportDialog(QDialog):
    def __init__(self, window, parent=None) -> None:
        """``window``: the MainWindow (scene, timeline, viewport, frame renderer)."""
        super().__init__(parent or window)
        self.main_window = window
        self.setWindowTitle("Export video")
        self.resize(560, 300)
        anim = window.scene.animation
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.preset = QComboBox()
        for preset in available_presets():
            self.preset.addItem(preset.label, preset)
        self.preset.currentIndexChanged.connect(self._update_extension)
        form.addRow("Format", self.preset)
        self.resolution = QComboBox()
        for res, label in _resolutions():
            self.resolution.addItem(label, res)
        index = self.resolution.findData(window.scene.store["export.resolution"])
        self.resolution.setCurrentIndex(max(0, index))
        form.addRow("Resolution", self.resolution)
        self.start = double_spin(0.0, anim.duration, decimals=2, suffix=" s", value=0.0)
        self.end = double_spin(0.0, anim.duration, decimals=2, suffix=" s", value=anim.duration)
        form.addRow("Range", row(self.start, QLabel("to"), self.end))
        form.addRow("Frame rate", QLabel(f"{anim.fps:g} fps (timeline setting)"))
        self.path = QLineEdit()
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        form.addRow("File", row((self.path, 1), browse))
        layout.addLayout(form)
        self.bar = QProgressBar()
        layout.addWidget(self.bar)
        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.export_btn = QPushButton("Export")
        self.cancel_btn = QPushButton("Close")
        buttons.addWidget(self.export_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)
        self.export_btn.clicked.connect(self.start_export)
        self.cancel_btn.clicked.connect(self._cancel_or_close)
        folder = window.project.folder if window.project else Path.cwd()
        scene = window.scene.path.stem if window.scene.path else "earthling"
        self.path.setText(str(folder / f"{scene}.mov"))
        self._update_extension()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._render_next)
        self._writer: VideoWriter | None = None
        self._frames = None
        self._images = None

    def _update_extension(self) -> None:
        preset = self.preset.currentData()
        if preset is not None and self.path.text():
            self.path.setText(str(Path(self.path.text()).with_suffix(preset.extension)))

    def _browse(self) -> None:
        preset = self.preset.currentData()
        pattern = f"Video (*{preset.extension})" if preset else "Video (*.*)"
        path, _ = QFileDialog.getSaveFileName(self, "Export video", self.path.text(), pattern)
        if path:
            self.path.setText(path)

    @property
    def running(self) -> bool:
        return self._writer is not None

    def start_export(self) -> None:
        preset = self.preset.currentData()
        window = self.main_window
        if preset is None or window.viewport.renderer is None:
            return
        width, height = parse_resolution(self.resolution.currentData())
        anim = window.scene.animation
        self.fps = anim.fps
        self.first = int(round(self.start.value() * self.fps))
        self.count = max(0, int(round(self.end.value() * self.fps)) - self.first)
        if self.count == 0:
            return
        try:
            self._writer = VideoWriter(Path(self.path.text()), width, height, self.fps, preset)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        window.timeline.pause()
        window.viewport.suspended = True
        window.viewport.makeCurrent()
        self._frames = window.frame_renderer()
        times = [(self.first + i) / self.fps for i in range(self.count)]
        self._images = self._frames.iter_frames(times, width, height, preset.bits)
        window.viewport.doneCurrent()
        self.index = 0
        self.started = time.monotonic()
        self.bar.setMaximum(self.count)
        self.export_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")
        self._timer.start(0)

    def _render_next(self) -> None:
        if self._writer is None or self._frames is None:
            return
        t = (self.first + self.index) / self.fps
        vp = self.main_window.viewport
        vp.makeCurrent()
        try:
            self._writer.write(next(self._images))
        except (VideoWriterError, RuntimeError, StopIteration) as exc:
            vp.doneCurrent()
            self._finish(error=str(exc))
            return
        vp.doneCurrent()
        self.index += 1
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.index * (self.count - self.index)
        self.bar.setValue(self.index)
        self.status.setText(
            f"Frame {self.index}/{self.count}  ·  {self.index / elapsed:.2f} fps  ·  "
            f"ETA {int(eta // 60)}:{int(eta % 60):02d}"
        )
        self.main_window.timeline.time = t
        self.main_window.timeline.time_changed.emit(t)
        if self.index >= self.count:
            self._finish()

    def _finish(self, error: str | None = None, cancelled: bool = False) -> None:
        self._timer.stop()
        writer, self._writer = self._writer, None
        if writer is not None:
            if cancelled or error:
                writer.abort()
            else:
                try:
                    writer.close()
                except VideoWriterError as exc:
                    error = str(exc)
        vp = self.main_window.viewport
        if self._frames is not None:
            vp.makeCurrent()
            self._images.close()
            self._frames.release()
            vp.doneCurrent()
            self._frames = None
        vp.suspended = False
        vp.request_render()
        self.export_btn.setEnabled(True)
        self.cancel_btn.setText("Close")
        if error:
            self.status.setText(f"Failed: {error}")
            QMessageBox.critical(self, "Export failed", error)
        elif cancelled:
            self.status.setText("Cancelled")
        else:
            self.status.setText(f"Done: {self.path.text()}")

    def _cancel_or_close(self) -> None:
        if self.running:
            self._finish(cancelled=True)
        else:
            self.close()

    def closeEvent(self, event) -> None:
        if self.running:
            self._finish(cancelled=True)
        super().closeEvent(event)
