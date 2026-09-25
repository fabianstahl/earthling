"""Dialog running tile downloads in a background thread."""

from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from earthling.core.aoi import AVG_TILE_BYTES, format_bytes
from earthling.core.session import Session
from earthling.data.downloader import DownloadProgress
from earthling.data.jobs import DownloadJob


class _Worker(QObject):
    progress = pyqtSignal(str, object)
    job_finished = pyqtSignal(str, object)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, jobs: list[DownloadJob]) -> None:
        super().__init__()
        self.jobs = jobs
        self._cancelled = False
        self._current: DownloadJob | None = None

    def cancel(self) -> None:
        self._cancelled = True
        if self._current is not None:
            self._current.cancel()

    def run(self) -> None:
        try:
            for job in self.jobs:
                if self._cancelled:
                    break
                self._current = job
                last = [0.0]

                def report(p: DownloadProgress, job=job, last=last) -> None:
                    now = time.monotonic()
                    if now - last[0] > 0.1 or p.finished:
                        last[0] = now
                        self.progress.emit(job.name, p)

                result = job.run(report)
                self.job_finished.emit(job.name, result)
        except Exception as exc:  # surface unexpected errors in the dialog
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class DownloadDialog(QDialog):
    data_changed = pyqtSignal()

    def __init__(self, session: Session, jobs_factory, parent=None) -> None:
        """``jobs_factory(kinds: set[str]) -> list[DownloadJob]``"""
        super().__init__(parent)
        self.session = session
        self.jobs_factory = jobs_factory
        self.setWindowTitle("Download data")
        self.resize(640, 420)
        layout = QVBoxLayout(self)
        plan = session.plan
        imagery = session.provider_tiles("imagery")
        count = sum(len(t) for _, tiles in imagery for t in tiles.values())
        names = " + ".join(p.name for p, tiles in imagery if tiles)
        self.imagery_box = QCheckBox(
            f"Imagery – {names}: {count} tiles "
            f"(≈ {format_bytes(count * AVG_TILE_BYTES['imagery'])})"
        )
        self.imagery_box.setChecked(True)
        self.dem_box = QCheckBox(
            f"Elevation (DEM): {plan.count('dem')} tiles "
            f"(≈ {format_bytes(plan.estimated_bytes('dem'))})"
        )
        self.dem_box.setChecked(True)
        self.topo_box = QCheckBox(
            "Topographic map tiles (otherwise downloaded on demand, slowly: max 2 requests/s)"
        )
        self.topo_box.setChecked(False)
        self.topo_box.setEnabled(bool(session.topo_providers))
        layout.addWidget(self.imagery_box)
        layout.addWidget(self.dem_box)
        layout.addWidget(self.topo_box)
        layout.addWidget(QLabel(f"Cache: {session.cache.root}"))
        self.status = QLabel("Idle")
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        layout.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.start_btn = QPushButton("Start")
        self.cancel_btn = QPushButton("Cancel")
        self.close_btn = QPushButton("Close")
        self.cancel_btn.setEnabled(False)
        for b in (self.start_btn, self.cancel_btn, self.close_btn):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self._cancel)
        self.close_btn.clicked.connect(self.close)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

    def start(self) -> None:
        kinds = set()
        if self.imagery_box.isChecked():
            kinds.add("imagery")
        if self.dem_box.isChecked():
            kinds.add("dem")
        if self.topo_box.isChecked():
            kinds.add("topo")
        jobs = self.jobs_factory(kinds)
        if not jobs:
            return
        self._thread = QThread(self)
        self._worker = _Worker(jobs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.job_finished.connect(self._on_job_finished)
        self._worker.failed.connect(lambda msg: self.log.appendPlainText(f"ERROR: {msg}"))
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._thread.start()

    def _on_progress(self, name: str, p: DownloadProgress) -> None:
        self.bar.setMaximum(max(1, p.total))
        self.bar.setValue(p.done)
        self.status.setText(
            f"{name}: {p.done}/{p.total} – {p.downloaded} new, {p.skipped} cached, "
            f"{p.missing} unavailable, {p.failed} failed, {format_bytes(p.bytes)}"
        )

    def _on_job_finished(self, name: str, p: DownloadProgress) -> None:
        self._on_progress(name, p)
        self.log.appendPlainText(
            f"{name}: done – {p.downloaded} downloaded, {p.skipped} already cached, "
            f"{p.missing} unavailable, {p.failed} failed"
        )
        for err in p.errors[:20]:
            self.log.appendPlainText(f"  {err}")
        self.data_changed.emit()

    def _on_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log.appendPlainText("cancelling…")

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._cancel()
            self._thread.quit()
            self._thread.wait(10_000)
        super().closeEvent(event)
