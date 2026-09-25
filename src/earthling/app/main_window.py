"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox

from earthling import __version__
from earthling.app.download_dialog import DownloadDialog
from earthling.app.goto_dialog import GoToDialog
from earthling.app.plan_dialog import PlanDialog
from earthling.app.tracks_dock import TracksDock
from earthling.app.viewport import Viewport
from earthling.core.config import ConfigError, Project
from earthling.core.gpx import Track
from earthling.core.session import Session
from earthling.data.jobs import download_jobs

MAX_RECENT = 8


class MainWindow(QMainWindow):
    project_changed = pyqtSignal(object)
    tracks_changed = pyqtSignal()
    data_changed = pyqtSignal()  # new tiles in the cache

    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self.session: Session | None = None
        self.settings = QSettings()
        self.resize(1600, 1000)
        self._build_menus()
        self.viewport = Viewport(dev_mode=dev_mode)
        self.setCentralWidget(self.viewport)
        self.tracks_dock = TracksDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.tracks_dock)
        self.view_menu.addAction(self.tracks_dock.toggleViewAction())
        self.tracks_dock.visibility_changed.connect(self.tracks_changed.emit)
        self.tracks_dock.track_activated.connect(self._frame_track)
        self.viewport.mode_changed.connect(
            lambda mode: self.fly_mode_action.setChecked(mode == self.viewport.FLY)
        )
        self._camera_label = QLabel()
        self.statusBar().addPermanentWidget(self._camera_label)
        self.viewport.camera_changed.connect(self._camera_label.setText)
        self._fps_label = QLabel()
        self.statusBar().addPermanentWidget(self._fps_label)
        self.viewport.stats_changed.connect(self._fps_label.setText)
        self.viewport.shader_error.connect(
            lambda msg: self.statusBar().showMessage(f"Shader error: {msg}", 10000)
        )
        self._update_title()
        self.data_changed.connect(self._reload_terrain)
        self.statusBar().showMessage("Ready")

    # --- menus -------------------------------------------------------------------------
    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.file_menu = bar.addMenu("&File")
        self._add_action(
            self.file_menu, "&New Project…", self._new_project, QKeySequence.StandardKey.New
        )
        self._add_action(
            self.file_menu, "&Open Project Folder…", self._open_project_dialog,
            QKeySequence.StandardKey.Open,
        )  # fmt: skip
        self.recent_menu = self.file_menu.addMenu("Open &Recent")
        self._add_action(self.file_menu, "Re&load Project", self._reload_project, "F5")
        self.file_menu.addSeparator()
        self._add_action(self.file_menu, "&Quit", self.close, QKeySequence.StandardKey.Quit)
        self._rebuild_recent_menu()

        self.view_menu = bar.addMenu("&View")
        self._add_action(self.view_menu, "&Frame All", lambda: self.viewport.frame_all(), "Home")
        self._add_action(self.view_menu, "&Go To Coordinate…", self._go_to, "Ctrl+G")
        self.fly_mode_action = QAction("&Fly Camera (Tab)", self, checkable=True)
        self.fly_mode_action.toggled.connect(
            lambda v: self.viewport.set_mode(self.viewport.FLY if v else self.viewport.ORBIT)
        )
        self.view_menu.addAction(self.fly_mode_action)
        self.view_menu.addSeparator()
        self.debug_lod_action = QAction("Debug: Show Terrain &LOD", self, checkable=True)
        self.debug_lod_action.toggled.connect(lambda v: self.viewport.set_debug_lod(v))
        self.view_menu.addAction(self.debug_lod_action)
        self.on_demand_action = QAction("Download Missing Imagery &On Demand", self, checkable=True)
        self.on_demand_action.setChecked(True)
        self.on_demand_action.toggled.connect(lambda v: self.viewport.set_on_demand(v))
        self.show_outlines_action = QAction("Show &Area Outlines", self, checkable=True)
        self.show_outlines_action.setChecked(True)
        self.show_outlines_action.toggled.connect(lambda v: self.viewport.set_outlines_visible(v))
        self.view_menu.addAction(self.show_outlines_action)
        self.view_menu.addSeparator()

        self.data_menu = bar.addMenu("&Data")
        self._add_action(self.data_menu, "Download &Plan…", self._show_plan)
        self._add_action(self.data_menu, "&Download Data…", self._show_download, "Ctrl+D")
        self.data_menu.addAction(self.on_demand_action)

        help_menu = bar.addMenu("&Help")
        self._add_action(help_menu, "&About Earthling", self._show_about)

    def _add_action(self, menu, text, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    @property
    def project(self) -> Project | None:
        return self.session.project if self.session else None

    @property
    def tracks(self) -> list[Track]:
        return self.session.tracks if self.session else []

    # --- project handling --------------------------------------------------------------
    def open_project(self, folder: str | Path) -> bool:
        try:
            session = Session(Project.load(folder))
        except ConfigError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))
            return False
        self._set_session(session)
        return True

    def _set_session(self, session: Session) -> None:
        project = session.project
        self.session = session
        self._remember_recent(project.folder)
        self._update_title()
        self.tracks_dock.set_tracks(session.tracks)
        self.viewport.set_scene(session.frame, session.tracks)
        self.viewport.set_outlines(session.outline_lines())
        self._reload_terrain(reframe=True)
        self.statusBar().showMessage(
            f"Opened project {project.folder} – {len(session.tracks)} track(s)", 5000
        )
        if session.load_errors:
            QMessageBox.warning(self, "GPX problems", "\n".join(session.load_errors))
        self.project_changed.emit(project)
        self.tracks_changed.emit()

    def _reload_terrain(self, reframe: bool = False) -> None:
        if self.session is None:
            return
        self.viewport.set_terrain_source(self.session.terrain_data(), self.session.terrain_nodes())
        if reframe:
            self.viewport.frame_all()

    def _show_plan(self) -> None:
        if self.session is None or self.session.plan is None:
            QMessageBox.information(self, "Download plan", "Open a project with tracks first.")
            return
        PlanDialog(self.session, self).exec()

    def _show_download(self) -> None:
        if self.session is None or self.session.plan is None:
            QMessageBox.information(self, "Download", "Open a project with tracks first.")
            return
        session = self.session
        dialog = DownloadDialog(session, lambda kinds: download_jobs(session, kinds), self)
        dialog.data_changed.connect(self.data_changed.emit)
        dialog.exec()

    def _new_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose (empty) folder for new project")
        if folder:
            project = Project.create(folder)
            self.open_project(project.folder)

    def _open_project_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open project folder")
        if folder:
            self.open_project(folder)

    def _reload_project(self) -> None:
        if self.project is not None:
            self.open_project(self.project.folder)

    def _recent(self) -> list[str]:
        value = self.settings.value("recent_projects", [])
        if isinstance(value, str):
            value = [value]
        return list(value or [])

    def _remember_recent(self, folder: Path) -> None:
        recent = [str(folder)] + [p for p in self._recent() if p != str(folder)]
        self.settings.setValue("recent_projects", recent[:MAX_RECENT])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self._recent()
        for path in recent:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _=False, p=path: self.open_project(p))
        self.recent_menu.setEnabled(bool(recent))

    def _update_title(self) -> None:
        title = "Earthling"
        if self.project is not None:
            title = f"{self.project.name} – Earthling"
        self.setWindowTitle(title)

    def _go_to(self) -> None:
        geo = self.viewport.camera_geodetic() or (46.0, 7.0, 0.0)
        dialog = GoToDialog(geo[0], geo[1], self)
        if dialog.exec():
            self.viewport.go_to(*dialog.values())

    def _frame_track(self, index: int) -> None:
        if self.session is None or not (0 <= index < len(self.tracks)):
            return
        track = self.tracks[index]
        lat, lon, ele = track.all_points()
        enu = self.session.frame.geodetic_to_enu(lat, lon, np.nan_to_num(ele))
        self.viewport.frame_box(enu.min(axis=0), enu.max(axis=0))

    def closeEvent(self, event) -> None:
        self.viewport.shutdown()
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(self, "About Earthling", f"Earthling {__version__}")
