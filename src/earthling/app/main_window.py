"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QByteArray, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QUndoStack
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QUndoView,
)

from earthling import __version__
from earthling.app.download_dialog import DownloadDialog
from earthling.app.goto_dialog import GoToDialog
from earthling.app.plan_dialog import PlanDialog
from earthling.app.tracks_dock import TracksDock
from earthling.app.undo import set_property
from earthling.app.viewport import Viewport
from earthling.core.config import ConfigError, Project
from earthling.core.gpx import Track
from earthling.core.scene import DEFAULT_SCENE_NAME, Scene, SceneError
from earthling.core.session import Session
from earthling.data.jobs import download_jobs
from earthling.ui.property_panel import PropertyPanel

MAX_RECENT = 8


class MainWindow(QMainWindow):
    project_changed = pyqtSignal(object)
    tracks_changed = pyqtSignal()
    data_changed = pyqtSignal()  # new tiles in the cache

    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self.session: Session | None = None
        self.scene = Scene()
        self.scene.on_dirty_changed(lambda dirty: self._update_title())
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(lambda clean: self.scene.mark_dirty(not clean))
        self.settings = QSettings()
        self.resize(1600, 1000)
        self._build_menus()
        self.viewport = Viewport(dev_mode=dev_mode)
        self.setCentralWidget(self.viewport)
        self.viewport.set_store(self.scene.store)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.viewport.shutdown)
        self.parameters_dock = QDockWidget("Parameters", self)
        self.parameters_dock.setObjectName("ParametersDock")
        self.property_panel = PropertyPanel(self.scene.store, on_edit=self.set_property)
        self.parameters_dock.setWidget(self.property_panel)
        self.parameters_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_dock)
        self.view_menu.addAction(self.parameters_dock.toggleViewAction())
        self.history_dock = QDockWidget("History", self)
        self.history_dock.setObjectName("HistoryDock")
        self.history_dock.setWidget(QUndoView(self.undo_stack))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.history_dock)
        self.history_dock.hide()
        self.view_menu.addAction(self.history_dock.toggleViewAction())
        self.tracks_dock = TracksDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.tracks_dock)
        self.view_menu.addAction(self.tracks_dock.toggleViewAction())
        self.tracks_dock.visibility_changed.connect(self.tracks_changed.emit)
        self.tracks_changed.connect(self.viewport.request_render)
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
        self._add_action(self.file_menu, "Open &Scene…", self._open_scene_dialog)
        self._add_action(
            self.file_menu, "&Save Scene", lambda: self.save_scene(), QKeySequence.StandardKey.Save
        )
        self._add_action(self.file_menu, "Save Scene &As…", self.save_scene_as, "Ctrl+Shift+S")
        self._add_action(self.file_menu, "Re&vert Scene", self._revert_scene)
        self.file_menu.addSeparator()
        self._add_action(self.file_menu, "&Quit", self.close, QKeySequence.StandardKey.Quit)
        self._rebuild_recent_menu()

        edit_menu = bar.addMenu("&Edit")
        undo = self.undo_stack.createUndoAction(self, "&Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self.undo_stack.createRedoAction(self, "&Redo")
        redo.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)

        self.view_menu = bar.addMenu("&View")
        self._add_action(self.view_menu, "&Frame All", lambda: self.viewport.frame_all(), "Home")
        self._add_action(self.view_menu, "&Go To Coordinate…", self._go_to, "Ctrl+G")
        self.fly_mode_action = QAction("&Fly Camera (Tab)", self, checkable=True)
        self.fly_mode_action.toggled.connect(
            lambda v: self.viewport.set_mode(self.viewport.FLY if v else self.viewport.ORBIT)
        )
        self.view_menu.addAction(self.fly_mode_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(
            self._store_toggle("terrain.debug_lod", "Debug: Show Terrain &LOD")
        )
        self.on_demand_action = QAction("Download Missing Imagery &On Demand", self, checkable=True)
        self.on_demand_action.setChecked(True)
        self.on_demand_action.toggled.connect(lambda v: self.viewport.set_on_demand(v))
        self.view_menu.addAction(self._store_toggle("view.show_outlines", "Show &Area Outlines"))
        self.view_menu.addSeparator()

        self.data_menu = bar.addMenu("&Data")
        self._add_action(self.data_menu, "Download &Plan…", self._show_plan)
        self._add_action(self.data_menu, "&Download Data…", self._show_download, "Ctrl+D")
        self.data_menu.addAction(self.on_demand_action)

        help_menu = bar.addMenu("&Help")
        self._add_action(help_menu, "&About Earthling", self._show_about)

    def set_property(self, pid: str, value, interactive: bool = False) -> None:
        """All interactive property edits go through the undo stack."""
        set_property(self.undo_stack, self.scene.store, pid, value, interactive)

    def _store_toggle(self, pid: str, text: str) -> QAction:
        """Checkable action mirroring a boolean scene property."""
        action = QAction(text, self, checkable=True)
        action.setChecked(bool(self.scene.store[pid]))
        action.toggled.connect(lambda v: self.set_property(pid, v))

        def sync(changed: str, value) -> None:
            if changed == pid and action.isChecked() != bool(value):
                action.setChecked(bool(value))

        self.scene.store.subscribe(sync)
        return action

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
        if not self._confirm_discard():
            return False
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
        self.viewport.set_scene(
            session.frame, session.tracks, timezone=session.config.project.timezone
        )
        self.viewport.set_outlines(session.outline_lines())
        self._reload_terrain(reframe=True)
        scene_path = project.folder / DEFAULT_SCENE_NAME
        if scene_path.exists():
            self.load_scene(scene_path)
        else:
            self.scene.reset(scene_path)
            self._init_new_scene(session)
            self.undo_stack.clear()
            self._update_title()
        self.statusBar().showMessage(
            f"Opened project {project.folder} – {len(session.tracks)} track(s)", 5000
        )
        if session.load_errors:
            QMessageBox.warning(self, "GPX problems", "\n".join(session.load_errors))
        self.project_changed.emit(project)
        self.tracks_changed.emit()

    def _init_new_scene(self, session: Session) -> None:
        """Sensible starting values for a project's first scene."""
        start = session.first_local_start()
        if start is not None:
            self.scene.store.set("sun.datetime", start.replace(hour=12, minute=0, second=0))
        self.scene.mark_dirty(False)

    # --- scene files -------------------------------------------------------------------
    def load_scene(self, path: Path) -> bool:
        try:
            problems = self.scene.load(path)
        except SceneError as exc:
            QMessageBox.critical(self, "Cannot open scene", str(exc))
            return False
        self.undo_stack.clear()
        if self.scene.camera:
            self.viewport.restore_camera_state(self.scene.camera)
        self._restore_ui_state()
        self._update_title()
        if problems:
            QMessageBox.warning(self, "Scene problems", "\n".join(problems[:30]))
        return True

    def save_scene(self, path: Path | None = None) -> bool:
        if path is None and self.scene.path is None:
            return self.save_scene_as()
        self.scene.camera = self.viewport.camera_state()
        self.scene.ui = self._ui_state()
        try:
            saved = self.scene.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save scene", str(exc))
            return False
        self.undo_stack.setClean()
        self._update_title()
        self.statusBar().showMessage(f"Saved {saved}", 4000)
        return True

    def save_scene_as(self) -> bool:
        start = str(self.scene.path or (self.project.folder if self.project else Path.cwd()))
        path, _ = QFileDialog.getSaveFileName(self, "Save scene as", start, "Scenes (*.json)")
        return bool(path) and self.save_scene(Path(path))

    def _open_scene_dialog(self) -> None:
        if not self._confirm_discard():
            return
        start = str(self.project.folder if self.project else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", start, "Scenes (*.json)")
        if path:
            self.load_scene(Path(path))

    def _revert_scene(self) -> None:
        if self.scene.path is not None and self.scene.path.exists():
            self.load_scene(self.scene.path)

    def _confirm_discard(self) -> bool:
        """Ask to save unsaved changes. False if the user cancelled."""
        if not self.scene.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The scene has unsaved changes. Save them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_scene()
        return answer == QMessageBox.StandardButton.Discard

    def _ui_state(self) -> dict:
        return {"window_state": bytes(self.saveState().toBase64()).decode("ascii")}

    def _restore_ui_state(self) -> None:
        state = self.scene.ui.get("window_state")
        if state:
            self.restoreState(QByteArray.fromBase64(state.encode("ascii")))

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
            scene = self.scene.path.name if self.scene.path else "untitled"
            title = f"{self.project.name} – {scene}{'*' if self.scene.dirty else ''} – Earthling"
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
        if not self._confirm_discard():
            event.ignore()
            return
        self.viewport.shutdown()
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(self, "About Earthling", f"Earthling {__version__}")
