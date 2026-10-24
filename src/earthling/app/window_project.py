"""Projects and scene files in the main window: open, reload, recent projects, the start
screen, saving and reverting scenes, the scene frame, project settings, new lines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QUndoCommand
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from earthling.app.download_dialog import DownloadDialog
from earthling.app.plan_dialog import PlanDialog
from earthling.core.config import ConfigError, Project
from earthling.core.gpx import Track
from earthling.core.scene import DEFAULT_SCENE_NAME, SceneError
from earthling.data.jobs import download_jobs
from earthling.session import Session

MAX_RECENT = 8


class SceneOriginCommand(QUndoCommand):
    def __init__(self, window, before, after) -> None:
        super().__init__("Scene frame")
        self.window, self.before, self.after = window, before, after

    def redo(self) -> None:
        self.window._reframe(self.after)

    def undo(self) -> None:
        self.window._reframe(self.before)


class ProjectMixin:
    """Part of :class:`earthling.app.main_window.MainWindow` (its ``self``)."""

    # --- start screen ---------------------------------------------------------------------
    def _connect_start_screen(self) -> None:
        start = self.start_screen
        start.open_requested.connect(self.open_project)
        start.new_requested.connect(self.new_project_from_gpx)
        start.choose_new.connect(self._choose_new_project_gpx)
        start.browse_requested.connect(self._open_project_dialog)
        start.continue_requested.connect(lambda: self.show_start(False))
        start.shortcuts_requested.connect(self.show_shortcuts)
        start.show_switch.toggled.connect(lambda v: self.settings.setValue("start/show", v))

    def show_start(self, on: bool) -> None:
        """The start screen instead of the viewport, the docks and the toolbar."""
        if on:
            self.start_screen.set_recent(self._recent())
        if on == getattr(self, "_start_shown", None):
            return
        self._start_shown = on
        docks = [self.scene_dock, self.parameters_dock, self.timeline_dock, self.tracks_dock]
        if on:
            self._hidden_docks = [d for d in docks if not d.isHidden()]
            for dock in docks:
                dock.hide()
            self.toolbar.hide()
            self.central.setCurrentWidget(self.start_screen)
        else:
            for dock in getattr(self, "_hidden_docks", docks):
                dock.show()
            self._hidden_docks = []
            self.toolbar.show()
            self.central.setCurrentWidget(self.viewport)
            self.timeline_dock.raise_()

    def _choose_new_project_gpx(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "GPX files for the new project",
                                                str(Path.home()), "GPX files (*.gpx)")  # fmt: skip
        if paths:
            self.new_project_from_gpx([Path(p) for p in paths])

    def new_project_from_gpx(self, paths: list[Path], folder: Path | None = None) -> bool:
        """A new project in ``folder`` (asked for) with copies of the GPX files, opened."""
        import shutil

        if folder is None:
            start = str(paths[0].parent) if paths else str(Path.home())
            chosen = QFileDialog.getExistingDirectory(
                self, "Folder for the new project (empty or new)", start
            )
            if not chosen:
                return False
            folder = Path(chosen)
        project = Project.create(folder)
        project.gpx_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, project.gpx_dir / path.name)
        return self.open_project(project.folder)

    def _save_thumbnail(self) -> None:
        """A small picture of the view for the start screen's recent projects."""
        from earthling.app.start_screen import thumbnail_path

        if self.project is None or self.viewport.renderer is None or not self.viewport.isVisible():
            return
        image = self.viewport.grabFramebuffer()
        if image.isNull():
            return
        path = thumbnail_path(self.project.folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.scaledToWidth(480, Qt.TransformationMode.SmoothTransformation).save(
            str(path), "JPG", 85
        )

    @property
    def project(self) -> Project | None:
        return self.session.project if self.session else None

    @property
    def tracks(self) -> list[Track]:
        return self.session.all_tracks if self.session else []

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
        self.show_start(False)
        self._remember_recent(project.folder)
        self._update_title()
        self.tracks_dock.set_tracks(session.all_tracks)
        self.scene.set_dynamic("track_groups", session.track_group_properties())
        self._show_session(reframe=True)
        scene_path = project.folder / DEFAULT_SCENE_NAME
        if scene_path.exists():
            self.load_scene(scene_path)
        else:
            self.scene.reset(scene_path)
            self._init_new_scene(session)
            self._scene_replaced()
            self.undo_stack.clear()
            self._update_title()
        self.statusBar().showMessage(
            f"Opened project {project.folder} – {len(session.tracks)} track(s)", 5000
        )
        if session.load_errors:
            QMessageBox.warning(self, "GPX problems", "\n".join(session.load_errors))
        self.tracks_changed.emit()

    def _init_new_scene(self, session: Session) -> None:
        """Sensible starting values for a project's first scene."""
        start = session.first_local_start()
        if start is not None:
            self.scene.store.set("sun.datetime", start.replace(hour=12, minute=0, second=0))

    def _show_session(self, reframe: bool = False) -> None:
        """(Re)build the 3D scene of the session: tracks, area outlines, terrain."""
        session = self.session
        self.viewport.set_scene(
            session.frame, session.all_tracks, timezone=session.config.project.timezone
        )
        self.viewport.set_outlines(session.outline_lines())
        self._reload_terrain(reframe=reframe)

    def _scene_replaced(self) -> None:
        """A new or loaded scene: the views catch up, starting at 0 s and clean."""
        self.timeline.settings_changed.emit()
        self.timeline.set_time(0.0)
        self.keys.changed.emit()  # views caching the curves (graph editor, dope sheet, camera path)
        self.scene.mark_dirty(False)

    # --- scene files -------------------------------------------------------------------
    def load_scene(self, path: Path) -> bool:
        try:
            problems = self.scene.load(path)
        except SceneError as exc:
            QMessageBox.critical(self, "Cannot open scene", str(exc))
            return False
        self.undo_stack.clear()
        self._apply_scene_origin()
        self.timeline.pause()
        self._scene_replaced()
        has_camera_keys = self.scene.animation.is_animated("camera.pose")
        self.viewport.set_animated_camera(has_camera_keys)
        if self.scene.camera and not has_camera_keys:
            self.viewport.restore_camera_state(self.scene.camera)
        self._restore_ui_state()
        self.poi_editor.refresh()
        self.photo_editor.refresh()
        self.scene_list.refresh()
        self.inspector.show_category("camera")
        self._pois_changed()
        self._update_title()
        if problems:
            QMessageBox.warning(self, "Scene problems", "\n".join(problems[:30]))
        return True

    def _view_centre(self) -> tuple[float, float] | None:
        """(lat, lon) of the terrain in the middle of the view (or of the orbit target)."""
        if self.session is None:
            return None
        point = self.viewport.pick_center()
        if point is None:
            point = self.viewport.orbit.target
        lat, lon, _ = self.session.frame.enu_to_geodetic(np.asarray(point, dtype=np.float64))
        return float(lat), float(lon)

    def set_scene_origin(self, origin: tuple[float, float] | None) -> None:
        """Centre the scene's local frame on ``origin`` (lat, lon; None: the tracks' centre),
        keeping every camera where it is. Far from the origin the frame's "up" tilts (5° at
        600 km): a scene far away is level only in its own frame. Undoable."""
        if self.session is None or origin == self.scene.origin:
            return
        self.undo_stack.push(SceneOriginCommand(self, self.scene.origin, origin))

    def _reframe(self, origin: tuple[float, float] | None) -> None:
        from earthling.scenekit import move_origin, reframe_pose

        old = self.session.frame
        self.session.set_origin(origin)
        new = self.session.frame
        move_origin(self.scene, old, new)
        pose = reframe_pose(self.viewport.camera_pose(), old, new)
        self.scene.origin = origin
        self._show_session()
        self.viewport.apply_pose(pose)
        self.scene.mark_dirty()
        self.animation_changed()

    def _apply_scene_origin(self) -> None:
        """Rebuild the 3D scene when the scene file has its own frame origin (or the previous
        scene had one)."""
        if self.session is not None and self.session.set_origin(self.scene.origin):
            self._show_session()

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
        self._save_thumbnail()
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
        self.viewport.set_terrain_source(
            self.session.terrain_data(),
            self.session.terrain_nodes(),
            self.session.labels,
            self.session.attribution(),
        )
        if reframe:
            self.viewport.frame_all()

    def _planned_session(self, title: str) -> Session | None:
        """The session if it has a download plan, else a hint titled ``title``."""
        if self.session is None or self.session.plan is None:
            QMessageBox.information(self, title, "Open a project with tracks first.")
            return None
        return self.session

    def _test_source(self) -> None:
        from earthling.app.source_dialog import SourceDialog

        if self.session is not None:
            SourceDialog(self.session, self.choose_place, self).exec()

    def _show_plan(self) -> None:
        if (session := self._planned_session("Download plan")) is not None:
            PlanDialog(session, self).exec()

    def _show_download(self) -> None:
        if (session := self._planned_session("Download")) is None:
            return
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
        """Load the project again (tracks, data sources) and reopen the scene that was open."""
        if self.project is None:
            return
        scene = self.scene.path
        if not self.open_project(self.project.folder):
            return
        if scene is not None and scene.exists() and scene != self.scene.path:
            self.load_scene(scene)

    def new_line(self) -> None:
        """A line between two places (straight, a flight, a road route) as a GPX track."""
        from earthling.app.line_dialog import LineDialog, add_group_text, group_folder
        from earthling.data.lines import make_line

        if self.session is None:
            return
        session = self.session
        groups = [g.name for g in session.groups]
        dialog = LineDialog(groups, session.config.project.timezone, self.choose_place,
                            session.first_local_start(), self)  # fmt: skip
        if not dialog.exec():
            return
        name, group = dialog.name.text().strip() or "line", dialog.group_name()
        folder = group_folder(session, group)
        if folder is None:
            QMessageBox.warning(self, "New line", f"Track group '{group}' is a single GPX file: "
                                "choose another group or a new one.")  # fmt: skip
            return
        start, end = dialog.times()
        (a, b), kind = dialog.places.values(), dialog.kind.currentData()
        try:
            make_line(folder / f"{dialog.file_name()}.gpx", name, kind, a, b, start, end,
                      session.cache.root)  # fmt: skip
        except Exception as exc:  # (the road route needs the internet)
            QMessageBox.warning(self, "New line", f"Cannot make the line: {exc}")
            return
        if session.group(group) is None:
            path = session.project.config_path
            text = add_group_text(path.read_text(encoding="utf-8"), session, group,
                                  kind == "flight")  # fmt: skip
            path.write_text(text, encoding="utf-8")
        self._reload_project()

    def edit_settings(self) -> None:
        """The project's earthling.toml in a checking editor; saving reloads the project."""
        from earthling.app.settings_dialog import SettingsDialog

        if self.project is None:
            return
        if SettingsDialog(self.project.config_path, self).exec():
            self._reload_project()

    def _recent(self) -> list[str]:
        value = self.settings.value("recent_projects", [])
        if isinstance(value, str):
            value = [value]
        return list(value or [])

    def _remember_recent(self, folder: Path) -> None:
        recent = [str(folder)] + [p for p in self._recent() if p != str(folder)]
        self.settings.setValue("recent_projects", recent[:MAX_RECENT])
        self._rebuild_recent_menu()

    def _update_title(self) -> None:
        if sip.isdeleted(self):  # (the undo stack turns clean while the window is destroyed)
            return
        title = "Earthling"
        if self.project is not None:
            scene = self.scene.path.name if self.scene.path else "untitled"
            title = f"{self.project.name} – {scene}{'*' if self.scene.dirty else ''} – Earthling"
        self.setWindowTitle(title)
