"""Main application window: the docks, the viewport, the timeline and every edit (undoable).

The rest of it lives in mixins, one topic each: menus, toolbar and shortcuts
(``window_actions``), the command palette (``window_palette``), projects and scene files
(``window_project``), what the scene list selects and adds (``window_items``)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QUndoStack
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QUndoView,
    QWidget,
)

from earthling.app.keyframing import KeyframeEditor
from earthling.app.looks import make_looks
from earthling.app.photo_editor import PhotoEditor
from earthling.app.poi_editor import PoiEditor
from earthling.app.scene_list import SceneList
from earthling.app.start_screen import StartScreen
from earthling.app.timeline import TimelineController
from earthling.app.tracks_dock import TracksDock
from earthling.app.undo import set_property
from earthling.app.viewport import Viewport
from earthling.app.viewport_overlay import ViewportOverlay
from earthling.app.window_actions import ActionsMixin
from earthling.app.window_items import ItemsMixin
from earthling.app.window_palette import PaletteMixin
from earthling.app.window_project import ProjectMixin
from earthling.core.animation import layer_switch_warnings
from earthling.core.scene import Scene
from earthling.render.camera_rig import CameraRig
from earthling.session import Session
from earthling.ui.dope_sheet import KeyButton
from earthling.ui.inspector import Inspector, search_action
from earthling.ui.timeline_widget import TimelineWidget


class MainWindow(ActionsMixin, PaletteMixin, ProjectMixin, ItemsMixin, QMainWindow):
    tracks_changed = pyqtSignal()
    data_changed = pyqtSignal()  # new tiles in the cache

    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self.session: Session | None = None
        self.scene = Scene()
        self.scene.on_dirty_changed(lambda dirty: self._update_title())
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(lambda clean: self.scene.mark_dirty(not clean))
        self.timeline = TimelineController(self.scene.animation, self)
        self.keys = KeyframeEditor(self.scene.animation, self.timeline, self.undo_stack)
        self.keys.changed.connect(self._camera_keys_changed)
        self.rig = CameraRig(self.scene.animation, self.scene.store)
        self.settings = QSettings()
        self.resize(1600, 1000)
        self._build_menus()
        self._build_toolbar()
        self.viewport = Viewport(dev_mode=dev_mode)
        show_start = self.settings.value("start/show", True) not in (False, "false")
        self.start_screen = StartScreen(self._recent(), show_start)
        self.central = QStackedWidget()
        self.central.addWidget(self.start_screen)
        self.central.addWidget(self.viewport)
        self.setCentralWidget(self.central)
        self.viewport.set_store(self.scene.store)
        self.viewport.pose_provider = self._animated_pose
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.viewport.shutdown)
        self._build_docks()
        self.viewport.timeline_time_provider = lambda: self.timeline.time
        self._pois_changed()
        self._connect_viewport()
        self.viewport_overlay = ViewportOverlay(self)
        self._update_title()
        self.data_changed.connect(self._reload_terrain)
        self.statusBar().showMessage("Ready")
        self._connect_start_screen()
        self._name_shortcuts_in_tooltips()
        self.show_start(show_start)

    def _build_docks(self) -> None:
        """Parameters, history, timeline, tracks and POI docks (each listed in the View menu)."""
        area = Qt.DockWidgetArea
        right, bottom = area.RightDockWidgetArea, area.BottomDockWidgetArea
        self.inspector = self.property_panel = Inspector(
            self.scene.store,
            on_edit=self._on_panel_edit,
            row_extra=lambda d: KeyButton(d.id, self.keys) if d.animatable else None,
            key_state=self.key_state,
            is_animated=self.scene.animation.is_animated,
            batch=self.undo_batch,
            settings=self.settings,
        )
        self.inspector.rail.add_button("gear-six", "Project", "Project settings (Ctrl+,)",
                                       self.edit_settings)  # fmt: skip
        self.keys.changed.connect(self.inspector.refresh_key_states)
        self.timeline.time_changed.connect(lambda t: self.inspector.refresh_key_states())
        # project specific properties (track groups, POIs) appear as their own sections
        self.scene.dynamic_listeners.append(self.inspector.rebuild)
        self.parameters_dock = self._add_dock(
            right, QDockWidget("Inspector", self), "ParametersDock", self.inspector
        )
        self.parameters_dock.setTitleBarWidget(QWidget())  # (the inspector has its own header)
        self.parameters_dock.setMinimumWidth(440)
        self.view_menu.addAction(search_action(self.inspector, self))
        self.history_dock = self._add_dock(
            right, QDockWidget("History", self), "HistoryDock", QUndoView(self.undo_stack)
        )
        self.history_dock.hide()
        self.timeline_widget = TimelineWidget(
            self.timeline, self.keys,
            extra=[self.camera_key_action, ("magic-wand", "Generate…", self.animation_menu)],
        )  # fmt: skip
        self.timeline_dock = self._add_dock(
            bottom, QDockWidget("Timeline", self), "TimelineDock", self.timeline_widget
        )
        self.timeline_dock.setTitleBarWidget(QWidget())
        self.timeline.settings_changed.connect(lambda: self.scene.mark_dirty())
        self.tracks_dock = self._add_dock(bottom, TracksDock(self))
        self.tabifyDockWidget(self.timeline_dock, self.tracks_dock)
        self.timeline_dock.raise_()
        self.tracks_dock.visibility_changed.connect(self.tracks_changed.emit)
        self.tracks_changed.connect(self.viewport.request_render)
        self.tracks_dock.track_activated.connect(self._frame_track)
        # the forms of the selected POI / photo (shown in the inspector)
        self.poi_editor = PoiEditor(self.scene, self.undo_stack, self.hiker_lonlat,
                                    self._pick_lonlat, self._scene_dir, self,
                                    choose_place=self.choose_place)  # fmt: skip
        self.photo_editor = PhotoEditor(self.scene, self.undo_stack, self.hiker_lonlat,
                                        self._pick_lonlat, self._scene_dir, self,
                                        choose_place=self.choose_place,
                                        place_at_time=self.lonlat_at_time)  # fmt: skip
        for kind, editor in (("poi", self.poi_editor), ("photo", self.photo_editor)):
            editor.hide()
            editor.changed.connect(self._pois_changed)
            editor.current_changed.connect(lambda i, k=kind: self._select_item(k, i))
        self.inspector.reveal_item = self._reveal_item
        self.looks = make_looks(self)
        self.inspector.looks = self.looks
        self.inspector.rebuild()
        self.scene_list = SceneList(self)
        self.scene_list.selected.connect(self._scene_selected)
        self.scene_dock = self._add_dock(area.LeftDockWidgetArea, QDockWidget("Scene", self),
                                         "SceneDock", self.scene_list)  # fmt: skip
        self.scene_dock.setTitleBarWidget(QWidget())
        self.scene_dock.setMinimumWidth(250)
        self.tracks_changed.connect(self.scene_list.refresh)
        self.keys.changed.connect(self.scene_list.refresh)
        self.scene.dynamic_listeners.append(self.scene_list.refresh)
        watched = {"layers.a", "layers.b", "layers.mix", "borders.overlay", "contours.overlay"}

        def store_changed(pid: str, value) -> None:
            if pid in watched or (pid.startswith("trackgroup.") and pid.endswith(".visible")):
                self.scene_list.refresh()

        self.scene.store.subscribe(store_changed)
        self._add_keyboard_navigation()

    def _add_dock(self, area, dock: QDockWidget, name: str = "", widget=None) -> QDockWidget:
        """Dock ``dock`` (optionally naming it and setting its widget) and list it in View."""
        if name:
            dock.setObjectName(name)
        if widget is not None:
            dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self.view_menu.addAction(dock.toggleViewAction())
        return dock

    def _connect_viewport(self) -> None:
        """Viewport state in the View menu actions and the status bar."""
        self.viewport.mode_changed.connect(
            lambda mode: self.fly_mode_action.setChecked(mode == self.viewport.FLY)
        )
        self.viewport.animated_camera_changed.connect(self.animated_camera_action.setChecked)
        self._camera_label = QLabel()
        self.statusBar().addPermanentWidget(self._camera_label)
        self.viewport.camera_changed.connect(self._camera_label.setText)
        self._fps_label = QLabel()
        self.statusBar().addPermanentWidget(self._fps_label)
        self.viewport.stats_changed.connect(self._fps_label.setText)
        self.viewport.shader_error.connect(
            lambda msg: self.statusBar().showMessage(f"Shader error: {msg}", 10000)
        )

    # --- camera animation ----------------------------------------------------------------
    def add_camera_key(self) -> None:
        """Key the current view at the playhead."""
        self.keys.set_key("camera.pose", self.viewport.camera_pose())

    def animation_changed(self) -> None:
        """Keys or the duration changed outside the key editor (generators, their undo)."""
        self.timeline.settings_changed.emit()
        self.timeline.refresh()
        self.keys.changed.emit()
        self.viewport.set_animated_camera(self.scene.animation.is_animated("camera.pose"))
        self.viewport.request_render()

    def check_animation(self) -> list[str]:
        messages = [message for _, _, message in layer_switch_warnings(self.scene.animation)]
        if messages:
            QMessageBox.warning(self, "Animation check", "\n".join(messages))
        else:
            QMessageBox.information(self, "Animation check", "No problems found.")
        return messages

    def frame_renderer(self):
        from earthling.export.frames import FrameRenderer

        return FrameRenderer(self.viewport.renderer, self.scene.animation, self.rig)

    def render_still(self, path: Path, width: int, height: int) -> None:
        """Render the current timeline frame at full export quality."""
        from earthling.export.frames import save_png

        self.viewport.makeCurrent()
        frames = self.frame_renderer()
        try:
            image = frames.render(self.timeline.time, width, height)
        finally:
            frames.release()
            self.viewport.doneCurrent()
        save_png(image, path)
        self.viewport.request_render()

    def render_still_dialog(self) -> None:
        if self.viewport.renderer is None:
            return
        from earthling.export.frames import parse_resolution

        start = str(self.project.folder / "still.png") if self.project else "still.png"
        path, _ = QFileDialog.getSaveFileName(self, "Render still", start, "PNG images (*.png)")
        if not path:
            return
        width, height = parse_resolution(self.scene.store["export.resolution"])
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"Rendering {width} × {height}…")
        try:
            self.render_still(Path(path), width, height)
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Saved {path}", 5000)

    def export_video_dialog(self) -> None:
        from earthling.app.export_dialog import ExportDialog

        if self.viewport.renderer is None:
            return
        ExportDialog(self).exec()

    def _on_panel_edit(self, pid: str, value, interactive: bool) -> None:
        """Panel edits: animated properties are keyed at the playhead (auto-key)."""
        if not self.keys.on_property_edited(pid, value, interactive):
            self.set_property(pid, value, interactive)

    def key_state(self, pid: str) -> str:
        """How the panel tints a value: "keyed" (a key at the playhead), "animated", "static"."""
        if not self.scene.animation.is_animated(pid):
            return "static"
        return "keyed" if self.keys.has_key_now(pid) else "animated"

    def undo_batch(self, label: str, fn) -> None:
        """Run ``fn`` (several edits) as one undo step."""
        self.undo_stack.beginMacro(label)
        try:
            fn()
        finally:
            self.undo_stack.endMacro()

    def set_property(self, pid: str, value, interactive: bool = False) -> None:
        """All interactive property edits go through the undo stack."""
        set_property(self.undo_stack, self.scene.store, pid, value, interactive)

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self._save_thumbnail()
        self.viewport.shutdown()
        # editors must not report edits while Qt destroys them (focus moves, lists clear)
        for widget in (self.inspector, self.poi_editor, self.photo_editor, self.scene_list):
            for child in widget.findChildren(QWidget):
                child.blockSignals(True)
        super().closeEvent(event)
