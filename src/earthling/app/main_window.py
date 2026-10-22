"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QByteArray, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QUndoView,
    QWidget,
)

from earthling import __version__
from earthling.app.download_dialog import DownloadDialog
from earthling.app.goto_dialog import GoToDialog
from earthling.app.keyframing import KeyframeEditor
from earthling.app.looks import make_looks
from earthling.app.photo_editor import PhotoEditor
from earthling.app.plan_dialog import PlanDialog
from earthling.app.poi_editor import PoiEditor
from earthling.app.scene_list import PHOTO_SUFFIXES, SceneList
from earthling.app.timeline import TimelineController
from earthling.app.tracks_dock import TracksDock
from earthling.app.undo import set_property
from earthling.app.viewport import Viewport
from earthling.core.animation import layer_switch_warnings
from earthling.core.camera_path import camera_path_lines
from earthling.core.config import ConfigError, Project
from earthling.core.gpx import Track
from earthling.core.poi_library import KINDS
from earthling.core.scene import DEFAULT_SCENE_NAME, Scene, SceneError
from earthling.data.jobs import download_jobs
from earthling.render.camera_rig import CameraRig
from earthling.session import Session
from earthling.ui import theme
from earthling.ui.dope_sheet import KeyButton
from earthling.ui.inspector import Inspector, search_action
from earthling.ui.timeline_widget import TimelineWidget

MAX_RECENT = 8


class SceneOriginCommand(QUndoCommand):
    def __init__(self, window, before, after) -> None:
        super().__init__("Scene frame")
        self.window, self.before, self.after = window, before, after

    def redo(self) -> None:
        self.window._reframe(self.after)

    def undo(self) -> None:
        self.window._reframe(self.before)


class MainWindow(QMainWindow):
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
        self.setCentralWidget(self.viewport)
        self.viewport.set_store(self.scene.store)
        self.viewport.pose_provider = self._animated_pose
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.viewport.shutdown)
        self._build_docks()
        self.viewport.timeline_time_provider = lambda: self.timeline.time
        self._pois_changed()
        self._connect_viewport()
        self._update_title()
        self.data_changed.connect(self._reload_terrain)
        self.statusBar().showMessage("Ready")

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
        self.timeline_widget = TimelineWidget(self.timeline, self.keys)
        self.timeline_dock = self._add_dock(
            bottom, QDockWidget("Timeline", self), "TimelineDock", self.timeline_widget
        )
        self.timeline.settings_changed.connect(lambda: self.scene.mark_dirty())
        self.tracks_dock = self._add_dock(bottom, TracksDock(self))
        self.tabifyDockWidget(self.timeline_dock, self.tracks_dock)
        self.timeline_dock.raise_()
        self._install_timeline_shortcuts()
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

    # --- menus -------------------------------------------------------------------------
    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.file_menu = bar.addMenu("&File")
        self._add_action(
            self.file_menu, "&New Project…", self._new_project, QKeySequence.StandardKey.New,
            "file-plus",
        )  # fmt: skip
        self.open_action = self._add_action(
            self.file_menu, "&Open Project Folder…", self._open_project_dialog,
            QKeySequence.StandardKey.Open, "folder-open",
        )  # fmt: skip
        self.recent_menu = self.file_menu.addMenu("Open &Recent")
        self._add_action(self.file_menu, "Re&load Project", self._reload_project, "F5",
                         "arrows-clockwise")  # fmt: skip
        self.settings_action = self._add_action(
            self.file_menu, "Project Se&ttings…", self.edit_settings, "Ctrl+,", "gear-six"
        )
        self._add_action(self.file_menu, "New &Line…", self.new_line, None, "path")
        self.file_menu.addSeparator()
        self._add_action(self.file_menu, "Open &Scene…", self._open_scene_dialog)
        self.save_action = self._add_action(
            self.file_menu, "&Save Scene", lambda: self.save_scene(), QKeySequence.StandardKey.Save,
            "floppy-disk",
        )  # fmt: skip
        self._add_action(self.file_menu, "Save Scene &As…", self.save_scene_as, "Ctrl+Shift+S")
        self._add_action(self.file_menu, "Re&vert Scene", self._revert_scene)
        self.file_menu.addSeparator()
        self.still_action = self._add_action(
            self.file_menu, "Render &Still…", self.render_still_dialog, "Ctrl+R", "camera"
        )
        self.export_action = self._add_action(
            self.file_menu, "&Export Video…", self.export_video_dialog, "Ctrl+E", "film-strip"
        )
        self.file_menu.addSeparator()
        self._add_action(self.file_menu, "&Quit", self.close, QKeySequence.StandardKey.Quit)
        self._rebuild_recent_menu()

        edit_menu = bar.addMenu("&Edit")
        undo = self.undo_action = self.undo_stack.createUndoAction(self, "&Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        undo.setIcon(theme.icon("arrow-counter-clockwise"))
        redo = self.redo_action = self.undo_stack.createRedoAction(self, "&Redo")
        redo.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        redo.setIcon(theme.icon("arrow-clockwise"))
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)

        self.animation_menu = bar.addMenu("&Animation")
        from earthling.app import generators

        for text, action, icon_name in (
            ("Generate &Camera…", generators.generate_camera, "video-camera"),
            ("&Draw Tracks…", generators.draw_tracks, "path"),
            ("&Sun from the Tracks…", generators.sun_from_tracks, "sun-horizon"),
            ("Add Still &Seconds…", generators.add_still_seconds, "clock-counter-clockwise"),
        ):
            self._add_action(self.animation_menu, text, lambda a=action: a(self), None, icon_name)
        self.animation_menu.addSeparator()
        self._add_action(self.animation_menu, "&Check Animation…", self.check_animation, None,
                         "check")  # fmt: skip

        self.view_menu = bar.addMenu("&View")
        self.frame_all_action = self._add_action(
            self.view_menu, "&Frame All", lambda: self.viewport.frame_all(), "Home", "corners-out"
        )
        self._add_action(self.view_menu, "&Go To Coordinate…", self._go_to, "Ctrl+G",
                         "map-trifold")  # fmt: skip
        self.fly_mode_action = QAction("&Fly Camera (Tab)", self, checkable=True)
        self.fly_mode_action.setIcon(theme.tool_icon("airplane-tilt"))
        self.fly_mode_action.toggled.connect(
            lambda v: self.viewport.set_mode(self.viewport.FLY if v else self.viewport.ORBIT)
        )
        self.view_menu.addAction(self.fly_mode_action)
        self.animated_camera_action = QAction("&Animated Camera (C)", self, checkable=True)
        self.animated_camera_action.setShortcut(QKeySequence("C"))
        self.animated_camera_action.setIcon(theme.tool_icon("video-camera"))
        self.animated_camera_action.setToolTip(
            "Look through the keyframed camera; manual navigation switches back to the free camera"
        )
        self.animated_camera_action.toggled.connect(lambda v: self.viewport.set_animated_camera(v))
        self.view_menu.addAction(self.animated_camera_action)
        self.camera_key_action = self._add_action(
            self.view_menu, "Add &Camera Key (K)", self.add_camera_key, "K", "diamond"
        )
        self.pick_target_action = self._add_action(
            self.view_menu, "Pick Look-at &Target (T)", self.pick_look_at_target, "T", "crosshair"
        )
        self.view_menu.addSeparator()
        self._add_action(self.view_menu, "Centre Scene &Frame on View",
                         lambda: self.set_scene_origin(self._view_centre()))  # fmt: skip
        self._add_action(self.view_menu, "&Reset Scene Frame", lambda: self.set_scene_origin(None))
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
        self._add_action(self.data_menu, "Download &Plan…", self._show_plan, None, "list-bullets")
        self._add_action(self.data_menu, "&Download Data…", self._show_download, "Ctrl+D",
                         "cloud-arrow-down")  # fmt: skip
        self._add_action(self.data_menu, "&Test a Source…", self._test_source, None, "plug")
        self.data_menu.addAction(self.on_demand_action)

        help_menu = bar.addMenu("&Help")
        self.palette_action = self._add_action(
            help_menu, "&Command Palette…", self.show_palette, "Ctrl+K", "magnifying-glass"
        )
        self._add_action(help_menu, "&Keyboard Shortcuts", self.show_shortcuts, "F1", "keyboard")
        help_menu.addSeparator()
        self._add_action(help_menu, "&About Earthling", self._show_about, None, "info")

    def _build_toolbar(self) -> None:
        """File, undo and camera actions as icons; Still and Export on the right."""
        bar = self.toolbar = QToolBar("Toolbar", self)
        bar.setObjectName("MainToolbar")
        bar.setMovable(False)
        bar.setIconSize(QSize(18, 18))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        groups = (
            (self.open_action, self.save_action),
            (self.undo_action, self.redo_action),
            (self.animated_camera_action, self.camera_key_action, self.pick_target_action),
        )
        for i, actions in enumerate(groups):
            if i:
                bar.addSeparator()
            for action in actions:
                bar.addAction(action)
        gap = QWidget()
        gap.setFixedWidth(10)
        bar.addWidget(gap)
        pill = QPushButton(theme.icon("magnifying-glass", theme.TEXT_3),
                           "Search settings and actions…        Ctrl K")  # fmt: skip
        pill.setObjectName("commandPill")
        pill.setIconSize(QSize(15, 15))
        pill.setMinimumWidth(300)
        pill.setToolTip("Command palette: every setting, action, look and help (Ctrl+K)")
        pill.clicked.connect(lambda: self.show_palette())
        bar.addWidget(pill)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        for action, accent in ((self.still_action, False), (self.export_action, True)):
            text = action.text().replace("&", "")
            name = "film-strip" if accent else "camera"
            ink = "#06101F" if accent else theme.TEXT_2  # dark on the accent button
            label = text.rstrip("…").replace("Render ", "")
            button = QPushButton(theme.icon(name, ink, ink), label)
            button.setProperty("accent", accent)
            button.setIconSize(QSize(16, 16))
            button.setToolTip(f"{text} ({action.shortcut().toString()})")
            button.clicked.connect(action.trigger)
            bar.addWidget(button)
            if not accent:
                gap = QWidget()
                gap.setFixedWidth(6)
                bar.addWidget(gap)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

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

    # --- points of interest ---------------------------------------------------------------
    def _scene_dir(self) -> Path | None:
        if self.scene.path is not None:
            return self.scene.path.parent
        return self.project.folder if self.project else None

    def _pois_changed(self) -> None:
        self.viewport.set_pois(self.scene.pois, self._scene_dir(), self.scene.animation)
        self.viewport.set_photos(self.scene.photos, self._scene_dir())

    # --- command palette ------------------------------------------------------------------
    def show_palette(self, text: str = "") -> None:
        from earthling.ui.palette import CommandPalette, mark_recent, menu_entries

        if getattr(self, "command_palette", None) is None:
            mark_recent(self.export_action, self.still_action, self.camera_key_action,
                        self.open_action, self.save_action)  # fmt: skip
            self.command_palette = CommandPalette(
                [lambda: menu_entries(self.menuBar()), self._setting_entries,
                 self._look_entries, self._help_entries], self)  # fmt: skip
        self.command_palette.open(text)

    def _setting_entries(self) -> list:
        from earthling.ui.inspector import category_of, section_display
        from earthling.ui.palette import Entry
        from earthling.ui.property_panel import format_value, has_editor

        out = []
        store = self.scene.store
        for d in store.registry:
            if not has_editor(d):
                continue
            category = category_of(d.section)
            crumb = (f"{category.title} › {section_display(d.section)}" if category
                     else d.section)  # fmt: skip
            key = (lambda pid=d.id: self.keys.set_key(pid, store[pid])) if d.animatable else None
            out.append(Entry("Settings", d.label, crumb, category.icon if category else "map-pin",
                             lambda pid=d.id: self.inspector.reveal(pid),
                             format_value(d, store[d.id]), f"{d.tooltip} {d.keywords} {d.id}",
                             key))  # fmt: skip
        return out

    def _look_entries(self) -> list:
        from earthling.ui.inspector import CATEGORIES
        from earthling.ui.palette import Entry

        titles = {c.id: c.title for c in CATEGORIES}
        return [Entry("Looks", look.name, f"Look · {titles[cid]} (undoable)", look.icon,
                      look.apply, "Look", look.tooltip)
                for cid, looks in self.looks.items() for look in looks]  # fmt: skip

    def _help_entries(self) -> list:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        from earthling.ui.palette import Entry

        docs = Path(__file__).resolve().parents[3] / "docs"
        out = [Entry("Help", "Keyboard shortcuts", "Help", "keyboard", self.show_shortcuts, "F1")]
        for name, title in (("configuration.md", "earthling.toml: every key explained"),
                            ("adding-a-region.md", "Adding a region (new data sources)"),
                            ("architecture.md", "How Earthling is built")):  # fmt: skip
            path = docs / name
            if path.exists():
                out.append(Entry("Help", title, f"docs/{name}", "book-open",
                                 lambda p=path: QDesktopServices.openUrl(QUrl.fromLocalFile(
                                     str(p)))))  # fmt: skip
        return out

    def show_shortcuts(self) -> None:
        """A list of every keyboard shortcut."""
        from earthling.ui.palette import menu_entries

        rows = [(e.right, f"{e.title}  ({e.crumb})") for e in menu_entries(self.menuBar())
                if e.right]  # fmt: skip
        rows += [("Space", "Play / pause"), (", .", "Previous / next frame"),
                 ("Tab", "Orbit / fly camera"), ("Ctrl+Home / End", "Start / end"),
                 ("Ctrl+F", "Search the settings")]  # fmt: skip
        text = "\n".join(f"{key:<18} {what}" for key, what in sorted(rows))
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setText("<pre style='font-family: Inter'>" + text.replace("<", "&lt;") + "</pre>")
        box.exec()

    def _reveal_item(self, title: str) -> bool:
        """Select the POI or photo that owns the section ``title``."""
        for kind, items, prefix in (("poi", self.scene.pois, "POI: "),
                                    ("photo", self.scene.photos, "Photo: ")):  # fmt: skip
            for i, item in enumerate(items):
                if title == f"{prefix}{item.name}":
                    self.scene_list.select((kind, i))
                    return True
        return False

    def _select_item(self, kind: str, index: int) -> None:
        """An item was added or removed: select it (or its neighbour) in the scene list."""
        self.scene_list.refresh()
        if index >= 0:
            self.scene_list.select((kind, index))
        elif self.inspector.item is not None:
            self.inspector.show_category(self.inspector.category)

    def _scene_selected(self, key: tuple) -> None:
        """The inspector shows what was selected in the scene list."""
        kind = key[0]
        if kind == "camera":
            self.inspector.show_category("camera")
        elif kind == "layers":
            self.inspector.show_category("layers")
        elif kind == "header":
            category = {"Tracks": "tracks", "Map layers": "layers"}.get(key[1])
            if category:
                self.inspector.show_category(category)
        elif kind in ("group", "track"):
            self._show_group(key)
        elif kind in ("poi", "photo"):
            self.show_scene_item(kind, key[1])

    def _show_group(self, key: tuple) -> None:
        session = self.session
        if session is None:
            return
        if key[0] == "track":
            track = self.tracks[key[1]]
            group = next(g for g in session.groups if track in g.tracks)
            s = track.stats
            title = track.name
            subtitle = (f"Track of {group.label} · {s.distance_m / 1000:.1f} km · "
                        f"↑ {s.ascent_m:.0f} m")  # fmt: skip
        else:
            group = session.group(key[1])
            total = sum(t.stats.distance_m for t in group.tracks)
            title = group.label
            subtitle = f"Track group · {len(group.tracks)} tracks · {total / 1000:.1f} km"
        sections = [f"Tracks: {group.label}", f"Tracks: {group.label} – colours"]
        self.inspector.show_item(title, subtitle, "path", sections)

    def show_scene_item(self, kind: str, index: int) -> None:
        """The inspector shows the form and the settings of a POI or a photo."""
        editor = self.poi_editor if kind == "poi" else self.photo_editor
        items = editor.items()
        if not 0 <= index < len(items):
            return
        item = items[index]
        editor.select(index)
        if kind == "poi":
            what = next((k.name for k in KINDS if k.icon == item.icon), "custom icon")
            self.inspector.show_item(item.name, f"Point of interest · {what}", "map-pin",
                                     [f"POI: {item.name}"], top=editor)  # fmt: skip
        else:
            self.inspector.show_item(item.name, f"Photo · {item.path}", "image",
                                     [f"Photo: {item.name}"], top=editor)  # fmt: skip

    def add_gpx_dialog(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "Add GPX files", "Open or create a project first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add GPX files", str(Path.home()),
                                                "GPX files (*.gpx)")  # fmt: skip
        self.add_gpx_files([Path(p) for p in paths])

    def add_gpx_files(self, paths: list[Path]) -> None:
        """Copy GPX files into the project's GPX folder and load them."""
        import shutil

        if self.project is None or not paths:
            return
        folder = self.project.gpx_dir
        if not folder.is_dir():  # (a single file or a pattern: the files go next to it)
            folder = folder.parent
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            target = folder / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
        self._reload_project()

    def drop_paths(self, paths: list[Path]) -> None:
        """Dropped files: a project folder opens, GPX files join the project, pictures
        become photos."""
        for folder in [p for p in paths if p.is_dir()]:
            if (folder / "earthling.toml").exists():
                self.open_project(folder)
                return
        gpx = [p for p in paths if p.suffix.lower() == ".gpx"]
        if gpx:
            self.add_gpx_files(gpx)
        for picture in [p for p in paths if p.suffix.lower() in PHOTO_SUFFIXES]:
            self.photo_editor.add_photo(picture)

    def hiker_lonlat(self) -> tuple[float, float] | None:
        """(lon, lat) of the head of the drawn track."""
        renderer = self.viewport.renderer
        if renderer is None or renderer.tracks.path is None or self.session is None:
            return None
        s = self.scene.store
        path = renderer.tracks.path
        enu = path.position_at(path.distance_for(s["progress.head"], s["progress.mode"]))
        if enu is None:
            return None
        lat, lon, _ = self.session.frame.enu_to_geodetic(enu)
        return float(lon), float(lat)

    def lonlat_at_time(self, when) -> tuple[float, float] | None:
        """(lon, lat) of the walked tracks at a time (naive: the project's local time)."""
        from zoneinfo import ZoneInfo

        renderer = self.viewport.renderer
        if renderer is None or renderer.tracks.path is None or self.session is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=ZoneInfo(self.session.config.project.timezone))
        path = renderer.tracks.path
        t = path.time[np.isfinite(path.time)]
        epoch = when.timestamp()
        if not len(t) or not t[0] - 3600.0 <= epoch <= t[-1] + 3600.0:
            return None  # (not during the tracks)
        enu = path.position_at(path.distance_for_time(epoch))
        lat, lon, _ = self.session.frame.enu_to_geodetic(enu)
        return float(lon), float(lat)

    def built_tracks(self) -> list:
        """The tracks as drawn (BuiltTrack: ENU points, distances, times), [] before a project."""
        renderer = self.viewport.renderer
        return list(renderer.tracks._gpu) if renderer is not None else []

    def choose_place(self, title: str, start: tuple[float, float] | None):
        """(lon, lat) from the place dialog (coordinates or a track point), None if cancelled."""
        from earthling.app.place_dialog import PlaceDialog

        if self.session is None:
            return None
        dialog = PlaceDialog(title, self.built_tracks(), self.session.frame, start, self)
        return dialog.lonlat() if dialog.exec() else None

    def _pick_lonlat(self, callback) -> None:
        self.statusBar().showMessage("Click on the terrain to place the POI", 5000)

        def picked(point) -> None:
            if self.session is not None:
                lat, lon, _ = self.session.frame.enu_to_geodetic(point)
                callback(float(lon), float(lat))

        self.viewport.request_pick(picked)

    def pick_look_at_target(self) -> None:
        self.statusBar().showMessage("Click on the terrain to set the look-at target", 5000)

        def picked(point) -> None:
            self._on_panel_edit("camera.target", tuple(float(v) for v in point), False)
            self.statusBar().showMessage("Look-at target set", 3000)

        self.viewport.request_pick(picked)

    def _animated_pose(self):
        renderer = self.viewport.renderer
        self.rig.path = renderer.tracks.path if renderer is not None else None
        return self.viewport.clamp_pose(self.rig.pose(self.timeline.time))

    def _camera_keys_changed(self) -> None:
        curve = self.scene.animation.curves.get("camera.pose")
        self.viewport.set_camera_path(camera_path_lines(curve))

    def _install_timeline_shortcuts(self) -> None:
        for key, slot in (
            ("Space", self.timeline.toggle),
            (",", lambda: self.timeline.step(-1)),
            (".", lambda: self.timeline.step(1)),
            ("Ctrl+Home", self.timeline.go_start),
            ("Ctrl+End", self.timeline.go_end),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(slot)

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

    def _add_action(self, menu, text, slot, shortcut=None, icon_name: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if icon_name:
            action.setIcon(theme.icon(icon_name))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

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

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self._recent()
        for path in recent:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _=False, p=path: self.open_project(p))
        self.recent_menu.setEnabled(bool(recent))

    def _update_title(self) -> None:
        if sip.isdeleted(self):  # (the undo stack turns clean while the window is destroyed)
            return
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
        # editors must not report edits while Qt destroys them (focus moves, lists clear)
        for widget in (self.inspector, self.poi_editor, self.photo_editor, self.scene_list):
            for child in widget.findChildren(QWidget):
                child.blockSignals(True)
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(self, "About Earthling", f"Earthling {__version__}")
