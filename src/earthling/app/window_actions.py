"""Menus, toolbar, shortcuts and keyboard navigation of the main window."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QWidget,
)

from earthling import __version__
from earthling.app.goto_dialog import GoToDialog
from earthling.ui import theme


class ActionsMixin:
    """Part of :class:`earthling.app.main_window.MainWindow` (its ``self``)."""

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
        self._add_action(self.file_menu, "Start Scree&n", lambda: self.show_start(True), None,
                         "house-line")  # fmt: skip
        self._add_action(self.file_menu, "Re&load Project", self._reload_project, "F5",
                         "arrows-clockwise")  # fmt: skip
        self.settings_action = self._add_action(
            self.file_menu, "Project Se&ttings…", self.edit_settings, "Ctrl+,", "gear-six"
        )
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
        redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        redo.setIcon(theme.icon("arrow-clockwise"))
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)

        # what the scene list's "+" adds, also here: in the palette and without a mouse
        add_menu = bar.addMenu("&Add")
        for text, slot, icon_name in (
            ("&GPX Files…", lambda: self.add_gpx_dialog(), "file-plus"),
            ("New &Line (Flight, Ferry, Road)…", lambda: self.new_line(), "path"),
            ("POI at the &Marker", lambda: self.poi_editor.add_at_hiker(), "map-pin"),
            ("POI Where I &Click…", lambda: self.poi_editor.add_by_click(), "crosshair"),
            ("POI at a &Place…", lambda: self.poi_editor.add_at_place(), "magnifying-glass"),
            ("P&hotos…", lambda: self.photo_editor.add_dialog(), "image"),
        ):
            self._add_action(add_menu, text, slot, None, icon_name)

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

        # the timeline's keys as menu entries: in the command palette and in F1's list
        playback = bar.addMenu("&Playback")
        for text, slot, key, icon_name in (
            ("&Play / Pause", self.timeline.toggle, "Space", "play"),
            ("Previous &Frame", lambda: self.timeline.step(-1), ",", "caret-left"),
            ("&Next Frame", lambda: self.timeline.step(1), ".", "caret-right"),
            ("Previous &Key", lambda: self.timeline.step_key(-1), "Alt+,", "key-previous"),
            ("Next K&ey", lambda: self.timeline.step_key(1), "Alt+.", "key-next"),
            ("Go to &Start", self.timeline.go_start, "Ctrl+Home", "skip-back"),
            ("Go to &End", self.timeline.go_end, "Ctrl+End", "skip-forward"),
        ):
            self._add_action(playback, text, slot, key, icon_name)

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
        self.palette_action.setShortcuts([QKeySequence("Ctrl+K"), QKeySequence("Ctrl+Shift+P")])
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

    def _add_action(self, menu, text, slot, shortcut=None, icon_name: str = "") -> QAction:
        """A menu entry calling ``slot()`` - without the ``checked`` flag of ``triggered``,
        which would land in a slot's first optional parameter (show_palette(text) got False)."""
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if icon_name:
            action.setIcon(theme.icon(icon_name))
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

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

    def _add_keyboard_navigation(self) -> None:
        """Ctrl+1 .. Ctrl+0: an inspector category; F6 / Shift+F6: the next / previous pane
        (the viewport keeps Tab for its camera mode)."""
        from earthling.ui.inspector import CATEGORIES

        menu = self.view_menu.addMenu(theme.icon("squares-four"), "Inspector &Category")
        for category, digit in zip(CATEGORIES, "1234567890", strict=False):
            shortcut = f"Ctrl+{digit}"
            self._add_action(menu, category.title.replace("&", "&&"),
                             lambda c=category.id: self.show_inspector_category(c), shortcut,
                             category.icon)  # fmt: skip
            button = self.inspector.rail.buttons[category.id]
            button.setToolTip(f"{button.toolTip()} ({shortcut})")
        self._add_action(self.view_menu, "Next &Pane", lambda: self.focus_pane(1), "F6")
        self._add_action(self.view_menu, "Previous Pane", lambda: self.focus_pane(-1), "Shift+F6")

    def show_inspector_category(self, category_id: str) -> None:
        self.parameters_dock.show()
        self.inspector.show_category(category_id)

    def focus_pane(self, step: int) -> None:
        """Move the keyboard focus to the next (``step`` 1) or previous visible pane: the scene
        list, the viewport, the inspector (its search field), the timeline."""
        panes = [
            (self.scene_list, self.scene_list.tree),
            (self.viewport, self.viewport),
            (self.inspector, self.inspector.search),
            (self.timeline_widget, self.timeline_widget.dope_sheet or self.timeline_widget),
        ]
        panes = [(pane, target) for pane, target in panes if pane.isVisible()]
        if not panes:
            return
        focus = self.focusWidget()  # (the window's: also while another window is active)
        current = next((i for i, (pane, _) in enumerate(panes)
                        if focus is not None and (focus is pane or pane.isAncestorOf(focus))),
                       -1)  # fmt: skip
        panes[(current + step) % len(panes)][1].setFocus(Qt.FocusReason.OtherFocusReason)

    def show_shortcuts(self) -> None:
        """A list of every keyboard shortcut."""
        from earthling.ui.palette import menu_entries

        rows = [(e.right, f"{e.title}  ({e.crumb})") for e in menu_entries(self.menuBar())
                if e.right]  # fmt: skip
        rows += [("Tab", "Orbit / fly camera (in the viewport)"),
                 ("W A S D, arrows", "Fly (in the viewport; Q E or Page Up / Down: down, up)"),
                 ("Up / Down", "Change a value (in a field; Shift: finer, Ctrl: coarser)"),
                 ("Enter, F2", "Type a value (in a field)"),
                 ("Delete", "Delete the selected keys, POI or photo"),
                 ("Ctrl+A", "Select all keys (in the timeline)"),
                 ("Esc", "Close a dialog or the palette")]  # fmt: skip
        text = "\n".join(f"{key:<18} {what}" for key, what in sorted(rows))
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setText("<pre style='font-family: Inter'>" + text.replace("<", "&lt;") + "</pre>")
        box.exec()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self._recent()
        for path in recent:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _=False, p=path: self.open_project(p))
        self.recent_menu.setEnabled(bool(recent))

    def _go_to(self) -> None:
        geo = self.viewport.camera_geodetic() or (46.0, 7.0, 0.0)
        dialog = GoToDialog(geo[0], geo[1], self)
        if dialog.exec():
            self.viewport.go_to(*dialog.values())

    def _show_about(self) -> None:
        QMessageBox.about(self, "About Earthling", f"Earthling {__version__}")
