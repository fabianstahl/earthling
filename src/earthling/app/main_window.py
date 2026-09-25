"""Main application window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox

from earthling import __version__
from earthling.app.viewport import Viewport
from earthling.core.config import ConfigError, Project

MAX_RECENT = 8


class MainWindow(QMainWindow):
    project_changed = pyqtSignal(object)

    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self.project: Project | None = None
        self.settings = QSettings()
        self.resize(1600, 1000)
        self._build_menus()
        self.viewport = Viewport(dev_mode=dev_mode)
        self.setCentralWidget(self.viewport)
        self._fps_label = QLabel()
        self.statusBar().addPermanentWidget(self._fps_label)
        self.viewport.fps_changed.connect(lambda fps: self._fps_label.setText(f"{fps:5.1f} fps"))
        self.viewport.shader_error.connect(
            lambda msg: self.statusBar().showMessage(f"Shader error: {msg}", 10000)
        )
        self._update_title()
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

        help_menu = bar.addMenu("&Help")
        self._add_action(help_menu, "&About Earthling", self._show_about)

    def _add_action(self, menu, text, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # --- project handling --------------------------------------------------------------
    def open_project(self, folder: str | Path) -> bool:
        try:
            project = Project.load(folder)
        except ConfigError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))
            return False
        self._set_project(project)
        return True

    def _set_project(self, project: Project) -> None:
        self.project = project
        self._remember_recent(project.folder)
        self._update_title()
        self.statusBar().showMessage(f"Opened project {project.folder}", 5000)
        self.project_changed.emit(project)

    def _new_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose (empty) folder for new project")
        if folder:
            self._set_project(Project.create(folder))

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

    def _show_about(self) -> None:
        QMessageBox.about(self, "About Earthling", f"Earthling {__version__}")
