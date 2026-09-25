"""Main application window."""

from __future__ import annotations

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QLabel, QMainWindow, QMessageBox

from earthling import __version__
from earthling.app.viewport import Viewport


class MainWindow(QMainWindow):
    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Earthling")
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
        self.statusBar().showMessage("Ready")

    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.file_menu = bar.addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        self.file_menu.addAction(quit_action)

        self.view_menu = bar.addMenu("&View")

        help_menu = bar.addMenu("&Help")
        about = QAction("&About Earthling", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _show_about(self) -> None:
        QMessageBox.about(self, "About Earthling", f"Earthling {__version__}")
