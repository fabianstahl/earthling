"""Qt application bootstrap."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from earthling.app.main_window import MainWindow


def run_gui(project: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Earthling")
    app.setOrganizationName("Earthling")
    window = MainWindow()
    window.show()
    return app.exec()
