"""Qt application bootstrap."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from earthling.app.main_window import MainWindow
from earthling.app.viewport import configure_default_surface_format


def run_gui(project: str | None = None, dev_mode: bool = False) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_default_surface_format()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Earthling")
    app.setOrganizationName("Earthling")
    window = MainWindow(dev_mode=dev_mode)
    window.show()
    if project:
        window.open_project(project)
    return app.exec()
