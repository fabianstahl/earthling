"""Project settings: ``earthling.toml`` in a text editor, checked while typing (tracks, detail
zones, data sources, distance marks, own providers: docs/configuration.md)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from earthling.core.config import Config, ConfigError


def check_settings(text: str) -> str:
    """'' when ``text`` is a valid configuration, else the problems."""
    try:
        Config.from_toml(text)
    except ConfigError as exc:
        return str(exc)
    return ""


class SettingsDialog(QDialog):
    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self.path = path
        self.setWindowTitle(f"Project settings – {path}")
        self.resize(760, 640)
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.editor.setPlainText(path.read_text(encoding="utf-8"))
        self.status = QLabel()
        self.status.setWordWrap(True)
        hint = QLabel("Every key is described in docs/configuration.md. Saving reloads the "
                      "project (tracks, data sources) and keeps the open scene.")  # fmt: skip
        hint.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                        | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save and Reload")
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        for widget in (hint, self.editor, self.status, self.buttons):
            layout.addWidget(widget)
        self._timer = QTimer(self, singleShot=True, interval=300)
        self._timer.timeout.connect(self.check)
        self.editor.textChanged.connect(self._timer.start)
        self.check()

    def check(self) -> bool:
        problems = check_settings(self.editor.toPlainText())
        self.status.setText(problems or "The settings are valid.")
        self.status.setStyleSheet("color: #c0392b" if problems else "color: #2e7d32")
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(not problems)
        return not problems

    def _save(self) -> None:
        if self.check():
            self.path.write_text(self.editor.toPlainText(), encoding="utf-8")
            self.accept()
