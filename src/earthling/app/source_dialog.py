"""Test a data source (imagery or elevation) at a place: does it deliver, how fast, does it fit
the other sources (the report of ``earthling check-source``) and a strip of its tiles."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from earthling.data.check import check_dem, check_imagery, default_location
from earthling.data.dem import DEM_SOURCES
from earthling.data.providers import PROVIDERS, TileProvider
from earthling.ui.widgets import row


class SourceDialog(QDialog):
    def __init__(self, session, choose_place, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Test a data source")
        self.resize(780, 560)
        self.session, self.choose_place = session, choose_place
        self.source = QComboBox()
        for sid, source in sorted({**PROVIDERS, **DEM_SOURCES}.items()):
            kind = "elevation" if sid in DEM_SOURCES else source.kind
            self.source.addItem(f"{sid}  ({kind})", sid)
        self.place: tuple[float, float] | None = None  # (lon, lat); None: automatic
        self.place_label = QLabel("A track point inside the source's coverage")
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose)
        self.run_button = QPushButton("Test")
        self.run_button.clicked.connect(self.run)
        self.report = QPlainTextEdit(readOnly=True)
        self.report.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.tiles = QLabel()
        self.tiles.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form = QFormLayout()
        form.addRow("Source", self.source)
        form.addRow("Where", row((self.place_label, 1), choose))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.run_button)
        layout.addWidget(self.report, 1)
        layout.addWidget(self.tiles)
        layout.addWidget(buttons)

    def _choose(self) -> None:
        place = self.choose_place("Test the source at…", self.place)
        if place is not None:
            self.place = place
            self.place_label.setText(f"{place[1]:.5f}, {place[0]:.5f}")

    def run(self) -> str:
        """Test the chosen source (downloads a few tiles); returns the report text."""
        sid = self.source.currentData()
        source = PROVIDERS.get(sid) or DEM_SOURCES[sid]
        lon, lat = self.place or default_location(self.session, source)
        out = self.session.cache.root / "_checks"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if isinstance(source, TileProvider):
                report = check_imagery(source, lon, lat, out_dir=out)
            else:
                report = check_dem(source, lon, lat, self.session.cache)
        finally:
            QApplication.restoreOverrideCursor()
        text = report.text()
        self.report.setPlainText(text)
        sheet = out / f"check_{sid}.png"
        if isinstance(source, TileProvider) and sheet.exists():
            pixmap = QPixmap(str(sheet))
            self.tiles.setPixmap(
                pixmap.scaledToHeight(160, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self.tiles.clear()
        return text
