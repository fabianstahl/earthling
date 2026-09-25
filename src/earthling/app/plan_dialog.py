"""Dialog showing the tile download plan."""

from __future__ import annotations

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

from earthling.core.aoi import plan_report
from earthling.core.session import Session


class PlanDialog(QDialog):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Download plan")
        self.resize(520, 480)
        layout = QVBoxLayout(self)
        area = session.config.area
        zones = ", ".join(
            f"≤{z.within_km:g} km: imagery z{z.imagery_zoom} / dem z{z.dem_zoom}"
            for z in area.zones
        )
        layout.addWidget(QLabel(f"Border {area.border_km:g} km — zones: {zones}"))
        text = QPlainTextEdit(plan_report(session.plan))
        text.setReadOnly(True)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
