"""Dialog to jump the camera to a coordinate."""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

from earthling.ui.widgets import double_spin


class GoToDialog(QDialog):
    def __init__(self, lat: float, lon: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Go to coordinate")
        form = QFormLayout(self)
        self.lat = double_spin(-85, 85, decimals=6, suffix="°", value=lat)
        self.lon = double_spin(-180, 180, decimals=6, suffix="°", value=lon)
        self.above_ground = double_spin(0, 100_000, decimals=0, suffix=" m", value=500)
        form.addRow("Latitude", self.lat)
        form.addRow("Longitude", self.lon)
        form.addRow("Height above ground", self.above_ground)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[float, float, float]:
        return self.lat.value(), self.lon.value(), self.above_ground.value()
