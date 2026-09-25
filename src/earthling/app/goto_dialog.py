"""Dialog to jump the camera to a coordinate."""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout


class GoToDialog(QDialog):
    def __init__(self, lat: float, lon: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Go to coordinate")
        form = QFormLayout(self)
        self.lat = self._spin(-85, 85, lat, 6, "°")
        self.lon = self._spin(-180, 180, lon, 6, "°")
        self.height = self._spin(0, 100_000, 500, 0, " m")
        form.addRow("Latitude", self.lat)
        form.addRow("Longitude", self.lon)
        form.addRow("Height above ground", self.height)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _spin(lo, hi, value, decimals, suffix) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def values(self) -> tuple[float, float, float]:
        return self.lat.value(), self.lon.value(), self.height.value()
