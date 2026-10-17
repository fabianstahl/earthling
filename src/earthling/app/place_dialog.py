"""Dialog choosing a place: coordinates, or a point of a track (start, end, highest or lowest
point, or a walked distance)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QRadioButton,
    QVBoxLayout,
)

from earthling.render.track_geometry import track_point
from earthling.ui.widgets import double_spin, row

POINTS = [("start", "Start"), ("end", "End"), ("highest", "Highest point"),
          ("lowest", "Lowest point"), ("distance", "At distance")]  # fmt: skip


class PlaceDialog(QDialog):
    """``tracks``: the built tracks (BuiltTrack) to choose from; ``frame``: their LocalFrame;
    ``start``: (lon, lat) shown in the coordinate fields."""

    def __init__(self, title: str, tracks: list, frame, start: tuple[float, float] | None,
                 parent=None) -> None:  # fmt: skip
        super().__init__(parent)
        self.setWindowTitle(title)
        self.tracks, self.frame = tracks, frame
        lon, lat = start or (0.0, 0.0)
        self.lat = double_spin(-90.0, 90.0, decimals=6, suffix="°", value=lat)
        self.lon = double_spin(-180.0, 180.0, decimals=6, suffix="°", value=lon)
        self.track = QComboBox()
        for g in tracks:
            self.track.addItem(g.track.name)
        self.point = QComboBox()
        for key, label in POINTS:
            self.point.addItem(label, key)
        self.km = double_spin(0.0, 100000.0, decimals=2, suffix=" km", value=10.0)
        self.by_coords = QRadioButton("Coordinates")
        self.on_track = QRadioButton("On a track")
        group = QButtonGroup(self)
        for button in (self.by_coords, self.on_track):
            group.addButton(button)
        (self.on_track if tracks else self.by_coords).setChecked(True)
        self.on_track.setEnabled(bool(tracks))
        form = QFormLayout()
        form.addRow(self.by_coords, row((self.lat, 1), (self.lon, 1)))
        form.addRow(self.on_track, row((self.track, 2), (self.point, 1), self.km))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.point.currentIndexChanged.connect(self._update)
        group.buttonToggled.connect(self._update)
        self._update()

    def _update(self, *_args) -> None:
        coords = self.by_coords.isChecked()
        for w in (self.lat, self.lon):
            w.setEnabled(coords)
        for w in (self.track, self.point):
            w.setEnabled(not coords)
        self.km.setEnabled(not coords and self.point.currentData() == "distance")

    def lonlat(self) -> tuple[float, float]:
        """The chosen place."""
        if self.by_coords.isChecked() or not self.tracks:
            return self.lon.value(), self.lat.value()
        g = self.tracks[self.track.currentIndex()]
        enu = track_point(g, self.point.currentData(), self.km.value())
        lat, lon, _ = self.frame.enu_to_geodetic(enu)
        return float(lon), float(lat)
