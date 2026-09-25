"""Dock listing the loaded GPX tracks with their statistics."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem

from earthling.core.gpx import Track

COLUMNS = ["Track", "Date", "Distance", "Ascent", "Descent", "Max ele.", "Duration", "Points"]


def fmt_km(m: float) -> str:
    return f"{m / 1000:.1f} km"


def fmt_m(m: float | None) -> str:
    return "–" if m is None else f"{m:.0f} m"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    minutes = int(seconds // 60)
    return f"{minutes // 60}:{minutes % 60:02d} h"


class TracksDock(QDockWidget):
    visibility_changed = pyqtSignal()
    track_activated = pyqtSignal(int)  # double-click: index into the track list

    def __init__(self, parent=None) -> None:
        super().__init__("Tracks", parent)
        self.setObjectName("TracksDock")
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.setWidget(self.tree)
        self._tracks: list[Track] = []

    def set_tracks(self, tracks: list[Track]) -> None:
        self._tracks = tracks
        self.tree.blockSignals(True)
        self.tree.clear()
        total_dist = total_up = total_down = 0.0
        for index, track in enumerate(tracks):
            s = track.stats
            total_dist += s.distance_m
            total_up += s.ascent_m
            total_down += s.descent_m
            item = QTreeWidgetItem(
                [
                    track.name,
                    s.start_time.strftime("%Y-%m-%d") if s.start_time else "–",
                    fmt_km(s.distance_m),
                    fmt_m(s.ascent_m),
                    fmt_m(s.descent_m),
                    fmt_m(s.max_ele_m),
                    fmt_duration(s.duration_s),
                    str(s.point_count),
                ]
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                0, Qt.CheckState.Checked if track.visible else Qt.CheckState.Unchecked
            )
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            item.setToolTip(0, str(track.source))
            self.tree.addTopLevelItem(item)
        if tracks:
            total = QTreeWidgetItem(
                [
                    f"Total ({len(tracks)})",
                    "",
                    fmt_km(total_dist),
                    fmt_m(total_up),
                    fmt_m(total_down),
                ]
            )
            font = QFont()
            font.setBold(True)
            for col in range(len(COLUMNS)):
                total.setFont(col, font)
            self.tree.addTopLevelItem(total)
        for col in range(len(COLUMNS)):
            self.tree.resizeColumnToContents(col)
        self.tree.blockSignals(False)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is not None:
            self.track_activated.emit(index)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is None or column != 0:
            return
        self._tracks[index].visible = item.checkState(0) == Qt.CheckState.Checked
        self.visibility_changed.emit()
