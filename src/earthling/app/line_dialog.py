"""New Line: a line between two places (straight, a flight arc or the road route) with its
departure and arrival time, added to the project as a GPX track of a track group."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PyQt6.QtCore import QDateTime
from PyQt6.QtWidgets import (
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from earthling.data.lines import KINDS
from earthling.ui.widgets import row

NEW_GROUP = "__new__"


def group_folder(session, name: str) -> Path | None:
    """The folder new GPX files of track group ``name`` go into (None: the group is one file)."""
    group = session.group(name)
    if group is None:
        return session.project.folder / "gpx" / name
    path = Path(group.definition.gpx).expanduser()
    path = path if path.is_absolute() else session.project.folder / path
    if path.is_dir():
        return path
    return None if path.suffix.lower() == ".gpx" and "*" not in path.name else path.parent


def add_group_text(config_text: str, session, name: str, flight: bool) -> str:
    """``earthling.toml`` with a new walked track group ``name`` (gpx/<name>/) appended; a
    project without groups first gets its implicit one ("tracks") written out."""
    out = config_text.rstrip("\n") + "\n"
    if not session.config.tracks:
        gpx_dir = session.config.project.gpx_dir.as_posix()
        out += f'\n[[tracks]]\nname = "tracks"\ngpx = "{gpx_dir}"\nlabel = "Tracks"\n'
    out += f'\n[[tracks]]\nname = "{name}"\ngpx = "gpx/{name}/"\n'
    if flight:
        out += 'elevation = "gpx"   # flights keep their heights (an arc in the air)\n'
    return out


class LineDialog(QDialog):
    """``choose_place(title, start) -> (lon, lat) | None``; ``groups``: track group names."""

    def __init__(self, groups: list[str], timezone: str, choose_place, start_time: datetime | None,
                 parent=None) -> None:  # fmt: skip
        super().__init__(parent)
        self.setWindowTitle("New Line")
        self.tz = ZoneInfo(timezone)
        self.choose_place = choose_place
        self.places: dict[str, tuple[float, float] | None] = {"from": None, "to": None}
        self.name = QLineEdit("line")
        self.kind = QComboBox()
        for key, label in KINDS:
            self.kind.addItem(label, key)
        self.group = QComboBox()
        for name in groups:
            self.group.addItem(name, name)
        self.group.addItem("New group…", NEW_GROUP)
        self.new_group = QLineEdit("travel")
        when = QDateTime(start_time) if start_time is not None else QDateTime.currentDateTime()
        self.depart, self.arrive = QDateTimeEdit(when), QDateTimeEdit(when.addSecs(3600))
        for edit in (self.depart, self.arrive):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.labels = {key: QLabel("–") for key in self.places}
        form = QFormLayout()
        for key, title in (("from", "From"), ("to", "To")):
            button = QPushButton("Choose…")
            button.clicked.connect(lambda _=False, k=key, t=title: self._choose(k, t))
            form.addRow(title, row((self.labels[key], 1), button))
        form.addRow("Departure", self.depart)
        form.addRow("Arrival", self.arrive)
        form.addRow("Line", self.kind)
        form.addRow("Name", self.name)
        form.addRow("Track group", row((self.group, 1), (self.new_group, 1)))
        form.addRow("", QLabel(f"(times in the project's time zone, {timezone})"))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        self.group.currentIndexChanged.connect(self._update)
        self._update()

    def _choose(self, key: str, title: str) -> None:
        place = self.choose_place(f"{title}…", self.places[key] and self.places[key][::-1])
        if place is not None:
            lon, lat = place
            self.places[key] = (lat, lon)
            self.labels[key].setText(f"{lat:.5f}, {lon:.5f}")
        self._update()

    def _update(self, *_args) -> None:
        self.new_group.setEnabled(self.group.currentData() == NEW_GROUP)
        ok = all(self.places.values())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def group_name(self) -> str:
        if self.group.currentData() != NEW_GROUP:
            return self.group.currentData()
        return re.sub(r"[^a-z0-9_]+", "_", self.new_group.text().lower()).strip("_") or "travel"

    def file_name(self) -> str:
        return re.sub(r"[^\w-]+", "_", self.name.text()).strip("_") or "line"

    def times(self) -> tuple[datetime, datetime]:
        def aware(edit: QDateTimeEdit) -> datetime:
            return edit.dateTime().toPyDateTime().replace(tzinfo=self.tz)

        return aware(self.depart), aware(self.arrive)
