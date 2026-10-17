"""Points of interest dock: add POIs of a kind from the library (at the hiker, by clicking the
terrain or at a place), edit and remove them.

Structural edits are undoable (snapshots of the POIs with their property values and keys).
The animatable part of every POI appears as its own section in the parameter panel.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSize, QStandardPaths, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QUndoCommand
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from earthling.core.poi_library import KINDS, PoiKind, add_user_image, load_user_images
from earthling.core.pois import Poi, slug
from earthling.core.scene import Scene
from earthling.render.pois import BADGE, badge_image, builtin_icons, resolve_icon
from earthling.ui.widgets import double_spin, row

CUSTOM = "__custom__"
ADD_IMAGE = "__add_image__"


def user_images_file() -> Path:
    """The user's own POI images (shared by all projects)."""
    folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(folder) / "poi_images.json"


def icon_for(ref: str, caption: str = "", base_dir: Path | None = None) -> QIcon:
    """A small picture of an icon reference (a builtin, a file, the text badge, none)."""
    if ref == BADGE:
        image = badge_image(caption or "Text")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
        return QIcon(QPixmap.fromImage(qimage.copy()))
    path = resolve_icon(ref, base_dir)
    return QIcon(QPixmap(str(path))) if path is not None else QIcon()


def poi_snapshot(scene: Scene) -> str:
    """The POIs with the values and keys of their properties (JSON)."""
    props = {pid: scene.registry[pid].to_json(v) for pid, v in scene.store.values.items()
             if pid.startswith("poi.")}  # fmt: skip
    curves = {pid: c.to_json() for pid, c in scene.animation.curves.items()
              if pid.startswith("poi.")}  # fmt: skip
    return json.dumps({"pois": [p.to_json() for p in scene.pois], "props": props,
                       "curves": curves})  # fmt: skip


def restore_poi_snapshot(scene: Scene, snapshot: str) -> None:
    from earthling.core.animation import Curve

    doc = json.loads(snapshot)
    scene.set_pois([Poi.from_json(p) for p in doc["pois"]])
    for pid in [p for p in scene.animation.curves if p.startswith("poi.")]:
        del scene.animation.curves[pid]
    for pid, raw in doc["props"].items():
        if pid in scene.registry:
            scene.store.set(pid, scene.registry[pid].from_json(raw))
    for pid, data in doc["curves"].items():
        if pid in scene.registry:
            scene.animation.curves[pid] = Curve.from_json(scene.registry[pid], data)


class PoiCommand(QUndoCommand):
    def __init__(self, label: str, scene: Scene, before: str, after: str, on_change) -> None:
        super().__init__(label)
        self.scene, self.before, self.after, self.on_change = scene, before, after, on_change
        self._first = True

    def redo(self) -> None:
        if self._first:  # already applied when pushed
            self._first = False
            return
        restore_poi_snapshot(self.scene, self.after)
        self.on_change()

    def undo(self) -> None:
        restore_poi_snapshot(self.scene, self.before)
        self.on_change()


class PoiDock(QDockWidget):
    changed = pyqtSignal()  # POIs edited (the viewport redraws)

    def __init__(self, scene: Scene, undo_stack, hiker_position: Callable, request_pick: Callable,
                 base_dir: Callable[[], Path | None], parent=None,
                 choose_place: Callable | None = None) -> None:  # fmt: skip
        """``hiker_position() -> (lon, lat) | None``; ``request_pick(callback(lon, lat))``;
        ``choose_place(title, (lon, lat) | None) -> (lon, lat) | None`` (a place dialog)."""
        super().__init__("Points of Interest", parent)
        self.setObjectName("poi_dock")
        self.scene = scene
        self.undo_stack = undo_stack
        self.hiker_position = hiker_position
        self.request_pick = request_pick
        self.base_dir = base_dir
        self.choose_place = choose_place
        self.setWidget(self._build_body())
        self._updating = False
        self._connect()
        self.refresh()

    def _build_body(self) -> QWidget:
        """Add/remove buttons, the POI list and the form editing the current POI."""
        body = QWidget()
        layout = QVBoxLayout(body)
        self.kind = QComboBox()
        self.kind.setIconSize(QSize(22, 22))
        self.kind.setToolTip("What the new POI is: its icon, name and caption")
        self._fill_kinds()
        layout.addWidget(row(QLabel("New"), (self.kind, 1)))
        buttons = QHBoxLayout()
        self.add_hiker = QPushButton("+ At hiker")
        self.add_hiker.setToolTip("New POI at the head of the drawn track")
        self.add_click = QPushButton("+ On map…")
        self.add_click.setToolTip("New POI where you click on the terrain")
        self.add_place = QPushButton("+ At…")
        self.add_place.setToolTip(
            "New POI at coordinates or at a point of a track (start, end, "
            "highest or lowest point, a distance)"
        )
        self.remove = QPushButton("Remove")
        for b in (self.add_hiker, self.add_click, self.add_place, self.remove):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.list = QListWidget()
        self.list.setIconSize(QSize(28, 28))
        layout.addWidget(self.list, 1)
        form = QFormLayout()
        self.name = QLineEdit()
        self.caption = QLineEdit()
        self.icon = QComboBox()
        self.icon.addItem("No icon (caption only)", "none")
        self.icon.addItem(icon_for(BADGE), "Text badge (caption inside)", BADGE)
        for name in builtin_icons():
            self.icon.addItem(icon_for(f"builtin:{name}"), name, f"builtin:{name}")
        self.icon.addItem("Custom file…", CUSTOM)
        self.size_px = double_spin(8.0, 512.0, suffix=" px")
        self.lift = double_spin(0.0, 400.0, suffix=" px")
        self.cols = QSpinBox()
        self.rows = QSpinBox()
        for spin in (self.cols, self.rows):
            spin.setRange(1, 64)
        self.fps = double_spin(0.5, 60.0)
        sprite = row(QLabel("cols"), self.cols, QLabel("rows"), self.rows, QLabel("fps"), self.fps)
        self.lat = double_spin(-90.0, 90.0, decimals=6, suffix="°")
        self.lon = double_spin(-180.0, 180.0, decimals=6, suffix="°")
        self.lat.setToolTip("Latitude (north positive)")
        self.lon.setToolTip("Longitude (east positive)")
        self.move_hiker = QPushButton("To hiker")
        self.move_click = QPushButton("Pick…")
        self.move_place = QPushButton("Place…")
        form.addRow("Name", self.name)
        form.addRow("Caption", self.caption)
        form.addRow("Icon", self.icon)
        form.addRow("Size (1080p)", self.size_px)
        form.addRow("Pin height", self.lift)
        form.addRow("Sprite sheet", sprite)
        form.addRow("Position", row((self.lat, 1), (self.lon, 1)))
        form.addRow("", row(self.move_hiker, self.move_click, self.move_place))
        layout.addLayout(form)
        layout.addWidget(QLabel("Animate visibility, effects … in the POI's section of the "
                                "parameter panel."))  # fmt: skip
        return body

    def _fill_kinds(self, select: str | None = None) -> None:
        """The library grouped by category, the user's images and "Add image…"."""
        self.kind.clear()
        category = None
        for kind in [*KINDS, *load_user_images(user_images_file())]:
            if kind.category != category:
                if category is not None:
                    self.kind.insertSeparator(self.kind.count())
                category = kind.category
            label = kind.name if kind.icon != "none" else f"{kind.name} (no icon)"
            self.kind.addItem(icon_for(kind.icon, kind.caption), f"{label}", kind)
            self.kind.setItemData(self.kind.count() - 1, category, 3)  # (tooltip: the category)
        self.kind.insertSeparator(self.kind.count())
        self.kind.addItem("Add image…", ADD_IMAGE)
        index = next((i for i in range(self.kind.count())
                      if isinstance(self.kind.itemData(i), PoiKind)
                      and self.kind.itemData(i).icon == select), 0)  # fmt: skip
        self.kind.setCurrentIndex(index)

    def _kind_chosen(self, index: int) -> None:
        if self.kind.itemData(index) != ADD_IMAGE:
            return
        start = str(self.base_dir() or Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "POI image", start,
                                              "Images (*.png *.gif *.apng *.webp)")  # fmt: skip
        if path:
            self._fill_kinds(select=add_user_image(user_images_file(), Path(path)).icon)
        else:
            self.kind.setCurrentIndex(0)

    def current_kind(self) -> PoiKind:
        kind = self.kind.currentData()
        return kind if isinstance(kind, PoiKind) else KINDS[0]

    def _connect(self) -> None:
        self.kind.activated.connect(self._kind_chosen)
        self.add_hiker.clicked.connect(self._add_at_hiker)
        self.add_click.clicked.connect(self._add_by_click)
        self.add_place.clicked.connect(self._add_at_place)
        self.move_place.clicked.connect(self._move_to_place)
        self.remove.clicked.connect(self._remove)
        self.list.currentRowChanged.connect(self._show_current)
        self.icon.activated.connect(self._icon_chosen)
        for field, widget, value in (
            ("name", self.name, self.name.text),
            ("caption", self.caption, self.caption.text),
            ("size_px", self.size_px, self.size_px.value),
            ("lift_px", self.lift, self.lift.value),
            ("sprite_cols", self.cols, self.cols.value),
            ("sprite_rows", self.rows, self.rows.value),
            ("fps", self.fps, self.fps.value),
        ):
            widget.editingFinished.connect(lambda f=field, v=value: self._edit_field(f, v()))
        for spin in (self.lat, self.lon):
            spin.editingFinished.connect(
                lambda: self._updating or self._move_to(self.lon.value(), self.lat.value())
            )
        self.move_hiker.clicked.connect(self._move_to_hiker)
        self.move_click.clicked.connect(self._move_by_click)

    # --- helpers ------------------------------------------------------------------------
    def current(self) -> Poi | None:
        row = self.list.currentRow()
        return self.scene.pois[row] if 0 <= row < len(self.scene.pois) else None

    def refresh(self, select: int | None = None) -> None:
        row = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for poi in self.scene.pois:
            item = QListWidgetItem(icon_for(poi.icon, poi.caption, self.base_dir()), poi.name)
            item.setToolTip(f"{poi.caption}\n{poi.lat:.5f}, {poi.lon:.5f}".strip())
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.scene.pois:
            self.list.setCurrentRow(min(max(row, 0), len(self.scene.pois) - 1))
        self._show_current()

    def _show_current(self, *_args) -> None:
        poi = self.current()
        enabled = poi is not None
        for w in (self.name, self.caption, self.icon, self.size_px, self.lift, self.cols, self.rows,
                  self.fps, self.lat, self.lon, self.move_hiker, self.move_click,
                  self.move_place, self.remove):  # fmt: skip
            w.setEnabled(enabled)
        if poi is None:
            return
        self._updating = True
        self.name.setText(poi.name)
        self.caption.setText(poi.caption)
        index = self.icon.findData(poi.icon)
        if index < 0:
            self.icon.insertItem(self.icon.count() - 1, Path(poi.icon).name, poi.icon)
            index = self.icon.findData(poi.icon)
        self.icon.setCurrentIndex(index)
        self.size_px.setValue(poi.size_px)
        self.lift.setValue(poi.lift_px)
        self.cols.setValue(poi.sprite_cols)
        self.rows.setValue(poi.sprite_rows)
        self.fps.setValue(poi.fps)
        self.lat.setValue(poi.lat)
        self.lon.setValue(poi.lon)
        self._updating = False

    def _apply(self, label: str, mutate: Callable[[], None], select: int | None = None) -> None:
        before = poi_snapshot(self.scene)
        mutate()
        after = poi_snapshot(self.scene)
        if before == after:
            return
        self.undo_stack.push(PoiCommand(label, self.scene, before, after, self._after_undo))
        self.refresh(select)
        self.changed.emit()

    def _after_undo(self) -> None:
        self.refresh()
        self.changed.emit()

    # --- actions ------------------------------------------------------------------------
    def add_poi(self, lon: float, lat: float, name: str | None = None,
                kind: PoiKind | None = None) -> Poi:  # fmt: skip
        """A new POI of ``kind`` (default: the one chosen in the dock) at lon/lat."""
        kind = kind or self.current_kind()
        name = name or kind.name
        poi = Poi(slug(name, {p.id for p in self.scene.pois}), name, float(lon), float(lat),
                  icon=self._icon_ref(kind.icon), caption=kind.caption,
                  size_px=kind.size_px)  # fmt: skip
        self._apply("Add POI", lambda: self.scene.set_pois([*self.scene.pois, poi]),
                    select=len(self.scene.pois))  # fmt: skip
        return poi

    def _add_at_hiker(self) -> None:
        position = self.hiker_position()
        if position is not None:
            self.add_poi(*position)

    def _add_by_click(self) -> None:
        self.request_pick(lambda lon, lat: self.add_poi(lon, lat))

    def _add_at_place(self) -> None:
        place = (
            self.choose_place("New POI at…", self.hiker_position()) if self.choose_place else None
        )
        if place is not None:
            self.add_poi(*place)

    def _move_to_place(self) -> None:
        poi = self.current()
        if poi is not None and self.choose_place is not None:
            place = self.choose_place(f"Move {poi.name} to…", (poi.lon, poi.lat))
            if place is not None:
                self._move_to(*place)

    def _remove(self) -> None:
        poi = self.current()
        if poi is not None:
            row = self.list.currentRow()
            self._apply("Remove POI",
                        lambda: self.scene.set_pois([p for p in self.scene.pois if p is not poi]),
                        select=row - 1)  # fmt: skip

    def _edit_field(self, field: str, value) -> None:
        poi = self.current()
        if poi is None or self._updating or getattr(poi, field) == value:
            return

        def mutate() -> None:
            setattr(poi, field, value)
            # renaming changes the section title (the id stays: keys keep working)
            self.scene.set_pois(self.scene.pois)

        self._apply(f"POI {field}", mutate)

    def _icon_ref(self, ref: str) -> str:
        """Image files relative to the scene folder where possible."""
        if ref in ("none", BADGE) or ref.startswith("builtin:"):
            return ref
        base = self.base_dir()
        try:
            return str(Path(ref).relative_to(base)) if base else ref
        except ValueError:
            return ref

    def _icon_chosen(self, index: int) -> None:
        data = self.icon.itemData(index)
        if data == CUSTOM:
            path, _ = QFileDialog.getOpenFileName(
                self, "POI icon", str(self.base_dir() or Path.home()),
                "Images (*.png *.gif *.apng *.webp)",
            )  # fmt: skip
            if not path:
                self._show_current()
                return
            data = self._icon_ref(path)
        self._edit_field("icon", data)

    def _move_to(self, lon: float, lat: float) -> None:
        poi = self.current()
        if poi is None:
            return

        def mutate() -> None:
            poi.lon, poi.lat = float(lon), float(lat)

        self._apply("Move POI", mutate)

    def _move_to_hiker(self) -> None:
        position = self.hiker_position()
        if position is not None:
            self._move_to(*position)

    def _move_by_click(self) -> None:
        self.request_pick(self._move_to)
