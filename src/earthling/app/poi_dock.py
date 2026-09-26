"""Points of interest dock: add (at the hiker or by clicking the terrain), edit and remove POIs.

Structural edits are undoable (snapshots of the POIs with their property values and keys).
The animatable part of every POI appears as its own section in the parameter panel.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QUndoCommand
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
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

from earthling.core.pois import Poi, slug
from earthling.core.scene import Scene
from earthling.render.pois import builtin_icons, resolve_icon

CUSTOM = "__custom__"


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
                 base_dir: Callable[[], Path | None], parent=None) -> None:  # fmt: skip
        """``hiker_position() -> (lon, lat) | None``; ``request_pick(callback(lon, lat))``."""
        super().__init__("Points of Interest", parent)
        self.setObjectName("poi_dock")
        self.scene = scene
        self.undo_stack = undo_stack
        self.hiker_position = hiker_position
        self.request_pick = request_pick
        self.base_dir = base_dir
        body = QWidget()
        layout = QVBoxLayout(body)
        buttons = QHBoxLayout()
        self.add_hiker = QPushButton("+ At hiker")
        self.add_hiker.setToolTip("New POI at the head of the drawn track")
        self.add_click = QPushButton("+ On map…")
        self.add_click.setToolTip("New POI where you click on the terrain")
        self.remove = QPushButton("Remove")
        for b in (self.add_hiker, self.add_click, self.remove):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.list = QListWidget()
        self.list.setIconSize(QSize(28, 28))
        layout.addWidget(self.list, 1)
        form = QFormLayout()
        self.name = QLineEdit()
        self.caption = QLineEdit()
        self.icon = QComboBox()
        for name in builtin_icons():
            path = resolve_icon(f"builtin:{name}", None)
            self.icon.addItem(QIcon(QPixmap(str(path))), name, f"builtin:{name}")
        self.icon.addItem("Custom file…", CUSTOM)
        self.size = QDoubleSpinBox()
        self.size.setRange(8.0, 512.0)
        self.size.setSuffix(" px")
        self.lift = QDoubleSpinBox()
        self.lift.setRange(0.0, 400.0)
        self.lift.setSuffix(" px")
        self.cols = QSpinBox()
        self.rows = QSpinBox()
        for spin in (self.cols, self.rows):
            spin.setRange(1, 64)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.5, 60.0)
        sprite = QHBoxLayout()
        for w in (QLabel("cols"), self.cols, QLabel("rows"), self.rows, QLabel("fps"), self.fps):
            sprite.addWidget(w)
        sprite_widget = QWidget()
        sprite_widget.setLayout(sprite)
        self.position = QLabel("–")
        move = QHBoxLayout()
        self.move_hiker = QPushButton("To hiker")
        self.move_click = QPushButton("Pick…")
        move.addWidget(self.position, 1)
        move.addWidget(self.move_hiker)
        move.addWidget(self.move_click)
        move_widget = QWidget()
        move_widget.setLayout(move)
        form.addRow("Name", self.name)
        form.addRow("Caption", self.caption)
        form.addRow("Icon", self.icon)
        form.addRow("Size (1080p)", self.size)
        form.addRow("Pin height", self.lift)
        form.addRow("Sprite sheet", sprite_widget)
        form.addRow("Position", move_widget)
        layout.addLayout(form)
        layout.addWidget(QLabel("Animate visibility, effects … in the POI's section of the "
                                "parameter panel."))  # fmt: skip
        self.setWidget(body)
        self._updating = False
        self.add_hiker.clicked.connect(self._add_at_hiker)
        self.add_click.clicked.connect(self._add_by_click)
        self.remove.clicked.connect(self._remove)
        self.list.currentRowChanged.connect(self._show_current)
        self.name.editingFinished.connect(lambda: self._edit_field("name", self.name.text()))
        self.caption.editingFinished.connect(
            lambda: self._edit_field("caption", self.caption.text())
        )
        self.icon.activated.connect(self._icon_chosen)
        self.size.editingFinished.connect(lambda: self._edit_field("size_px", self.size.value()))
        self.lift.editingFinished.connect(lambda: self._edit_field("lift_px", self.lift.value()))
        self.cols.editingFinished.connect(
            lambda: self._edit_field("sprite_cols", self.cols.value())
        )
        self.rows.editingFinished.connect(
            lambda: self._edit_field("sprite_rows", self.rows.value())
        )
        self.fps.editingFinished.connect(lambda: self._edit_field("fps", self.fps.value()))
        self.move_hiker.clicked.connect(self._move_to_hiker)
        self.move_click.clicked.connect(self._move_by_click)
        self.refresh()

    # --- helpers ------------------------------------------------------------------------
    def current(self) -> Poi | None:
        row = self.list.currentRow()
        return self.scene.pois[row] if 0 <= row < len(self.scene.pois) else None

    def refresh(self, select: int | None = None) -> None:
        row = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for poi in self.scene.pois:
            path = resolve_icon(poi.icon, self.base_dir()) or resolve_icon("builtin:pin", None)
            item = QListWidgetItem(QIcon(QPixmap(str(path))), poi.name)
            item.setToolTip(f"{poi.caption}\n{poi.lat:.5f}, {poi.lon:.5f}".strip())
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.scene.pois:
            self.list.setCurrentRow(min(max(row, 0), len(self.scene.pois) - 1))
        self._show_current()

    def _show_current(self, *_args) -> None:
        poi = self.current()
        enabled = poi is not None
        for w in (self.name, self.caption, self.icon, self.size, self.lift, self.cols, self.rows,
                  self.fps, self.move_hiker, self.move_click, self.remove):  # fmt: skip
            w.setEnabled(enabled)
        if poi is None:
            self.position.setText("–")
            return
        self._updating = True
        self.name.setText(poi.name)
        self.caption.setText(poi.caption)
        index = self.icon.findData(poi.icon)
        if index < 0:
            self.icon.insertItem(self.icon.count() - 1, Path(poi.icon).name, poi.icon)
            index = self.icon.findData(poi.icon)
        self.icon.setCurrentIndex(index)
        self.size.setValue(poi.size_px)
        self.lift.setValue(poi.lift_px)
        self.cols.setValue(poi.sprite_cols)
        self.rows.setValue(poi.sprite_rows)
        self.fps.setValue(poi.fps)
        self.position.setText(f"{poi.lat:.5f}, {poi.lon:.5f}")
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
    def add_poi(self, lon: float, lat: float, name: str = "POI") -> Poi:
        poi = Poi(slug(name, {p.id for p in self.scene.pois}), name, float(lon), float(lat))
        self._apply("Add POI", lambda: self.scene.set_pois([*self.scene.pois, poi]),
                    select=len(self.scene.pois))  # fmt: skip
        return poi

    def _add_at_hiker(self) -> None:
        position = self.hiker_position()
        if position is not None:
            self.add_poi(*position)

    def _add_by_click(self) -> None:
        self.request_pick(lambda lon, lat: self.add_poi(lon, lat))

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
            base = self.base_dir()
            try:
                data = str(Path(path).relative_to(base)) if base else path
            except ValueError:
                data = path
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

    def select_poi(self, poi_id: str) -> None:
        for i, poi in enumerate(self.scene.pois):
            if poi.id == poi_id:
                self.list.setCurrentRow(i)
