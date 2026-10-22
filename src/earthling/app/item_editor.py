"""Base of the editors of placed scene items (POIs, photos): the form of the item selected in
the scene list, its position (coordinates, the marker, a click on the terrain or a place) and
undo. The inspector shows the form above the item's own (animatable) settings.

Structural edits are undoable as snapshots of the items with the values and keys of their
properties.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QUndoCommand
from PyQt6.QtWidgets import QFormLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from earthling.core.scene import Scene
from earthling.ui import theme
from earthling.ui.property_panel import LABEL_WIDTH
from earthling.ui.widgets import double_spin, row


def items_snapshot(scene: Scene, attr: str, prefix: str) -> str:
    """The items ``scene.<attr>`` with the values and keys of their ``<prefix>*`` properties."""
    props = {pid: scene.registry[pid].to_json(v) for pid, v in scene.store.values.items()
             if pid.startswith(prefix)}  # fmt: skip
    curves = {pid: c.to_json() for pid, c in scene.animation.curves.items()
              if pid.startswith(prefix)}  # fmt: skip
    return json.dumps({"items": [i.to_json() for i in getattr(scene, attr)], "props": props,
                       "curves": curves})  # fmt: skip


def restore_items(scene: Scene, attr: str, prefix: str, item_class, snapshot: str) -> None:
    from earthling.core.animation import Curve

    doc = json.loads(snapshot)
    getattr(scene, f"set_{attr}")([item_class.from_json(i) for i in doc["items"]])
    for pid in [p for p in scene.animation.curves if p.startswith(prefix)]:
        del scene.animation.curves[pid]
    for pid, raw in doc["props"].items():
        if pid in scene.registry:
            scene.store.set(pid, scene.registry[pid].from_json(raw))
    for pid, data in doc["curves"].items():
        if pid in scene.registry:
            scene.animation.curves[pid] = Curve.from_json(scene.registry[pid], data)


class ItemsCommand(QUndoCommand):
    def __init__(self, label: str, restore: Callable[[str], None], before: str, after: str,
                 on_change: Callable[[], None]) -> None:  # fmt: skip
        super().__init__(label)
        self.restore, self.before, self.after, self.on_change = restore, before, after, on_change
        self._first = True

    def redo(self) -> None:
        if self._first:  # already applied when pushed
            self._first = False
            return
        self.restore(self.after)
        self.on_change()

    def undo(self) -> None:
        self.restore(self.before)
        self.on_change()


def icon_button(icon: str, text: str, tip: str) -> QPushButton:
    button = QPushButton(theme.icon(icon), text)
    button.setIconSize(QSize(14, 14))
    button.setToolTip(tip)
    return button


class SceneItemEditor(QWidget):
    """Subclasses set ATTR ("pois"), PREFIX ("poi."), ITEM (the dataclass) and NOUN, and
    provide the form rows and how an item looks in the scene list."""

    changed = pyqtSignal()  # items edited (the viewport redraws, the scene list refreshes)
    current_changed = pyqtSignal(int)  # an item was added or removed: select this one
    ATTR = PREFIX = NOUN = ""
    ITEM = None

    def __init__(self, scene: Scene, undo_stack, hiker_position: Callable,
                 request_pick: Callable, base_dir: Callable[[], Path | None], parent=None,
                 choose_place: Callable | None = None) -> None:  # fmt: skip
        """``hiker_position() -> (lon, lat) | None``; ``request_pick(callback(lon, lat))``;
        ``choose_place(title, (lon, lat) | None) -> (lon, lat) | None`` (a place dialog)."""
        super().__init__(parent)
        self.setObjectName("itemEditor")
        self.scene = scene
        self.undo_stack = undo_stack
        self.hiker_position = hiker_position
        self.request_pick = request_pick
        self.base_dir = base_dir
        self.choose_place = choose_place
        self.current_index = -1
        self._updating = False
        self._build_body()
        self._connect()
        self.refresh()

    # --- subclass parts -------------------------------------------------------------------
    def _form_rows(self, form: QFormLayout) -> list[QWidget]:
        """Add the item's fields; returns the widgets enabled only with a current item."""
        return []

    def _show_item(self, item) -> None:
        """Fill the form fields from ``item``."""

    def item_icon(self, item) -> QIcon:
        return QIcon()

    def item_tooltip(self, item) -> str:
        return f"{item.lat:.5f}, {item.lon:.5f}" if item.lat is not None else ""

    def _connect_fields(self) -> None:
        """Connect the form fields (``_edit_field`` on edits)."""

    # --- layout ---------------------------------------------------------------------------
    def _build_body(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 12, 12)
        layout.setSpacing(8)
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        self._fields = self._form_rows(form)
        self.lat = double_spin(-90.0, 90.0, decimals=6, suffix="°")
        self.lon = double_spin(-180.0, 180.0, decimals=6, suffix="°")
        self.lat.setToolTip("Latitude (north positive)")
        self.lon.setToolTip("Longitude (east positive)")
        self.move_hiker = icon_button("bicycle", "At marker", "Move it to the head of the line")
        self.move_click = icon_button("crosshair", "Pick", "Click on the terrain to move it")
        self.move_place = icon_button("magnifying-glass", "Place…",
                                      "Coordinates or a point of a track")  # fmt: skip
        form.addRow("Position", row((self.lat, 1), (self.lon, 1)))
        form.addRow("", row(self.move_hiker, self.move_click, self.move_place))
        for i in range(form.rowCount()):
            label = form.itemAt(i, QFormLayout.ItemRole.LabelRole)
            if label is not None and label.widget() is not None:
                label.widget().setObjectName("rowLabel")
                label.widget().setFixedWidth(LABEL_WIDTH - 10)
            field = form.itemAt(i, QFormLayout.ItemRole.FieldRole)
            if field is not None and field.widget() is not None and field.widget().layout():
                field.widget().layout().setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form)
        self.remove = icon_button("trash", f"Remove {self.NOUN.lower()}",
                                  f"Remove this {self.NOUN.lower()} (undoable)")  # fmt: skip
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.remove)
        layout.addLayout(buttons)

    def _connect(self) -> None:
        self.remove.clicked.connect(self._remove)
        for spin in (self.lat, self.lon):
            spin.editingFinished.connect(
                lambda: self._updating or self._move_to(self.lon.value(), self.lat.value())
            )
        self.move_hiker.clicked.connect(self._move_to_hiker)
        self.move_click.clicked.connect(lambda: self.request_pick(self._move_to))
        self.move_place.clicked.connect(self._move_to_place)
        self._connect_fields()

    # --- items ----------------------------------------------------------------------------
    def items(self) -> list:
        return getattr(self.scene, self.ATTR)

    def set_items(self, items: list) -> None:
        getattr(self.scene, f"set_{self.ATTR}")(items)

    def current(self):
        items = self.items()
        return items[self.current_index] if 0 <= self.current_index < len(items) else None

    def select(self, index: int) -> None:
        """Show the item ``index`` in the form."""
        self.current_index = index
        self._show_current()

    def refresh(self, select: int | None = None) -> None:
        """Show the items again (after loading or undo); ``select``: and select that one."""
        index = self.current_index if select is None else select
        count = len(self.items())
        self.current_index = min(max(index, 0), count - 1) if count else -1
        self._show_current()
        if select is not None:
            self.current_changed.emit(self.current_index)

    def _show_current(self) -> None:
        item = self.current()
        for w in [*self._fields, self.lat, self.lon, self.move_hiker, self.move_click,
                  self.move_place, self.remove]:  # fmt: skip
            w.setEnabled(item is not None)
        if item is None:
            return
        self._updating = True
        self._show_item(item)
        if item.lat is not None:
            self.lat.setValue(item.lat)
            self.lon.setValue(item.lon)
        self._updating = False

    def _apply(self, label: str, mutate: Callable[[], None], select: int | None = None) -> None:
        before = items_snapshot(self.scene, self.ATTR, self.PREFIX)
        mutate()
        after = items_snapshot(self.scene, self.ATTR, self.PREFIX)
        if before == after:
            return

        def restore(snapshot: str) -> None:
            restore_items(self.scene, self.ATTR, self.PREFIX, self.ITEM, snapshot)

        self.undo_stack.push(ItemsCommand(label, restore, before, after, self._after_undo))
        self.changed.emit()
        self.refresh(select)

    def _after_undo(self) -> None:
        self.changed.emit()
        self.refresh(self.current_index)

    # --- edits ----------------------------------------------------------------------------
    def add_item(self, item, label: str | None = None) -> None:
        self._apply(label or f"Add {self.NOUN}", lambda: self.set_items([*self.items(), item]),
                    select=len(self.items()))  # fmt: skip

    def remove_index(self, index: int) -> None:
        items = self.items()
        if 0 <= index < len(items):
            item = items[index]
            self._apply(f"Remove {self.NOUN}",
                        lambda: self.set_items([i for i in self.items() if i is not item]),
                        select=max(index - 1, 0))  # fmt: skip

    def _remove(self) -> None:
        self.remove_index(self.current_index)

    def _edit_field(self, field: str, value) -> None:
        item = self.current()
        if item is None or self._updating or getattr(item, field) == value:
            return

        def mutate() -> None:
            setattr(item, field, value)
            # renaming changes the section title (the id stays: keys keep working)
            self.set_items(self.items())

        self._apply(f"{self.NOUN} {field}", mutate)

    def _move_to(self, lon: float, lat: float) -> None:
        item = self.current()
        if item is None:
            return

        def mutate() -> None:
            item.lon, item.lat = float(lon), float(lat)

        self._apply(f"Move {self.NOUN}", mutate)

    def _move_to_hiker(self) -> None:
        position = self.hiker_position()
        if position is not None:
            self._move_to(*position)

    def _move_to_place(self) -> None:
        item = self.current()
        if item is not None and self.choose_place is not None:
            start = (item.lon, item.lat) if item.lat is not None else self.hiker_position()
            place = self.choose_place(f"Move {item.name} to…", start)
            if place is not None:
                self._move_to(*place)

    def file_ref(self, path: str) -> str:
        """Files relative to the scene folder where possible."""
        base = self.base_dir()
        try:
            return str(Path(path).relative_to(base)) if base else path
        except ValueError:
            return path
