"""Auto-generated editor panels for declared properties."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QDateTime, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.core.properties import PropertyDef, PropertyStore, PType

SLIDER_STEPS = 1000

# on_edit(property_id, value, interactive): ``interactive`` is True while a slider is dragged
EditCallback = Callable[[str, Any, bool], None]


class PropertyEditor(QWidget):
    """Base class: shows a value and emits ``edited(value, interactive)``."""

    edited = pyqtSignal(object, bool)

    def __init__(self, definition: PropertyDef, parent=None) -> None:
        super().__init__(parent)
        self.definition = definition
        self._updating = False

    def set_value(self, value: Any) -> None:
        self._updating = True
        try:
            self._show(value)
        finally:
            self._updating = False

    def _show(self, value: Any) -> None:
        raise NotImplementedError

    def _emit(self, value: Any, interactive: bool = False) -> None:
        if not self._updating:
            self.edited.emit(value, interactive)


class FloatEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(d.decimals)
        self.spin.setRange(
            d.minimum if d.minimum is not None else -1e9,
            d.maximum if d.maximum is not None else 1e9,
        )
        self.spin.setSingleStep(d.step or 0.1)
        if d.unit:
            self.spin.setSuffix(f" {d.unit}")
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(90)
        self.slider: QSlider | None = None
        if d.minimum is not None and d.maximum is not None:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, SLIDER_STEPS)
            self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(self.slider, 1)
            self.slider.valueChanged.connect(self._slider_moved)
            self.slider.sliderReleased.connect(self._slider_released)
        layout.addWidget(self.spin)
        self.spin.valueChanged.connect(lambda v: self._emit(float(v)))

    def _to_slider(self, v: float) -> int:
        d = self.definition
        lo, hi = float(d.minimum), float(d.maximum)
        if d.logarithmic and lo > 0:
            t = (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            t = (v - lo) / (hi - lo) if hi > lo else 0.0
        return int(round(min(1.0, max(0.0, t)) * SLIDER_STEPS))

    def _from_slider(self, pos: int) -> float:
        d = self.definition
        lo, hi = float(d.minimum), float(d.maximum)
        t = pos / SLIDER_STEPS
        if d.logarithmic and lo > 0:
            return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
        return lo + t * (hi - lo)

    def _slider_moved(self, pos: int) -> None:
        if self._updating:
            return
        value = round(self._from_slider(pos), self.definition.decimals)
        self._updating = True
        self.spin.setValue(value)
        self._updating = False
        self._emit(value, interactive=bool(self.slider and self.slider.isSliderDown()))

    def _slider_released(self) -> None:
        self._emit(float(self.spin.value()), interactive=False)

    def _show(self, value: Any) -> None:
        self.spin.setValue(float(value))
        if self.slider is not None:
            self.slider.setValue(self._to_slider(float(value)))


class IntEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spin = QSpinBox()
        self.spin.setRange(
            int(d.minimum) if d.minimum is not None else -(2**31),
            int(d.maximum) if d.maximum is not None else 2**31 - 1,
        )
        self.spin.setSingleStep(int(d.step or 1))
        if d.unit:
            self.spin.setSuffix(f" {d.unit}")
        self.spin.setKeyboardTracking(False)
        layout.addWidget(self.spin)
        self.spin.valueChanged.connect(lambda v: self._emit(int(v)))

    def _show(self, value: Any) -> None:
        self.spin.setValue(int(value))


class BoolEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.box = QCheckBox()
        layout.addWidget(self.box)
        layout.addStretch()
        self.box.toggled.connect(lambda v: self._emit(bool(v)))

    def _show(self, value: Any) -> None:
        self.box.setChecked(bool(value))


class EnumEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for value, label in d.options:
            self.combo.addItem(label, value)
        layout.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(lambda i: self._emit(self.combo.itemData(i)))

    def _show(self, value: Any) -> None:
        index = self.combo.findData(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)


class ColorEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton()
        self.button.setFixedHeight(22)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._pick)
        self._value = (1.0, 1.0, 1.0)

    def _pick(self) -> None:
        initial = QColor.fromRgbF(*self._value)
        color = QColorDialog.getColor(initial, self, self.definition.label)
        if color.isValid():
            self._emit((color.redF(), color.greenF(), color.blueF()))

    def _show(self, value: Any) -> None:
        self._value = tuple(value)
        c = QColor.fromRgbF(*self._value)
        self.button.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")
        self.button.setToolTip(c.name())


class Vec3Editor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spins = []
        for _ in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(d.decimals)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._changed)
            layout.addWidget(spin)
            self.spins.append(spin)

    def _changed(self) -> None:
        self._emit(tuple(s.value() for s in self.spins))

    def _show(self, value: Any) -> None:
        for spin, v in zip(self.spins, value, strict=True):
            spin.setValue(float(v))


class DateTimeEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QDateTimeEdit()
        self.edit.setDisplayFormat("yyyy-MM-dd  HH:mm")
        self.edit.setCalendarPopup(True)
        self.edit.setKeyboardTracking(False)
        layout.addWidget(self.edit)
        self.edit.dateTimeChanged.connect(lambda dt: self._emit(dt.toPyDateTime()))

    def _show(self, value: Any) -> None:
        self.edit.setDateTime(QDateTime(value))


EDITORS = {
    PType.FLOAT: FloatEditor,
    PType.INT: IntEditor,
    PType.BOOL: BoolEditor,
    PType.ENUM: EnumEditor,
    PType.COLOR: ColorEditor,
    PType.VEC3: Vec3Editor,
    PType.DATETIME: DateTimeEditor,
}


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        self.header = QToolButton()
        self.header.setText(title.replace("&", "&&"))  # no mnemonics
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.DownArrow)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setStyleSheet("QToolButton { font-weight: bold; border: none; }")
        self.header.toggled.connect(self._toggle)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(12, 0, 4, 0)
        self.grid.setColumnStretch(1, 1)
        layout.addWidget(self.header)
        layout.addWidget(self.body)

    def _toggle(self, open_: bool) -> None:
        self.body.setVisible(open_)
        self.header.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)


class PropertyPanel(QScrollArea):
    """One collapsible section per property section; editors stay in sync with the store."""

    def __init__(
        self,
        store: PropertyStore,
        on_edit: EditCallback | None = None,
        row_extra: Callable[[PropertyDef], QWidget | None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.store = store
        self.on_edit = on_edit or (lambda pid, value, interactive: store.set(pid, value))
        self.row_extra = row_extra
        self.editors: dict[str, PropertyEditor] = {}
        self.sections: dict[str, CollapsibleSection] = {}
        self._unsubscribe = None
        self._build()

    def set_store(self, store: PropertyStore) -> None:
        self.store = store
        self._build()

    def _build(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        self.editors.clear()
        self.sections.clear()
        for title, defs in self.store.registry.sections():
            defs = [d for d in defs if d.type in EDITORS]  # e.g. camera poses have no editor
            if not defs:
                continue
            section = CollapsibleSection(title)
            for row, d in enumerate(defs):
                label = QLabel(d.label)
                label.setToolTip(d.tooltip or d.id)
                editor = EDITORS[d.type](d)
                editor.setToolTip(d.tooltip or d.id)
                editor.set_value(self.store[d.id])
                editor.edited.connect(
                    lambda value, interactive, pid=d.id: self.on_edit(pid, value, interactive)
                )
                section.grid.addWidget(label, row, 0)
                section.grid.addWidget(editor, row, 1)
                if self.row_extra is not None:
                    extra = self.row_extra(d)
                    if extra is not None:
                        section.grid.addWidget(extra, row, 2)
                self.editors[d.id] = editor
            layout.addWidget(section)
            self.sections[title] = section
        layout.addStretch()
        self.setWidget(container)
        self._unsubscribe = self.store.subscribe(self._on_store_changed)

    def _on_store_changed(self, pid: str, value: Any) -> None:
        editor = self.editors.get(pid)
        if editor is not None:
            editor.set_value(value)
