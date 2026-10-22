"""Editors for declared properties and the collapsible sections that hold them.

Every row shows a dot when the value differs from its default, the label, the editor and
(optionally) extra widgets such as the key button. Sections hide rows by the declarations:
``when`` conditions, ``advanced`` rows behind "More settings", a ``switch`` in the header, and
one tab per prefix where labels read "<tab>: <setting>" (the three cloud layers).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from typing import Any

from PyQt6.QtCore import QDateTime, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QDateTimeEdit,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.core.properties import PropertyDef, PropertyStore, PType
from earthling.ui import theme
from earthling.ui.widgets import double_spin, tight_hbox

LABEL_WIDTH = 132
CORNERS = {"top_left", "top_right", "bottom_left", "bottom_right"}

# on_edit(property_id, value, interactive): ``interactive`` is True while a value is dragged
EditCallback = Callable[[str, Any, bool], None]
KeyState = Callable[[str], str]  # "static" | "animated" | "keyed"


def _parse_number(text: str) -> float:
    return float(text.strip().replace(",", ".").replace("−", "-").split(" ")[0])


class DragField(QWidget):
    """A number shown inside a filled bar: drag sideways to change it (Shift: fine, Ctrl:
    coarse), click to type, arrow keys step. ``edited(value, interactive)``."""

    edited = pyqtSignal(float, bool)

    def __init__(self, minimum: float | None, maximum: float | None, step: float = 0.1,
                 decimals: int = 2, unit: str = "", logarithmic: bool = False,
                 integer: bool = False, parent=None) -> None:  # fmt: skip
        super().__init__(parent)
        self.minimum, self.maximum = minimum, maximum
        self.step = step or (1.0 if integer else 0.1)
        self.decimals = 0 if integer else decimals
        self.unit = unit
        self.logarithmic = logarithmic and minimum is not None and minimum > 0
        self.integer = integer
        self.value = 0.0
        self.state = "static"  # animated / keyed tint the bar
        self._press: tuple[float, float] | None = None  # (x, value)
        self._dragging = False
        self._edit: QLineEdit | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80)
        self.setFixedHeight(24)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(140, 24)

    # --- value mapping ----------------------------------------------------------------
    @property
    def bounded(self) -> bool:
        """A bar makes sense: a finite range of a sensible size."""
        lo, hi = self.minimum, self.maximum
        return lo is not None and hi is not None and hi > lo and (hi - lo) / self.step <= 1e6

    def fraction(self, v: float | None = None) -> float:
        v = self.value if v is None else v
        if not self.bounded:
            return 0.0
        lo, hi = float(self.minimum), float(self.maximum)
        if self.logarithmic:
            t = (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            t = (v - lo) / (hi - lo)
        return min(1.0, max(0.0, t))

    def from_fraction(self, t: float) -> float:
        lo, hi = float(self.minimum), float(self.maximum)
        t = min(1.0, max(0.0, t))
        if self.logarithmic:
            return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
        return lo + t * (hi - lo)

    def clamp(self, v: float) -> float:
        if self.minimum is not None:
            v = max(float(self.minimum), v)
        if self.maximum is not None:
            v = min(float(self.maximum), v)
        return float(round(v)) if self.integer else round(v, self.decimals)

    def text(self) -> str:
        decimals = 0 if self.integer else self.decimals
        number = f"{self.value:.{decimals}f}"
        return f"{number} {self.unit}".strip()

    def set_value(self, v: float) -> None:
        self.value = float(v)
        self.update()

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.update()

    def commit(self, v: float, interactive: bool = False) -> None:
        """Set and report a value (as if the user had typed it)."""
        self.value = self.clamp(v)
        self.update()
        self.edited.emit(self.value, interactive)

    # --- painting ---------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        fill = {"animated": theme.KEY_FIELD, "keyed": theme.KEY_FIELD_ON}.get(
            self.state, theme.VALUE_FILL)  # fmt: skip
        border = {"animated": "#5D4B1F", "keyed": "#A17A26"}.get(self.state, theme.LINE_STRONG)
        if self.hasFocus():
            border = theme.ACCENT
        elif self.underMouse():
            border = "#4A5463"
        p.setPen(QPen(QColor(border), 1))
        p.setBrush(QColor(theme.FIELD))
        p.drawRoundedRect(r, 6, 6)
        if self.bounded and self.isEnabled():
            width = (r.width() - 2) * self.fraction()
            if width > 0.5:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(fill))
                clip = QRectF(r.left() + 1, r.top() + 1, width, r.height() - 2)
                p.save()
                p.setClipRect(clip)
                p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 5, 5)
                p.restore()
        p.setFont(theme.ui_font(12, tabular=True))
        p.setPen(QColor(theme.TEXT if self.isEnabled() else theme.TEXT_3))
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter
                   | Qt.AlignmentFlag.AlignRight, self.text())  # fmt: skip
        p.end()

    # --- interaction ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._edit is None:
            self._press = (event.position().x(), self.value)
            self._dragging = False
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press is None:
            return
        x0, v0 = self._press
        dx = event.position().x() - x0
        if not self._dragging and abs(dx) < 3:
            return
        self._dragging = True
        mods = event.modifiers()
        gain = 0.1 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
        if mods & Qt.KeyboardModifier.ControlModifier:
            gain = 5.0
        if self.bounded:
            span = max(160.0, float(self.width()))
            value = self.from_fraction(self.fraction(v0) + dx / span * gain)
        else:
            value = v0 + dx * self.step * gain
        self.commit(value, interactive=True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is None:
            return
        self._press = None
        if self._dragging:
            self._dragging = False
            self.edited.emit(self.value, False)
        else:
            self.start_typing()

    def start_typing(self) -> None:
        if self._edit is not None:
            return
        edit = self._edit = QLineEdit(self)
        number = self.text().removesuffix(self.unit).strip()
        edit.setText(number)
        edit.setGeometry(self.rect())
        edit.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        edit.selectAll()
        edit.show()
        edit.setFocus()
        self._typed_text = number
        edit.textChanged.connect(self._remember_typed)
        edit.editingFinished.connect(self._typed)

    def _remember_typed(self, text: str) -> None:
        self._typed_text = text  # (read here, not from a field that may be destroyed)

    def _typed(self) -> None:
        edit, self._edit = self._edit, None
        if edit is None:
            return
        try:
            value = _parse_number(self._typed_text)
        except ValueError:
            value = None
        edit.deleteLater()
        if value is not None:
            self.commit(value)
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Right, Qt.Key.Key_Left):
            sign = 1 if key in (Qt.Key.Key_Up, Qt.Key.Key_Right) else -1
            if self.logarithmic:
                self.commit(self.value * (1.05**sign))
            else:
                self.commit(self.value + sign * self.step)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2):
            self.start_typing()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        event.ignore()  # the panel scrolls


class SwitchButton(QAbstractButton):
    """A small on/off switch."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(32, 18)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(32, 18)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        track = theme.ACCENT if on else "#353C48"
        if not self.isEnabled():
            track = theme.LINE_STRONG
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(track))
        p.drawRoundedRect(QRectF(0, 0, 32, 18), 9, 9)
        p.setBrush(QColor("#FFFFFF" if on else "#AEB6C2"))
        p.drawEllipse(QPointF(23 if on else 9, 9), 7, 7)
        p.end()


# --- editors ----------------------------------------------------------------------------------
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

    def set_key_state(self, state: str) -> None:
        """Animated values tint the editor (number fields)."""

    def _show(self, value: Any) -> None:
        raise NotImplementedError

    def _emit(self, value: Any, interactive: bool = False) -> None:
        if not self._updating:
            self.edited.emit(value, interactive)


class FloatEditor(PropertyEditor):
    integer = False

    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        self.field = DragField(d.minimum, d.maximum, d.step or (1 if self.integer else 0.1),
                               d.decimals, d.unit, d.logarithmic, self.integer)  # fmt: skip
        layout.addWidget(self.field)
        self.field.edited.connect(self._field_edited)

    def _field_edited(self, value: float, interactive: bool) -> None:
        self._emit(int(round(value)) if self.integer else float(value), interactive)

    def commit(self, value: float) -> None:
        self.field.commit(value)

    def _show(self, value: Any) -> None:
        self.field.set_value(float(value))

    def set_key_state(self, state: str) -> None:
        self.field.set_state(state)


class IntEditor(FloatEditor):
    integer = True


class BoolEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        self.box = SwitchButton()
        layout.addWidget(self.box)
        layout.addStretch()
        self.box.toggled.connect(self._toggled)

    def _toggled(self, on: bool) -> None:
        self._emit(bool(on))

    def _show(self, value: Any) -> None:
        self.box.setChecked(bool(value))


class EnumEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        self.combo = QComboBox()
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for value, label in d.options:
            self.combo.addItem(label, value)
        layout.addWidget(self.combo)
        self.values = [v for v, _ in d.options]
        # (activated: chosen by the user; a combo box being destroyed changes its index)
        self.combo.activated.connect(self._activated)

    def _activated(self, index: int) -> None:
        self._emit(self.values[index])

    def _show(self, value: Any) -> None:
        index = self.combo.findData(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)


class Segments(QFrame):
    """Exclusive buttons side by side; ``chosen(index)``."""

    chosen = pyqtSignal(int)

    def __init__(self, labels: Iterable[str], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("segments")
        layout = tight_hbox(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: list[QToolButton] = []
        for i, label in enumerate(labels):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(label)
            b.setCheckable(True)
            b.setProperty("segment", True)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.group.addButton(b, i)
            self.buttons.append(b)
            layout.addWidget(b)
        self.group.idClicked.connect(self.chosen.emit)

    def set_index(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)

    def index(self) -> int:
        return self.group.checkedId()


class SegmentedEditor(PropertyEditor):
    """Few short choices as segments."""

    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        self.values = [v for v, _ in d.options]
        self.segments = Segments([label for _, label in d.options])
        layout.addWidget(self.segments)
        self.segments.chosen.connect(self._chosen)

    def _chosen(self, index: int) -> None:
        self._emit(self.values[index])

    def _show(self, value: Any) -> None:
        if value in self.values:
            self.segments.set_index(self.values.index(value))


class CornerEditor(PropertyEditor):
    """A screen corner: four small buttons in a frame and the corner's name."""

    ORDER = ["top_left", "top_right", "bottom_left", "bottom_right"]

    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        frame = QFrame()
        frame.setObjectName("corners")
        frame.setFixedSize(42, 28)
        grid = QHBoxLayout(frame)
        grid.setContentsMargins(3, 3, 3, 3)
        self.labels = dict(d.options)
        self.group = QButtonGroup(self)
        cells = QWidget()
        cell_grid = QGridLayout(cells)
        cell_grid.setContentsMargins(0, 0, 0, 0)
        cell_grid.setSpacing(3)
        for i, value in enumerate(self.ORDER):
            b = QToolButton()
            b.setCheckable(True)
            b.setProperty("corner", True)
            b.setFixedSize(15, 9)
            b.setToolTip(self.labels.get(value, value))
            self.group.addButton(b, i)
            cell_grid.addWidget(b, i // 2, i % 2)
        grid.addWidget(cells)
        self.name = QLabel()
        self.name.setObjectName("dim")
        layout.addWidget(frame)
        layout.addWidget(self.name, 1)
        self.group.idClicked.connect(self._corner_clicked)

    def _corner_clicked(self, index: int) -> None:
        self._emit(self.ORDER[index])

    def _show(self, value: Any) -> None:
        if value in self.ORDER:
            self.group.button(self.ORDER.index(value)).setChecked(True)
            self.name.setText(self.labels.get(value, value))


class ColorEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        self.button = QToolButton()
        self.button.setObjectName("swatch")
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setIconSize(QSize(18, 18))
        self.button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        pixmap = QPixmap(36, 36)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 50), 2))
        p.setBrush(c)
        p.drawRoundedRect(QRectF(1, 1, 34, 34), 8, 8)
        p.end()
        self.button.setIcon(QIcon(pixmap))
        self.button.setText(c.name().upper())


class Vec3Editor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        layout.setSpacing(4)
        self.spins = []
        for _ in range(3):
            spin = double_spin(-1e9, 1e9, decimals=d.decimals, tracking=False)
            spin.setButtonSymbols(spin.ButtonSymbols.NoButtons)
            spin.setMinimumWidth(40)
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
        layout = tight_hbox(self)
        self.edit = QDateTimeEdit()
        self.edit.setDisplayFormat("yyyy-MM-dd  HH:mm")
        self.edit.setCalendarPopup(True)
        self.edit.setKeyboardTracking(False)
        layout.addWidget(self.edit)
        self.edit.dateTimeChanged.connect(self._date_changed)

    def _date_changed(self, value) -> None:
        self._emit(value.toPyDateTime())

    def _show(self, value: Any) -> None:
        self.edit.setDateTime(QDateTime(value))


class TextEditor(PropertyEditor):
    def __init__(self, d: PropertyDef, parent=None) -> None:
        super().__init__(d, parent)
        layout = tight_hbox(self)
        self.edit = QLineEdit()
        self._text = ""  # (kept here: a field being destroyed must not be read)
        layout.addWidget(self.edit)
        self.edit.textChanged.connect(self._changed)
        self.edit.editingFinished.connect(self._finished)

    def _finished(self) -> None:
        self._emit(self._text)

    def _changed(self, text: str) -> None:
        self._text = text

    def _show(self, value: Any) -> None:
        if self._text != value:
            self.edit.setText(str(value))


EDITORS = {
    PType.FLOAT: FloatEditor,
    PType.INT: IntEditor,
    PType.BOOL: BoolEditor,
    PType.ENUM: EnumEditor,
    PType.COLOR: ColorEditor,
    PType.VEC3: Vec3Editor,
    PType.DATETIME: DateTimeEditor,
    PType.TEXT: TextEditor,
}


def format_value(d: PropertyDef, value: Any) -> str:
    """A value as a short text (the palette shows it next to the setting)."""
    t = d.type
    if t in (PType.FLOAT, PType.INT):
        decimals = 0 if t is PType.INT else d.decimals
        return f"{float(value):.{decimals}f} {d.unit}".strip()
    if t is PType.BOOL:
        return "On" if value else "Off"
    if t is PType.ENUM:
        return dict(d.options).get(value, str(value))
    if t is PType.COLOR:
        return QColor.fromRgbF(*value).name().upper()
    if t is PType.DATETIME:
        return value.strftime("%Y-%m-%d %H:%M")
    if t is PType.TEXT:
        return f"“{value}”" if value else "–"
    return ""


def retire(view: QWidget | None) -> None:
    """Delete a replaced view safely: a focused field inside would report "editing finished"
    while it is destroyed (a crash), so its focus goes first and its signals stop."""
    if view is None:
        return
    focus = QApplication.focusWidget()
    if focus is not None and view.isAncestorOf(focus):
        focus.clearFocus()
    for child in view.findChildren(QWidget):
        child.blockSignals(True)
    view.hide()
    view.deleteLater()


def segmentable(d: PropertyDef) -> bool:
    """Few, short choices: shown as segments instead of a list."""
    labels = [label for _, label in d.options]
    return 2 <= len(labels) <= 3 and max(map(len, labels)) <= 11 and sum(map(len, labels)) <= 24


def editor_for(d: PropertyDef) -> PropertyEditor:
    if d.type is PType.ENUM:
        if {v for v, _ in d.options} == CORNERS:
            return CornerEditor(d)
        if segmentable(d):
            return SegmentedEditor(d)
    return EDITORS[d.type](d)


def has_editor(d: PropertyDef) -> bool:
    return d.type in EDITORS  # (camera poses have none)


# --- rows and sections ------------------------------------------------------------------------
def tab_of(d: PropertyDef, tabs: list[str]) -> str | None:
    """The tab a property belongs to (its label starts with "<tab>: " or is the tab)."""
    for tab in tabs:
        if d.label == tab or d.label.startswith(tab + ": "):
            return tab
    return None


def find_tabs(defs: list[PropertyDef]) -> list[str]:
    """Label prefixes shared by several properties, if a section has two or more of them."""
    counts: dict[str, int] = {}
    for d in defs:
        if ": " in d.label:
            prefix = d.label.split(": ", 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1
    labels = {d.label for d in defs}
    tabs = [t for t, n in counts.items() if n >= 2 or t in labels]
    return tabs if len(tabs) >= 2 else []


def row_label(d: PropertyDef, tab: str | None) -> str:
    if tab is None:
        return d.label
    if d.label == tab:
        return "On"
    rest = d.label[len(tab) + 2 :]
    return rest[:1].upper() + rest[1:]


class PropertyRow(QWidget):
    """Changed dot, label, editor and extra widgets (key button, pin) of one property."""

    reset_requested = pyqtSignal(str)

    def __init__(self, d: PropertyDef, editor: PropertyEditor, label: str,
                 extras: Iterable[QWidget] = (), parent=None) -> None:  # fmt: skip
        super().__init__(parent)
        self.definition = d
        self.editor = editor
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 2, 1)
        layout.setSpacing(6)
        self.dot = QLabel()
        self.dot.setFixedSize(6, 6)
        self.full_label = label
        metrics = QFontMetrics(theme.ui_font(12.5))
        shown = metrics.elidedText(label, Qt.TextElideMode.ElideRight, LABEL_WIDTH - 2)
        self.label = QLabel(shown.replace("&", "&&"))
        self.label.setObjectName("rowLabel")
        self.label.setFixedWidth(LABEL_WIDTH)
        tip = d.tooltip or d.label
        self.label.setToolTip(f"{label}: {tip}\n\n{d.id} · right-click: reset to the default")
        editor.setToolTip(tip)
        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        layout.addWidget(editor, 1)
        for extra in extras:
            if extra is None:  # keep the column (rows without a key button line up)
                extra = QWidget()
                extra.setFixedSize(20, 20)
            layout.addWidget(extra)
        self.setMinimumHeight(30)
        self.setObjectName("propertyRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.set_changed(False)

    def flash(self) -> None:
        """Highlight the row for a moment (it was searched for)."""
        self._set_flash(True)
        timer = QTimer(self)  # (dies with the row)
        timer.setSingleShot(True)
        timer.timeout.connect(self._flash_off)
        timer.start(1400)

    def _flash_off(self) -> None:
        self._set_flash(False)

    def _set_flash(self, on: bool) -> None:
        self.setProperty("flash", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_changed(self, changed: bool) -> None:
        self.dot.setStyleSheet(f"background: {theme.ACCENT if changed else 'transparent'};"
                               "border-radius: 3px;")  # fmt: skip
        self.changed = changed

    def _menu(self, pos) -> None:
        menu = QMenu(self)
        reset = menu.addAction(theme.icon("arrow-u-up-left"), "Reset to the default")
        reset.setEnabled(self.changed)
        reset.triggered.connect(lambda: self.reset_requested.emit(self.definition.id))
        copy = menu.addAction(theme.icon("copy"), "Copy the setting's id")
        copy.triggered.connect(lambda: QGuiApplication.clipboard().setText(self.definition.id))
        menu.exec(self.mapToGlobal(pos))


class SectionView(QWidget):
    """A collapsible section: header (switch, menu), rows, tabs and "More settings".

    ``flat``: a list of search or filter results; every row is shown (no tabs, switch or
    "More settings") and the title (``display``, e.g. "Weather › Height Fog") is a link to
    the section (``jump_requested``)."""

    jump_requested = pyqtSignal(str)

    def __init__(self, title: str, defs: list[PropertyDef], store: PropertyStore,
                 make_row: Callable[[PropertyDef, str], PropertyRow],
                 on_edit: EditCallback, batch: Callable[[str, Callable], None] | None = None,
                 parent=None, display: str | None = None, flat: bool = False) -> None:  # fmt: skip
        super().__init__(parent)
        self.title = title
        self.display = display or title
        self.flat = flat
        self.store = store
        self.on_edit = on_edit
        self.batch = batch or (lambda label, fn: fn())
        self.defs = [d for d in defs if has_editor(d)]
        self.switch_def = None if flat else next(
            (d for d in self.defs if d.switch and d.type is PType.BOOL), None)  # fmt: skip
        self.tabs = [] if flat else find_tabs(self.defs)
        self.tab = self.tabs[0] if self.tabs else None
        self.more_open = False
        self._make_row = make_row
        self.setObjectName("section")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header())
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(6, 0, 6, 6)
        body.setSpacing(0)
        self.off_note = QLabel()
        self.off_note.setObjectName("dim")
        self.off_note.setContentsMargins(28, 0, 0, 8)
        layout.addWidget(self.off_note)
        self.rows: dict[str, PropertyRow] = {}
        rows = [d for d in self.defs if d is not self.switch_def]
        untabbed = [d for d in rows if tab_of(d, self.tabs) is None]
        tabbed = [d for d in rows if tab_of(d, self.tabs) is not None]
        for d in untabbed:
            self._add_row(body, d, None)
        self.tab_bar = None
        if self.tabs:
            self.tab_bar = Segments(self.tabs)
            self.tab_bar.set_index(0)
            self.tab_bar.chosen.connect(self._choose_tab)
            holder = QWidget()
            hl = tight_hbox(holder)
            hl.setContentsMargins(6, 4, 2, 4)
            hl.addWidget(self.tab_bar)
            body.addWidget(holder)
            for d in tabbed:
                self._add_row(body, d, tab_of(d, self.tabs))
        self.more = QToolButton()
        self.more.setObjectName("more")
        self.more.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.more.clicked.connect(self._toggle_more)
        body.addWidget(self.more, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.body)
        self.refresh()

    def _add_row(self, body, d: PropertyDef, tab: str | None) -> None:
        row = self._make_row(d, row_label(d, tab))
        row.reset_requested.connect(self._reset)
        row.tab = tab
        self.rows[d.id] = row
        body.addWidget(row)

    # --- header -----------------------------------------------------------------------
    def _header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("sectionHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(4, 4, 6, 4)
        layout.setSpacing(6)
        self.caret = QToolButton()
        self.caret.setObjectName("sectionTitle")
        self.caret.setText(self.display.replace("&", "&&"))
        self.caret.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.caret.setIconSize(QSize(14, 14))
        self.caret.setCheckable(True)
        self.caret.setChecked(True)
        self.caret.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if self.flat:
            self.caret.setCheckable(False)
            self.caret.setObjectName("crumb")
            self.caret.setToolTip("Show this section")
            self.caret.clicked.connect(self._jump)
        else:
            self.caret.toggled.connect(self._caret_toggled)
        layout.addWidget(self.caret, 1)
        self.switch = None
        if self.switch_def is not None:
            self.switch = SwitchButton()
            self.switch.setToolTip(f"{self.switch_def.label}: on / off")
            self.switch.toggled.connect(self._switched)
            layout.addWidget(self.switch)
        menu_button = QToolButton()
        menu_button.setIcon(theme.icon("dots-three"))
        menu_button.setIconSize(QSize(16, 16))
        menu_button.setToolTip("Reset, copy or paste this section")
        menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(menu_button)
        menu.addAction(theme.icon("arrow-u-up-left"), "Reset the section", self.reset_all)
        menu.addAction(theme.icon("copy"), "Copy the values", self.copy_values)
        menu.addAction(theme.icon("clipboard-text"), "Paste values", self.paste_values)
        menu_button.setMenu(menu)
        layout.addWidget(menu_button)
        return header

    def _jump(self) -> None:
        self.jump_requested.emit(self.title)

    def _caret_toggled(self, _open: bool) -> None:
        self.refresh()

    def _switched(self, on: bool) -> None:
        self.on_edit(self.switch_def.id, bool(on), False)

    @property
    def is_open(self) -> bool:
        return self.flat or self.caret.isChecked()

    def set_open(self, open_: bool) -> None:
        self.caret.setChecked(open_)

    # --- visibility -------------------------------------------------------------------
    def _choose_tab(self, index: int) -> None:
        self.tab = self.tabs[index]
        self.refresh()

    def _toggle_more(self) -> None:
        self.more_open = not self.more_open
        self.refresh()

    def row_visible(self, d: PropertyDef, values: dict) -> bool:
        if self.flat:
            return True
        row = self.rows[d.id]
        if row.tab is not None and row.tab != self.tab:
            return False
        if d.advanced and not self.more_open:
            return False
        return d.shown(values)

    def depends_on(self, pid: str) -> bool:
        """Whether a change of ``pid`` can show or hide rows (conditions, the switch)."""
        if self.switch_def is not None and self.switch_def.id == pid:
            return True
        return any(cond[0] == pid for d in self.defs for cond in d.when)

    def has_content(self) -> bool:
        """False when every row is hidden by its conditions (and there is no switch)."""
        values = self.store.values
        if self.flat:
            return True
        if self.switch_def is not None:
            return self.switch_def.shown(values)
        return any(d.shown(values) for d in self.defs)

    def refresh(self) -> None:
        values = self.store.values
        open_ = self.is_open
        if self.flat:
            self.caret.setIcon(theme.icon("arrow-right", theme.TEXT_3, theme.ACCENT_TEXT))
        else:
            self.caret.setIcon(theme.icon("caret-down" if open_ else "caret-right", theme.TEXT_3))
        on = True
        if self.switch is not None:
            on = bool(values.get(self.switch_def.id))
            self.switch.blockSignals(True)
            self.switch.setChecked(on)
            self.switch.blockSignals(False)
            self.switch.setVisible(self.switch_def.shown(values))
        self.caret.setProperty("off", not on)
        self.caret.style().unpolish(self.caret)
        self.caret.style().polish(self.caret)
        for d in self.defs:
            if d.id in self.rows:
                self.rows[d.id].setVisible(self.row_visible(d, values))
        hidden_advanced = [
            d
            for d in self.defs
            if d.id in self.rows
            and d.advanced
            and d.shown(values)
            and (self.rows[d.id].tab in (None, self.tab))
        ]
        self.more.setVisible(bool(hidden_advanced))
        self.more.setText("Fewer settings" if self.more_open
                          else f"More settings · {len(hidden_advanced)}")  # fmt: skip
        self.more.setIcon(theme.icon("caret-down" if self.more_open else "caret-right",
                                     theme.TEXT_3))  # fmt: skip
        self.body.setVisible(open_ and on)
        count = len(self.rows)
        self.off_note.setText(f"Off · switch on to edit its {count} settings")
        self.off_note.setVisible(open_ and not on)
        self.setVisible(self.has_content())

    # --- section menu -----------------------------------------------------------------
    def _reset(self, pid: str) -> None:
        self.on_edit(pid, self.store.registry[pid].default, False)

    def reset_all(self) -> None:
        def run() -> None:
            for d in self.defs:
                if self.store[d.id] != d.coerce(d.default):
                    self.on_edit(d.id, d.default, False)

        self.batch(f"Reset {self.title}", run)

    def copy_values(self) -> None:
        values = {d.id: d.to_json(self.store[d.id]) for d in self.defs}
        QGuiApplication.clipboard().setText(json.dumps(values, indent=1))

    def paste_values(self) -> None:
        try:
            values = json.loads(QGuiApplication.clipboard().text())
        except ValueError:
            return
        if not isinstance(values, dict):
            return
        mine = {d.id: d for d in self.defs}

        def run() -> None:
            for pid, raw in values.items():
                if pid in mine:
                    try:
                        self.on_edit(pid, mine[pid].from_json(raw), False)
                    except (TypeError, ValueError):
                        continue

        self.batch(f"Paste {self.title}", run)


class PropertyPanel(QScrollArea):
    """One section per property section; editors stay in sync with the store."""

    def __init__(
        self,
        store: PropertyStore,
        on_edit: EditCallback | None = None,
        row_extra: Callable[[PropertyDef], QWidget | None] | None = None,
        key_state: KeyState | None = None,
        batch: Callable[[str, Callable], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.store = store
        self.on_edit = on_edit or (lambda pid, value, interactive: store.set(pid, value))
        self.row_extra = row_extra
        self.key_state = key_state
        self.batch = batch
        self.editors: dict[str, PropertyEditor] = {}
        self.rows: dict[str, PropertyRow] = {}
        self.sections: dict[str, SectionView] = {}
        self._unsubscribe = None
        self.rebuild()

    def make_row(self, d: PropertyDef, label: str) -> PropertyRow:
        editor = editor_for(d)
        editor.set_value(self.store[d.id])
        editor.edited.connect(
            lambda value, interactive, pid=d.id: self.on_edit(pid, value, interactive)
        )
        extras = [self.row_extra(d)] if self.row_extra is not None else []
        row = PropertyRow(d, editor, label, extras)
        row.set_changed(self.store[d.id] != d.coerce(d.default))
        self.editors[d.id] = editor
        self.rows[d.id] = row
        return row

    def section_view(self, title: str, defs: list[PropertyDef]) -> SectionView:
        return SectionView(title, defs, self.store, self.make_row, self.on_edit, self.batch)

    def rebuild(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
        container = QWidget()
        container.setObjectName("panelBody")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.editors.clear()
        self.rows.clear()
        self.sections.clear()
        for title, defs in self.store.registry.sections():
            if not any(has_editor(d) for d in defs):
                continue
            section = self.section_view(title, defs)
            layout.addWidget(section)
            self.sections[title] = section
        layout.addStretch()
        retire(self.takeWidget())
        self.setWidget(container)
        self._dependents = self._dependent_sections()
        self.refresh_key_states()
        self._unsubscribe = self.store.subscribe(self._on_store_changed)

    def _dependent_sections(self) -> dict[str, list[SectionView]]:
        """Property id -> sections whose rows depend on it (``when``, switches)."""
        out: dict[str, list[SectionView]] = {}
        for section in self.sections.values():
            pids = {pid for d in section.defs for pid, _, _ in d.when}
            if section.switch_def is not None:
                pids.add(section.switch_def.id)
            for pid in pids:
                out.setdefault(pid, []).append(section)
        return out

    def refresh_key_states(self) -> None:
        if self.key_state is None:
            return
        for pid, editor in self.editors.items():
            editor.set_key_state(self.key_state(pid))

    def _on_store_changed(self, pid: str, value: Any) -> None:
        editor = self.editors.get(pid)
        if editor is not None:
            editor.set_value(value)
            d = self.store.registry[pid]
            self.rows[pid].set_changed(value != d.coerce(d.default))
        for section in self._dependents.get(pid, []):
            section.refresh()
