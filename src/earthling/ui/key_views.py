"""Shared parts of the key views (dope sheet, graph editor, ruler): colours, the key symbol,
selection handling, keyboard navigation and context menu helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QKeyEvent, QKeySequence, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMenu, QWidget

from earthling.app.keyframing import KeyframeEditor, KeyRef
from earthling.core.animation import INTERP_LABELS, Curve, Interp, snap_to_frame
from earthling.core.properties import PropertyDef
from earthling.ui import theme
from earthling.ui.property_panel import FloatEditor, editor_for, has_editor

KEY_RADIUS = 5.0
KEY_COLOR = QColor(theme.KEY)
KEY_OUTLINE = QColor(theme.GROUND)
SELECTED_COLOR = QColor(255, 255, 255)
PLAYHEAD_COLOR = QColor(theme.ACCENT)
ADDITIVE = Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier


def diamond(center: QPointF, r: float) -> QPolygonF:
    """The keyframe symbol."""
    x, y = center.x(), center.y()
    return QPolygonF([QPointF(x, y - r), QPointF(x + r, y), QPointF(x, y + r), QPointF(x - r, y)])


def all_keys(curves: Iterable[tuple[str, Curve]]) -> set[KeyRef]:
    return {(pid, k.time) for pid, c in curves for k in c.keys}


def paint_band(p: QPainter, band: QRectF) -> None:
    """The rubber band of a box selection."""
    p.setPen(QPen(PLAYHEAD_COLOR, 1, Qt.PenStyle.DashLine))
    p.setBrush(theme.qcolor(theme.ACCENT, 40))
    p.drawRect(band)


def paint_focus(p: QPainter, widget: QWidget) -> None:
    """A thin accent frame around the visible part of a focused key view."""
    if widget.hasFocus():
        p.setPen(QPen(theme.qcolor(theme.ACCENT, 140), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(widget.visibleRegion().boundingRect()).adjusted(0.5, 0.5, -0.5, -0.5))


def paint_current(p: QPainter, center: QPointF) -> None:
    """The ring around the current key (the one the arrow keys move from)."""
    p.setPen(QPen(PLAYHEAD_COLOR, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(center, KEY_RADIUS + 3.5, KEY_RADIUS + 3.5)


def ask_key_value(parent: QWidget, d: PropertyDef, value: Any) -> Any | None:
    """The property's editor in a small dialog: the new value, or None if cancelled."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Key value")
    editor = editor_for(d)
    editor.set_value(value)
    result = [value]
    editor.edited.connect(lambda v, interactive: result.__setitem__(0, v))
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout = QFormLayout(dialog)
    layout.addRow(f"{d.section}: {d.label}", editor)
    layout.addRow(buttons)
    if isinstance(editor, FloatEditor):  # a number: typed right away, Enter confirms
        editor.field.start_typing()
        dialog.findChild(QLineEdit).returnPressed.connect(dialog.accept)
    return result[0] if dialog.exec() == QDialog.DialogCode.Accepted else None


def add_choice_menu(menu: QMenu, title: str, choices: Iterable[tuple[Any, str]],
                    slot: Callable[[Any], None], enabled: bool) -> None:  # fmt: skip
    """Submenu with one action per ``(value, label)``; triggering calls ``slot(value)``."""
    sub = menu.addMenu(title)
    for value, label in choices:
        action = QAction(label, menu)
        action.triggered.connect(lambda _=False, v=value: slot(v))
        action.setEnabled(enabled)
        sub.addAction(action)


def add_actions(menu: QMenu, actions: Iterable[tuple[str, Callable, bool]]) -> None:
    """One action per ``(text, slot, enabled)``."""
    for text, slot, enabled in actions:
        action = QAction(text, menu)
        action.triggered.connect(slot)
        action.setEnabled(enabled)
        menu.addAction(action)


class KeySelection:
    """Mixin for a widget with ``editor`` (KeyframeEditor), ``selection`` (set of KeyRef) and
    ``selection_changed``; subclasses implement ``select_all`` and ``key_rows``."""

    editor: KeyframeEditor
    selection: set[KeyRef]
    current: KeyRef | None = None  # where the arrow keys move from (the last clicked key)

    def select_all(self) -> None:
        raise NotImplementedError

    def key_rows(self) -> list[tuple[str, Curve]]:
        """The curves in the order shown (the rows of Up / Down)."""
        raise NotImplementedError

    def current_key(self) -> KeyRef | None:
        """The current key, else the first selected one (e.g. after a drag moved it)."""
        keys = [(pid, k.time) for pid, c in self.key_rows() for k in c.keys]
        if self.current in keys:
            return self.current
        return next((ref for ref in keys if ref in self.selection), None)

    def prune_selection(self) -> None:
        """Forget selected keys that no longer exist."""
        self.selection &= all_keys(self.editor.animation.curves.items())

    def delete_selected(self) -> None:
        if self.selection:
            self.editor.delete_keys(self.selection)
            self.selection = set()
            self.update()

    def set_interpolation(self, interp: Interp) -> None:
        if self.selection:
            self.editor.set_interpolation(self.selection, interp)

    def add_interpolation_menu(self, menu: QMenu) -> None:
        choices = [(interp, INTERP_LABELS[interp]) for interp in Interp]
        add_choice_menu(menu, "Interpolation", choices, self.set_interpolation,
                        bool(self.selection))  # fmt: skip

    def step_current(self, steps: int, rows: int, extend: bool) -> None:
        """Make the key ``steps`` keys later on its row, or the key nearest in time ``rows``
        rows further down, current and selected (``extend``: added to the selection)."""
        curves = self.key_rows()
        if not curves:
            return
        pids = [pid for pid, _ in curves]
        old = self.current_key()
        if old is None:  # start on the first row, at the playhead
            row, t, steps, rows = 0, self.editor.timeline.time, 0, 0
        else:
            row, t = pids.index(old[0]), old[1]
        row = min(max(row + rows, 0), len(curves) - 1)
        keys = curves[row][1].keys
        i = min(range(len(keys)), key=lambda j: abs(keys[j].time - t))
        new = (pids[row], keys[min(max(i + steps, 0), len(keys) - 1)].time)
        if extend:
            self.selection |= {new} if old is None else {old, new}
        else:
            self.selection = {new}
        self.current = new

    def nudge_selected(self, frames: int) -> None:
        """Move the selected keys by whole frames (one undo step)."""
        if not self.selection:
            return
        fps = self.editor.animation.fps
        dt = snap_to_frame(frames / fps, fps)
        old = self.current_key()
        if old in self.selection:  # (the current key moves along)
            self.current = (old[0], max(0.0, snap_to_frame(old[1] + dt, fps)))
        self.selection = set(self.editor.move_keys(self.selection, dt))

    def edit_current_value(self) -> None:
        """The value of the current key in a small dialog (undoable)."""
        ref = self.current_key()
        curve = self.editor.animation.curves.get(ref[0]) if ref is not None else None
        if curve is None or not has_editor(curve.definition):  # (camera poses have none)
            return
        value = ask_key_value(self, curve.definition, curve.key_at(ref[1]).value)
        if value is not None:
            self.editor.set_key_value(*ref, value)

    def handle_selection_key(self, event: QKeyEvent) -> bool:
        """Delete / Backspace deletes the selected keys, Ctrl+A selects all, Esc none; the
        arrows walk the keys (Shift: extend the selection), Alt+Left / Right move the selected
        keys by a frame (Shift: 10), Enter / F2 edit the current key's value. True if handled."""
        key, mods = event.key(), event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        plain = not mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                  Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}  # fmt: skip
        if event.matches(QKeySequence.StandardKey.Delete) or key == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            self.select_all()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and alt and plain:
            self.nudge_selected((1 if key == Qt.Key.Key_Right else -1) * (10 if shift else 1))
        elif key in arrows and not alt and plain:
            self.step_current(*arrows[key], extend=shift)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2):
            self.edit_current_value()
        elif key == Qt.Key.Key_Escape and self.selection:
            self.selection = set()
        else:
            return False
        self.selection_changed.emit()
        self.update()
        return True
