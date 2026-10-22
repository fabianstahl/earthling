"""Shared parts of the key views (dope sheet, graph editor, ruler): colours, the key symbol,
selection handling, keyboard and context menu helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QKeyEvent, QKeySequence, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QMenu

from earthling.app.keyframing import KeyframeEditor, KeyRef
from earthling.core.animation import INTERP_LABELS, Curve, Interp
from earthling.ui import theme

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
    """Mixin for a widget with ``editor`` (KeyframeEditor) and ``selection`` (set of KeyRef);
    subclasses implement ``select_all``."""

    editor: KeyframeEditor
    selection: set[KeyRef]

    def select_all(self) -> None:
        raise NotImplementedError

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

    def handle_selection_key(self, event: QKeyEvent) -> bool:
        """Delete / Backspace deletes the selected keys, Ctrl+A selects all. True if handled."""
        if event.matches(QKeySequence.StandardKey.Delete) or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            self.select_all()
        else:
            return False
        return True
