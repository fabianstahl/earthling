"""Dope sheet: one row per animated property with draggable keyframe diamonds."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import QMenu, QSizePolicy, QToolButton, QWidget

from earthling.app.keyframing import KeyframeEditor, KeyRef
from earthling.core.animation import INTERP_LABELS, Interp, layer_switch_warnings, snap_to_frame
from earthling.ui.timeline_widget import diamond

LABEL_WIDTH = 200
ROW_HEIGHT = 20
KEY_RADIUS = 5.0
KEY_COLOR = QColor(240, 190, 60)
SELECTED_COLOR = QColor(255, 255, 255)
PLAYHEAD_COLOR = QColor(90, 170, 255)
WARN_COLOR = QColor(235, 70, 60)  # keys that cause a visible jump


class DopeSheet(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, editor: KeyframeEditor, time_to_x, x_to_time, parent=None) -> None:
        """``time_to_x`` / ``x_to_time`` are shared with the ruler so both line up."""
        super().__init__(parent)
        self.editor = editor
        self.time_to_x = time_to_x
        self.x_to_time = x_to_time
        self.selection: set[KeyRef] = set()
        self._drag_origin: QPointF | None = None
        self._drag_keys: list[KeyRef] = []
        self._drag_dt = 0.0
        self._band: QRectF | None = None
        self._warnings: list = []
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(ROW_HEIGHT * 3)
        editor.changed.connect(self._on_animation_changed)
        editor.timeline.time_changed.connect(lambda t: self.update())
        editor.timeline.settings_changed.connect(self.update)

    # --- data -------------------------------------------------------------------------
    def rows(self) -> list[str]:
        registry = self.editor.animation.store.registry
        order = {d.id: i for i, d in enumerate(registry)}
        pids = [pid for pid, c in self.editor.animation.curves.items() if c.keys]
        return sorted(pids, key=lambda pid: order.get(pid, 1_000_000))

    def row_y(self, index: int) -> float:
        return index * ROW_HEIGHT + ROW_HEIGHT / 2

    def key_at(self, pos: QPointF) -> KeyRef | None:
        index = int(pos.y() // ROW_HEIGHT)
        rows = self.rows()
        if not 0 <= index < len(rows):
            return None
        curve = self.editor.animation.curves[rows[index]]
        best, best_dx = None, KEY_RADIUS + 2
        for k in curve.keys:
            dx = abs(self.time_to_x(k.time) - pos.x())
            if dx <= best_dx:
                best, best_dx = (rows[index], k.time), dx
        return best

    def _on_animation_changed(self) -> None:
        existing = {
            (pid, k.time) for pid, c in self.editor.animation.curves.items() for k in c.keys
        }
        self.selection &= existing
        self.setMinimumHeight(max(ROW_HEIGHT * 3, ROW_HEIGHT * len(self.rows())))
        self.update()

    # --- painting ---------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        rows = self.rows()
        registry = self.editor.animation.store.registry
        self._warnings = layer_switch_warnings(self.editor.animation)
        for i, pid in enumerate(rows):
            y0 = i * ROW_HEIGHT
            if i % 2:
                p.fillRect(QRectF(0, y0, self.width(), ROW_HEIGHT), pal.alternateBase())
            p.setPen(pal.text().color())
            d = registry[pid]
            p.drawText(
                QRectF(6, y0, LABEL_WIDTH - 10, ROW_HEIGHT),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"{d.section}: {d.label}",
            )
            curve = self.editor.animation.curves[pid]
            # segments with interpolation hint
            pen = QPen(pal.mid().color(), 2)
            p.setPen(pen)
            for k0, k1 in zip(curve.keys, curve.keys[1:], strict=False):
                if k0.interp is not Interp.STEP and k0.value != k1.value:
                    y = y0 + ROW_HEIGHT / 2
                    p.drawLine(
                        QPointF(self.time_to_x(k0.time), y), QPointF(self.time_to_x(k1.time), y)
                    )
            warn_times = {t for w_pid, t, _ in self._warnings if w_pid == pid}
            for k in curve.keys:
                t = k.time
                selected = (pid, t) in self.selection
                warned = t in warn_times
                if selected and self._drag_keys and (pid, t) in self._drag_keys:
                    t = t + self._drag_dt
                p.setPen(QPen(QColor(30, 30, 30), 1))
                p.setBrush(SELECTED_COLOR if selected else (WARN_COLOR if warned else KEY_COLOR))
                p.drawPolygon(diamond(QPointF(self.time_to_x(t), y0 + ROW_HEIGHT / 2), KEY_RADIUS))
        if not rows:
            p.setPen(pal.placeholderText().color())
            p.drawText(
                self.rect().adjusted(LABEL_WIDTH, 0, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "No animated properties yet – click ◆ next to a parameter to add a key",
            )
        p.setPen(QPen(pal.mid().color()))
        p.drawLine(QPointF(LABEL_WIDTH, 0), QPointF(LABEL_WIDTH, self.height()))
        x = self.time_to_x(self.editor.timeline.time)
        p.setPen(QPen(PLAYHEAD_COLOR, 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        if self._band is not None:
            p.setPen(QPen(PLAYHEAD_COLOR, 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(90, 170, 255, 40))
            p.drawRect(self._band)
        p.end()

    # --- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        key = self.key_at(pos)
        mods = event.modifiers()
        additive = bool(
            mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier)
        )
        if event.button() == Qt.MouseButton.RightButton:
            if key is not None and key not in self.selection:
                self.selection = {key}
            self._context_menu(event.globalPosition().toPoint())
            return
        if key is not None:
            if additive:
                self.selection ^= {key}
            elif key not in self.selection:
                self.selection = {key}
            self._drag_origin = pos
            self._drag_keys = sorted(self.selection)
            self._drag_dt = 0.0
        else:
            if not additive:
                self.selection = set()
            self._band = QRectF(pos, pos)
        self.selection_changed.emit()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._drag_origin is not None and self._drag_keys:
            dt = self.x_to_time(pos.x()) - self.x_to_time(self._drag_origin.x())
            self._drag_dt = snap_to_frame(dt, self.editor.animation.fps)
            self.update()
        elif self._band is not None:
            self._band.setBottomRight(pos)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None and self._drag_keys and self._drag_dt != 0:
            moved = self.editor.move_keys(self._drag_keys, self._drag_dt)
            self.selection = set(moved)
        elif self._band is not None:
            band = self._band.normalized()
            rows = self.rows()
            for i, pid in enumerate(rows):
                if band.top() <= self.row_y(i) <= band.bottom():
                    for k in self.editor.animation.curves[pid].keys:
                        if band.left() <= self.time_to_x(k.time) <= band.right():
                            self.selection.add((pid, k.time))
        self._drag_origin = None
        self._drag_keys = []
        self._drag_dt = 0.0
        self._band = None
        self.selection_changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        key = self.key_at(event.position())
        if key is not None:
            self.editor.timeline.set_time(key[1])

    # --- keyboard / menu --------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.matches(QKeySequence.StandardKey.Delete) or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.matches(QKeySequence.StandardKey.Copy):
            self.editor.copy(self.selection)
        elif event.matches(QKeySequence.StandardKey.Paste):
            self.selection = set(self.editor.paste())
            self.update()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            self.selection = {
                (pid, k.time) for pid, c in self.editor.animation.curves.items() for k in c.keys
            }
            self.update()
        else:
            super().keyPressEvent(event)

    def delete_selected(self) -> None:
        if self.selection:
            self.editor.delete_keys(self.selection)
            self.selection = set()
            self.update()

    def _context_menu(self, global_pos) -> None:
        menu = QMenu(self)
        interp_menu = menu.addMenu("Interpolation")
        for interp in Interp:
            action = QAction(INTERP_LABELS[interp], menu)
            action.triggered.connect(
                lambda _=False, i=interp: self.editor.set_interpolation(self.selection, i)
            )
            action.setEnabled(bool(self.selection))
            interp_menu.addAction(action)
        menu.addSeparator()
        for text, slot, enabled in (
            ("Copy", lambda: self.editor.copy(self.selection), bool(self.selection)),
            ("Paste at playhead", lambda: setattr(self, "selection", set(self.editor.paste())),
             bool(self.editor.clipboard)),
            ("Delete", self.delete_selected, bool(self.selection)),
        ):  # fmt: skip
            action = QAction(text, menu)
            action.triggered.connect(slot)
            action.setEnabled(enabled)
            menu.addAction(action)
        menu.exec(global_pos)


class KeyButton(QToolButton):
    """◆ next to a property: hollow grey = not animated, hollow orange = animated,
    filled orange = key at the current time. Click toggles the key at the playhead."""

    def __init__(self, pid: str, editor: KeyframeEditor, parent=None) -> None:
        super().__init__(parent)
        self.pid = pid
        self.editor = editor
        self.setAutoRaise(True)
        self.setFixedSize(20, 20)
        self.clicked.connect(lambda: editor.toggle_key(pid))
        # bound methods (not lambdas): disconnected automatically when the panel is rebuilt
        editor.changed.connect(self.update)
        editor.timeline.time_changed.connect(self._on_time_changed)
        self.setToolTip("Insert / delete a key at the current time")

    def _on_time_changed(self, time: float) -> None:
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        animated = self.editor.animation.is_animated(self.pid)
        keyed = animated and self.editor.has_key_now(self.pid)
        color = KEY_COLOR if animated else self.palette().mid().color()
        p.setPen(QPen(color, 1.5))
        p.setBrush(color if keyed else Qt.BrushStyle.NoBrush)
        p.drawPolygon(diamond(QPointF(self.width() / 2, self.height() / 2), 5.5))
        p.end()
