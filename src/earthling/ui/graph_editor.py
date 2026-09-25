"""Graph editor: the animation curves as value over time with editable Bézier handles.

Float/int/datetime properties are plotted by value; colours, vectors and the camera by
"progress" (key index + blend factor), where the handles shape the timing between keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.app.keyframing import KeyframeEditor, KeyRef
from earthling.core.animation import (
    HANDLE_LABELS,
    INTERP_LABELS,
    Curve,
    HandleMode,
    Interp,
    Keyframe,
    snap_to_frame,
)
from earthling.core.properties import PType
from earthling.ui.timeline_widget import diamond, nice_tick_step

CHANNEL_COLORS = [
    QColor(240, 110, 90), QColor(110, 200, 110), QColor(100, 160, 250), QColor(240, 190, 60),
    QColor(200, 120, 230), QColor(90, 210, 210), QColor(245, 120, 180), QColor(180, 180, 180),
]  # fmt: skip
SELECTED_COLOR = QColor(255, 255, 255)
PLAYHEAD_COLOR = QColor(90, 170, 255)
KEY_RADIUS = 5.0
HANDLE_RADIUS = 3.5
PICK_RADIUS = 7.0
AXIS_LEFT = 58  # value labels
AXIS_BOTTOM = 20  # time labels (click/drag there to scrub)


def nice_value_step(span: float, pixels: float, min_spacing_px: float = 36.0) -> float:
    raw = max(span, 1e-12) / max(pixels, 1.0) * min_spacing_px
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * magnitude >= raw:
            return m * magnitude
    return 10 * magnitude


@dataclass
class _Drag:
    kind: str  # "keys", "handle", "pan", "scrub"
    press: QPointF
    keys: list[tuple[str, Keyframe, float, float, object]] | None = None  # pid, key, t, v, value
    handle: tuple[str, Keyframe, str] | None = None  # pid, key, side
    view: tuple[float, float, float, float] | None = None
    moved: bool = False


class GraphView(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, editor: KeyframeEditor, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self.channels: list[str] = []  # visible curves (property ids)
        self.colors: dict[str, QColor] = {}
        self.selection: set[KeyRef] = set()
        self.normalized = False
        self.t0, self.t1 = 0.0, max(editor.animation.duration, 1.0)
        self.v0, self.v1 = -0.1, 1.1
        self._drag: _Drag | None = None
        self._band: QRectF | None = None
        self._frozen: dict[str, tuple[float, float]] | None = None  # normalisation during drags
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(120)
        editor.changed.connect(self._on_changed)
        # bound methods (not lambdas) are disconnected automatically when the view is deleted
        editor.timeline.time_changed.connect(self._on_time_changed)

    # --- data -------------------------------------------------------------------------
    def curve(self, pid: str) -> Curve | None:
        curve = self.editor.animation.curves.get(pid)
        return curve if curve is not None and curve.keys else None

    def visible_curves(self) -> list[tuple[str, Curve]]:
        out = []
        for pid in self.channels:
            curve = self.curve(pid)
            if curve is not None and curve.graph_kind is not None:
                out.append((pid, curve))
        return out

    def _on_time_changed(self, time: float) -> None:
        self.update()

    def _on_changed(self) -> None:
        existing = {
            (pid, k.time) for pid, c in self.editor.animation.curves.items() for k in c.keys
        }
        self.selection &= existing
        self.update()

    def curve_range(self, curve: Curve) -> tuple[float, float]:
        """Scalar range of a curve including its handles (for framing / normalising)."""
        values = curve.scalars()
        points = list(values)
        for i in range(len(curve.keys)):
            points += [v for _, v in curve.handle_points(i, values).values()]
        t_first, t_last = curve.keys[0].time, curve.keys[-1].time
        if t_last > t_first:
            for n in range(65):
                points.append(curve.scalar_at(t_first + (t_last - t_first) * n / 64))
        return min(points), max(points)

    def transform(self, pid: str, curve: Curve) -> tuple[float, float]:
        """(offset, scale): displayed = (scalar - offset) * scale."""
        if not self.normalized:
            return 0.0, 1.0
        if self._frozen is not None and pid in self._frozen:
            return self._frozen[pid]
        lo, hi = self.curve_range(curve)
        return lo, (1.0 / (hi - lo) if hi - lo > 1e-12 else 1.0)

    # --- view mapping -----------------------------------------------------------------
    def plot_rect(self) -> QRectF:
        return QRectF(AXIS_LEFT, 6, max(10, self.width() - AXIS_LEFT - 8),
                      max(10, self.height() - 6 - AXIS_BOTTOM))  # fmt: skip

    def tx(self, t: float) -> float:
        r = self.plot_rect()
        return r.left() + (t - self.t0) / (self.t1 - self.t0) * r.width()

    def vy(self, v: float) -> float:
        r = self.plot_rect()
        return r.bottom() - (v - self.v0) / (self.v1 - self.v0) * r.height()

    def x_to_t(self, x: float) -> float:
        r = self.plot_rect()
        return self.t0 + (x - r.left()) / r.width() * (self.t1 - self.t0)

    def y_to_v(self, y: float) -> float:
        r = self.plot_rect()
        return self.v0 + (r.bottom() - y) / r.height() * (self.v1 - self.v0)

    def point(self, pid: str, curve: Curve, t: float, scalar: float) -> QPointF:
        offset, scale = self.transform(pid, curve)
        return QPointF(self.tx(t), self.vy((scalar - offset) * scale))

    def frame(self, t_range: tuple[float, float], v_range: tuple[float, float]) -> None:
        (ta, tb), (va, vb) = t_range, v_range
        if tb - ta < 1e-6:
            ta, tb = ta - 1.0, tb + 1.0
        if vb - va < 1e-9:
            pad = max(abs(va) * 0.1, 0.5)
            va, vb = va - pad, vb + pad
        pt, pv = (tb - ta) * 0.05, (vb - va) * 0.1
        self.t0, self.t1, self.v0, self.v1 = ta - pt, tb + pt, va - pv, vb + pv
        self.update()

    def _extent(self, only_selected: bool) -> tuple[tuple[float, float], tuple[float, float]]:
        ts, vs = [], []
        for pid, curve in self.visible_curves():
            offset, scale = self.transform(pid, curve)
            values = curve.scalars()
            for i, k in enumerate(curve.keys):
                if only_selected and (pid, k.time) not in self.selection:
                    continue
                ts.append(k.time)
                vs.append((values[i] - offset) * scale)
                for t, v in curve.handle_points(i, values).values():
                    ts.append(t)
                    vs.append((v - offset) * scale)
            if not only_selected:
                lo, hi = self.curve_range(curve)
                vs += [(lo - offset) * scale, (hi - offset) * scale]
        if not ts:
            return (0.0, max(self.editor.animation.duration, 1.0)), (0.0, 1.0)
        return (min(ts), max(ts)), (min(vs), max(vs))

    def frame_all(self) -> None:
        self.frame(*self._extent(False))

    def frame_selected(self) -> None:
        if self.selection:
            self.frame(*self._extent(True))
        else:
            self.frame_all()

    def set_normalized(self, on: bool) -> None:
        self.normalized = on
        self.frame_all()

    # --- hit testing ------------------------------------------------------------------
    def hit(self, pos: QPointF):
        """('handle', pid, key, side) | ('key', pid, key) | None. Keys win over handles at
        the same distance (short handles must not hide their key)."""
        best_handle, handle_d = None, PICK_RADIUS
        best_key, key_d = None, PICK_RADIUS
        for pid, curve in self.visible_curves():
            values = curve.scalars()
            for i, k in enumerate(curve.keys):
                d = _dist(self.point(pid, curve, k.time, values[i]), pos)
                if d < key_d:
                    best_key, key_d = ("key", pid, k), d
                if (pid, k.time) in self.selection:
                    for side, (t, v) in curve.handle_points(i, values).items():
                        d = _dist(self.point(pid, curve, t, v), pos)
                        if d < handle_d:
                            best_handle, handle_d = ("handle", pid, k, side), d
        if best_handle is not None and handle_d < key_d:
            return best_handle
        return best_key

    # --- painting ---------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        r = self.plot_rect()
        duration = self.editor.animation.duration
        # outside of the animation range
        shade = QColor(pal.text().color())
        shade.setAlpha(14)
        for a, b in ((self.t0, 0.0), (duration, self.t1)):
            if b > a:
                p.fillRect(QRectF(self.tx(a), r.top(), self.tx(b) - self.tx(a), r.height()), shade)
        self._paint_grid(p, r)
        curves = self.visible_curves()
        p.setClipRect(r)
        for pid, curve in curves:
            self._paint_curve(p, pid, curve)
        x = self.tx(self.editor.timeline.time)
        p.setPen(QPen(PLAYHEAD_COLOR, 2))
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom() + AXIS_BOTTOM))
        p.setClipping(False)
        if self._band is not None:
            p.setPen(QPen(PLAYHEAD_COLOR, 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(90, 170, 255, 40))
            p.drawRect(self._band)
        if not curves:
            p.setPen(pal.placeholderText().color())
            hint = "No curves to show – animate a numeric property or tick a channel"
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, hint)
        p.end()

    def _paint_grid(self, p: QPainter, r: QRectF) -> None:
        pal = self.palette()
        grid = QPen(pal.mid().color(), 1)
        grid.setCosmetic(True)
        text = QPen(pal.text().color())
        step = nice_tick_step(r.width() / (self.t1 - self.t0))
        t = math.ceil(self.t0 / step) * step
        while t <= self.t1:
            x = self.tx(t)
            p.setPen(grid)
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(text)
            label = f"{t:.0f}s" if step >= 1 else f"{t:.2f}s"
            p.drawText(QPointF(x + 3, r.bottom() + 14), label)
            t += step
        vstep = nice_value_step(self.v1 - self.v0, r.height())
        v = math.ceil(self.v0 / vstep) * vstep
        datetime_axis = self._datetime_axis()
        while v <= self.v1:
            y = self.vy(v)
            p.setPen(grid)
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(text)
            p.drawText(QRectF(0, y - 8, AXIS_LEFT - 6, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self._format_value(v, vstep, datetime_axis))  # fmt: skip
            v += vstep
        p.setPen(QPen(pal.mid().color()))
        p.drawRect(r)

    def _datetime_axis(self):
        """The timezone-aware 'like' value when all visible channels are datetimes."""
        curves = self.visible_curves()
        if self.normalized or not curves:
            return None
        if all(c.definition.type is PType.DATETIME for _, c in curves):
            return curves[0][1].keys[0].value
        return None

    @staticmethod
    def _format_value(v: float, step: float, datetime_like) -> str:
        if datetime_like is not None:
            from datetime import datetime

            dt = datetime.fromtimestamp(v, tz=datetime_like.tzinfo)
            return dt.strftime("%H:%M")
        if abs(v) < step * 1e-6:
            v = 0.0
        return f"{v:.4g}"

    def _paint_curve(self, p: QPainter, pid: str, curve: Curve) -> None:
        color = self.colors.get(pid, CHANNEL_COLORS[0])
        r = self.plot_rect()
        path = QPainterPath()
        n = max(2, int(r.width() / 2))
        for j in range(n + 1):
            t = self.t0 + (self.t1 - self.t0) * j / n
            pt = self.point(pid, curve, t, curve.scalar_at(t))
            if j == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, 1.8))
        p.drawPath(path)
        values = curve.scalars()
        dragged = self._drag.handle[1] if self._drag and self._drag.handle else None
        for i, k in enumerate(curve.keys):
            center = self.point(pid, curve, k.time, values[i])
            selected = (pid, k.time) in self.selection
            if selected or k is dragged:
                for _side, (t, v) in curve.handle_points(i, values).items():
                    hp = self.point(pid, curve, t, v)
                    p.setPen(QPen(color.lighter(130), 1))
                    p.drawLine(center, hp)
                    p.setBrush(SELECTED_COLOR)
                    p.setPen(QPen(QColor(30, 30, 30), 1))
                    p.drawEllipse(hp, HANDLE_RADIUS, HANDLE_RADIUS)
            p.setPen(QPen(QColor(30, 30, 30), 1))
            p.setBrush(SELECTED_COLOR if selected else color)
            p.drawPolygon(diamond(center, KEY_RADIUS))

    # --- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        button = event.button()
        mods = event.modifiers()
        if button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.AltModifier
        ):
            self._drag = _Drag("pan", pos, view=(self.t0, self.t1, self.v0, self.v1))
            return
        if button == Qt.MouseButton.RightButton:
            hit = self.hit(pos)
            if hit is not None and hit[0] == "key" and (hit[1], hit[2].time) not in self.selection:
                self.selection = {(hit[1], hit[2].time)}
                self.selection_changed.emit()
            self._context_menu(event.globalPosition().toPoint())
            return
        if button != Qt.MouseButton.LeftButton:
            return
        if pos.y() > self.plot_rect().bottom():
            self._drag = _Drag("scrub", pos)
            self._scrub(pos)
            return
        additive = bool(
            mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier)
        )
        hit = self.hit(pos)
        if hit is not None and hit[0] == "handle":
            _, pid, key, side = hit
            self._freeze_normalisation()
            self.editor.begin([pid])
            self._drag = _Drag("handle", pos, handle=(pid, key, side))
        elif hit is not None:
            _, pid, key = hit
            ref = (pid, key.time)
            if additive:
                self.selection ^= {ref}
            elif ref not in self.selection:
                self.selection = {ref}
            self._start_key_drag(pos)
        else:
            if not additive:
                self.selection = set()
            self._band = QRectF(pos, pos)
        self.selection_changed.emit()
        self.update()

    def _freeze_normalisation(self) -> None:
        self._frozen = {pid: self.transform(pid, c) for pid, c in self.visible_curves()}

    def _start_key_drag(self, pos: QPointF) -> None:
        keys = []
        curves = dict(self.visible_curves())
        for pid, t in sorted(self.selection):
            curve = curves.get(pid)
            key = curve.key_at(t) if curve is not None else None
            if key is not None:
                i = curve.keys.index(key)
                keys.append((pid, key, key.time, curve.scalars()[i], key.value))
        if not keys:
            return
        self._freeze_normalisation()
        self.editor.begin({pid for pid, *_ in keys})
        self._drag = _Drag("keys", pos, keys=keys)

    def _scrub(self, pos: QPointF) -> None:
        self.editor.timeline.pause()
        self.editor.timeline.set_time(min(max(self.x_to_t(pos.x()), 0.0),
                                          self.editor.animation.duration))  # fmt: skip

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        drag = self._drag
        if drag is None:
            if self._band is not None:
                self._band.setBottomRight(pos)
                self.update()
            return
        if drag.kind == "pan":
            t0, t1, v0, v1 = drag.view
            r = self.plot_rect()
            dt = (pos.x() - drag.press.x()) / r.width() * (t1 - t0)
            dv = (pos.y() - drag.press.y()) / r.height() * (v1 - v0)
            self.t0, self.t1, self.v0, self.v1 = t0 - dt, t1 - dt, v0 + dv, v1 + dv
            self.update()
        elif drag.kind == "scrub":
            self._scrub(pos)
        elif drag.kind == "keys":
            self._drag_keys(drag, pos, event.modifiers())
        elif drag.kind == "handle":
            pid, key, side = drag.handle
            curve = self.curve(pid)
            if curve is None or key not in curve.keys:
                return
            offset, scale = self.transform(pid, curve)
            scalar = self.y_to_v(pos.y()) / scale + offset
            curve.drag_handle(curve.keys.index(key), side, self.x_to_t(pos.x()), scalar)
            drag.moved = True
            self.editor.live()

    def _drag_keys(self, drag: _Drag, pos: QPointF, mods) -> None:
        fps = self.editor.animation.fps
        dt = snap_to_frame(self.x_to_t(pos.x()) - self.x_to_t(drag.press.x()), fps)
        dv_display = self.y_to_v(pos.y()) - self.y_to_v(drag.press.y())
        # Shift: constrain to the dominant axis
        if mods & Qt.KeyboardModifier.ShiftModifier:
            if abs(pos.x() - drag.press.x()) > abs(pos.y() - drag.press.y()):
                dv_display = 0.0
            else:
                dt = 0.0
        touched = set()
        for pid, key, t, v, value in drag.keys:
            curve = self.curve(pid)
            if curve is None:
                continue
            key.time = max(0.0, t + dt)
            if curve.graph_kind == "value":
                _, scale = self.transform(pid, curve)
                key.value = curve.from_scalar(v + dv_display / scale, value)
            touched.add(pid)
        for pid in touched:
            self.curve(pid).sort()
        drag.moved = True
        self.selection = {(pid, key.time) for pid, key, *_ in drag.keys}
        self.editor.live()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        drag, self._drag = self._drag, None
        self._frozen = None
        if drag is not None and drag.kind == "keys":
            if drag.moved:
                self._resolve_collisions(drag)
            self.editor.commit("Move keys")
            self.selection = {(pid, key.time) for pid, key, *_ in drag.keys}
        elif drag is not None and drag.kind == "handle":
            self.editor.commit("Edit handle")
        elif self._band is not None:
            band = self._band.normalized()
            for pid, curve in self.visible_curves():
                values = curve.scalars()
                for i, k in enumerate(curve.keys):
                    if band.contains(self.point(pid, curve, k.time, values[i])):
                        self.selection.add((pid, k.time))
            self._band = None
        self.selection_changed.emit()
        self.update()

    def _resolve_collisions(self, drag: _Drag) -> None:
        """Dragged keys replace keys they were dropped on."""
        dragged = {id(key) for _, key, *_ in drag.keys}
        for pid in {pid for pid, *_ in drag.keys}:
            curve = self.curve(pid)
            if curve is None:
                continue
            times = {round(k.time, 6) for k in curve.keys if id(k) in dragged}
            curve.keys = [
                k for k in curve.keys if id(k) in dragged or round(k.time, 6) not in times
            ]

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        hit = self.hit(event.position())
        if hit is not None and hit[0] == "key":
            self.editor.timeline.set_time(hit[2].time)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 0.85 ** (event.angleDelta().y() / 120.0)
        pos = event.position()
        mods = event.modifiers()
        t, v = self.x_to_t(pos.x()), self.y_to_v(pos.y())
        if not mods & Qt.KeyboardModifier.ControlModifier:  # Ctrl: values only
            self.t0, self.t1 = t + (self.t0 - t) * factor, t + (self.t1 - t) * factor
        if not mods & Qt.KeyboardModifier.ShiftModifier:  # Shift: time only
            self.v0, self.v1 = v + (self.v0 - v) * factor, v + (self.v1 - v) * factor
        self.update()

    # --- keyboard / menu --------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if event.matches(QKeySequence.StandardKey.Delete) or key == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif key == Qt.Key.Key_Home:
            self.frame_all()
        elif key == Qt.Key.Key_F:
            self.frame_selected()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            self.selection = {(pid, k.time) for pid, c in self.visible_curves() for k in c.keys}
            self.selection_changed.emit()
            self.update()
        else:
            super().keyPressEvent(event)

    def delete_selected(self) -> None:
        if self.selection:
            self.editor.delete_keys(self.selection)
            self.selection = set()
            self.update()

    def set_handle_mode(self, mode: HandleMode) -> None:
        if self.selection:
            self.editor.set_handle_mode(self.selection, mode)

    def set_interpolation(self, interp: Interp) -> None:
        if self.selection:
            self.editor.set_interpolation(self.selection, interp)

    def _context_menu(self, global_pos) -> None:
        menu = QMenu(self)
        has_sel = bool(self.selection)
        handles = menu.addMenu("Handles")
        for mode in HandleMode:
            action = QAction(HANDLE_LABELS[mode], menu)
            action.triggered.connect(lambda _=False, m=mode: self.set_handle_mode(m))
            action.setEnabled(has_sel)
            handles.addAction(action)
        interp_menu = menu.addMenu("Interpolation")
        for interp in Interp:
            action = QAction(INTERP_LABELS[interp], menu)
            action.triggered.connect(lambda _=False, i=interp: self.set_interpolation(i))
            action.setEnabled(has_sel)
            interp_menu.addAction(action)
        menu.addSeparator()
        for text, slot, enabled in (
            ("Frame all (Home)", self.frame_all, True),
            ("Frame selected (F)", self.frame_selected, has_sel),
            ("Delete", self.delete_selected, has_sel),
        ):
            action = QAction(text, menu)
            action.triggered.connect(slot)
            action.setEnabled(enabled)
            menu.addAction(action)
        menu.exec(global_pos)


def _dist(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())


def _swatch(color: QColor) -> QIcon:
    pix = QPixmap(12, 12)
    pix.fill(color)
    return QIcon(pix)


class GraphEditor(QWidget):
    """Channel list (tick the curves to show) + toolbar + :class:`GraphView`."""

    def __init__(self, editor: KeyframeEditor, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self._hidden: set[str] = set()
        self._pids: list[str] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.view = GraphView(editor)
        self.buttons: dict[str, QToolButton] = {}

        def button(name: str, text: str, tip: str, slot, checkable: bool = False) -> None:
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            (btn.toggled if checkable else btn.clicked).connect(slot)
            bar.addWidget(btn)
            self.buttons[name] = btn

        button("frame_all", "Frame all", "Show all curves (Home)", self.view.frame_all)
        button("frame_sel", "Frame selected", "Zoom to the selected keys (F)",
               self.view.frame_selected)  # fmt: skip
        button("normalize", "Normalize", "Scale every curve to 0…1 to compare shapes",
               self.view.set_normalized, checkable=True)  # fmt: skip
        bar.addSpacing(16)
        for mode in HandleMode:
            button(f"handles_{mode.value}", HANDLE_LABELS[mode].split(" ")[0],
                   f"{HANDLE_LABELS[mode]} handles for the selected keys",
                   lambda _=False, m=mode: self.view.set_handle_mode(m))  # fmt: skip
        bar.addStretch()
        layout.addLayout(bar)
        split = QSplitter()
        self.channel_list = QListWidget()
        self.channel_list.setIconSize(QSize(12, 12))
        self.channel_list.itemChanged.connect(self._on_item_changed)
        split.addWidget(self.channel_list)
        split.addWidget(self.view)
        split.setStretchFactor(1, 1)
        split.setSizes([200, 800])
        layout.addWidget(split, 1)
        editor.changed.connect(self.refresh_channels)
        self.refresh_channels()
        self.view.frame_all()

    def refresh_channels(self) -> None:
        animation = self.editor.animation
        registry = animation.store.registry
        order = {d.id: i for i, d in enumerate(registry)}
        pids = sorted(
            (pid for pid, c in animation.curves.items() if c.keys and c.graph_kind),
            key=lambda pid: order.get(pid, 1_000_000),
        )
        if pids == self._pids:  # e.g. during drags: nothing to rebuild
            self.view.update()
            return
        self._pids = pids
        new = [pid for pid in pids if pid not in self.view.colors]
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        for i, pid in enumerate(pids):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            self.view.colors[pid] = color
            d = registry[pid]
            kind = "" if animation.curves[pid].graph_kind == "value" else "  (timing)"
            item = QListWidgetItem(_swatch(color), f"{d.section}: {d.label}{kind}")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked if pid in self._hidden else Qt.CheckState.Checked
            )
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)
        self._apply_channels()
        if new and len(new) == len(pids):  # first curves appeared: show them
            self.view.frame_all()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self._hidden.discard(pid)
        else:
            self._hidden.add(pid)
        self._apply_channels()

    def _apply_channels(self) -> None:
        self.view.channels = [
            self.channel_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.channel_list.count())
            if self.channel_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.view.update()
