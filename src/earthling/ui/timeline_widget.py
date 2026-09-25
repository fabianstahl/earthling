"""Timeline dock: transport controls and a scrubbable time ruler."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.app.timeline import TimelineController, format_time

FPS_CHOICES = [24.0, 25.0, 30.0, 50.0, 60.0]
TICK_STEPS = [1 / 60, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]


def nice_tick_step(pixels_per_second: float, min_spacing_px: float = 70.0) -> float:
    for step in TICK_STEPS:
        if step * pixels_per_second >= min_spacing_px:
            return step
    return TICK_STEPS[-1]


def diamond(center: QPointF, r: float) -> QPolygonF:
    return QPolygonF(
        [
            QPointF(center.x(), center.y() - r),
            QPointF(center.x() + r, center.y()),
            QPointF(center.x(), center.y() + r),
            QPointF(center.x() - r, center.y()),
        ]
    )


class TimeRuler(QWidget):
    """Time axis with the playhead; click/drag to scrub. Shows all key times as diamonds."""

    scrubbed = pyqtSignal(float)
    MARGIN = 12

    def __init__(self, controller: TimelineController, left_inset: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.left_inset = left_inset  # room for the dope sheet's label column
        self.setMinimumHeight(42)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        controller.time_changed.connect(lambda t: self.update())
        controller.settings_changed.connect(self.update)
        self._dragging = False

    # coordinate mapping
    def _span(self) -> tuple[float, float]:
        left = self.left_inset + self.MARGIN
        return left, max(1.0, self.width() - left - self.MARGIN)

    def time_to_x(self, t: float) -> float:
        left, width = self._span()
        return left + t / max(self.controller.duration, 1e-6) * width

    def x_to_time(self, x: float) -> float:
        left, width = self._span()
        return (x - left) / width * self.controller.duration

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        duration = self.controller.duration
        pps = self._span()[1] / max(duration, 1e-6)
        step = nice_tick_step(pps)
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        p.setFont(font)
        text_pen = QPen(pal.text().color())
        tick_pen = QPen(pal.mid().color())
        n = int(math.floor(duration / step + 1e-9))
        for i in range(n + 1):
            t = i * step
            x = self.time_to_x(t)
            p.setPen(tick_pen)
            p.drawLine(QPointF(x, 22), QPointF(x, self.height()))
            p.setPen(text_pen)
            label = f"{t:.0f}s" if step >= 1 else f"{t:.2f}s"
            p.drawText(QPointF(x + 3, 14), label)
        # key times (all animated properties)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(240, 190, 60))
        for t in self.controller.animation.key_times():
            p.drawPolygon(diamond(QPointF(self.time_to_x(t), self.height() - 10), 5))
        # playhead
        x = self.time_to_x(self.controller.time)
        p.setPen(QPen(QColor(90, 170, 255), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(QColor(90, 170, 255))
        p.drawRect(QRectF(x - 4, 0, 8, 6))
        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._dragging = True
        self.controller.pause()
        self.controller.set_time(self.x_to_time(event.position().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self.controller.set_time(self.x_to_time(event.position().x()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False


class TimelineWidget(QWidget):
    def __init__(self, controller: TimelineController, editor=None, parent=None) -> None:
        """``editor``: optional KeyframeEditor; shows the dope sheet when given."""
        super().__init__(parent)
        self.controller = controller
        self.editor = editor
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.buttons: dict[str, QToolButton] = {}
        for name, text, tip, slot in (
            ("start", "⏮", "Jump to start (Home)", controller.go_start),
            ("prev", "◀|", "Previous frame (,)", lambda: controller.step(-1)),
            ("play", "▶", "Play / pause (Space)", controller.toggle),
            ("next", "|▶", "Next frame (.)", lambda: controller.step(1)),
            ("end", "⏭", "Jump to end (End)", controller.go_end),
        ):
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
            self.buttons[name] = btn
        self.loop_btn = QToolButton()
        self.loop_btn.setText("⟲")
        self.loop_btn.setToolTip("Loop playback")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(controller.loop)
        self.loop_btn.toggled.connect(lambda v: setattr(controller, "loop", v))
        bar.addWidget(self.loop_btn)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(170)
        bar.addWidget(self.time_label)
        bar.addStretch()
        bar.addWidget(QLabel("Duration"))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 3600.0)
        self.duration.setDecimals(2)
        self.duration.setSuffix(" s")
        self.duration.setKeyboardTracking(False)
        self.duration.valueChanged.connect(controller.set_duration)
        bar.addWidget(self.duration)
        bar.addWidget(QLabel("FPS"))
        self.fps = QComboBox()
        for fps in FPS_CHOICES:
            self.fps.addItem(f"{fps:g}", fps)
        self.fps.currentIndexChanged.connect(lambda i: controller.set_fps(self.fps.itemData(i)))
        bar.addWidget(self.fps)
        if editor is not None:
            self.auto_key = QToolButton()
            self.auto_key.setText("● Auto-key")
            self.auto_key.setToolTip("Editing an animated property updates its key at the playhead")
            self.auto_key.setCheckable(True)
            self.auto_key.setChecked(editor.auto_key)
            self.auto_key.toggled.connect(lambda v: setattr(editor, "auto_key", v))
            bar.insertWidget(bar.count() - 5, self.auto_key)
        layout.addLayout(bar)
        from earthling.ui.dope_sheet import LABEL_WIDTH, DopeSheet

        self.ruler = TimeRuler(controller, left_inset=LABEL_WIDTH if editor is not None else 0)
        layout.addWidget(self.ruler)
        self.dope_sheet = None
        if editor is not None:
            self.dope_sheet = DopeSheet(editor, self.ruler.time_to_x, self.ruler.x_to_time)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.dope_sheet)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            layout.addWidget(scroll, 1)
            editor.changed.connect(self.ruler.update)
        controller.time_changed.connect(self._update_label)
        controller.playing_changed.connect(
            lambda playing: self.buttons["play"].setText("⏸" if playing else "▶")
        )
        controller.settings_changed.connect(self.sync_settings)
        self.sync_settings()
        self._update_label(controller.time)

    def sync_settings(self) -> None:
        for widget, value in ((self.duration, self.controller.duration),):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        index = self.fps.findData(self.controller.fps)
        self.fps.blockSignals(True)
        if index < 0:
            self.fps.addItem(f"{self.controller.fps:g}", self.controller.fps)
            index = self.fps.count() - 1
        self.fps.setCurrentIndex(index)
        self.fps.blockSignals(False)
        self._update_label(self.controller.time)

    def _update_label(self, t: float) -> None:
        c = self.controller
        self.time_label.setText(f"{format_time(t, c.fps)} / {format_time(c.duration, c.fps)}")
