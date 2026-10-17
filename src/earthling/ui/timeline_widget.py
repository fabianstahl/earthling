"""Timeline dock: transport controls and a scrubbable time ruler."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFont, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.app.timeline import TimelineController, format_time
from earthling.ui.key_views import KEY_COLOR, PLAYHEAD_COLOR, diamond
from earthling.ui.widgets import double_spin, tool_button

FPS_CHOICES = [24.0, 25.0, 30.0, 50.0, 60.0]
TICK_STEPS = [1 / 60, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]


def nice_tick_step(pixels_per_second: float, min_spacing_px: float = 70.0) -> float:
    for step in TICK_STEPS:
        if step * pixels_per_second >= min_spacing_px:
            return step
    return TICK_STEPS[-1]


def tick_label(t: float, step: float) -> str:
    return f"{t:.0f}s" if step >= 1 else f"{t:.2f}s"


class TimeRuler(QWidget):
    """Time axis with the playhead; click/drag to scrub. Shows all key times as diamonds."""

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
            p.drawText(QPointF(x + 3, 14), tick_label(t, step))
        # key times (all animated properties)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(KEY_COLOR)
        for t in self.controller.animation.key_times():
            p.drawPolygon(diamond(QPointF(self.time_to_x(t), self.height() - 10), 5))
        # playhead
        x = self.time_to_x(self.controller.time)
        p.setPen(QPen(PLAYHEAD_COLOR, 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(PLAYHEAD_COLOR)
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
    """Transport bar over the ruler; with a KeyframeEditor also the dope sheet and graph editor."""

    def __init__(self, controller: TimelineController, editor=None, parent=None) -> None:
        """``editor``: optional KeyframeEditor; shows the dope sheet when given."""
        super().__init__(parent)
        self.controller = controller
        self.editor = editor
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(self._build_bar())
        self._build_views(layout)
        controller.time_changed.connect(self._update_label)
        controller.playing_changed.connect(
            lambda playing: self.buttons["play"].setText("⏸" if playing else "▶")
        )
        controller.settings_changed.connect(self.sync_settings)
        self.sync_settings()
        self._update_label(controller.time)

    def _build_bar(self) -> QHBoxLayout:
        """Transport buttons, time, (auto-key), duration and fps."""
        controller, editor = self.controller, self.editor
        bar = QHBoxLayout()
        self.buttons: dict[str, QToolButton] = {}
        for name, text, tip, slot in (
            ("start", "⏮", "Jump to start (Ctrl+Home)", controller.go_start),
            ("prev", "◀|", "Previous frame (,)", lambda: controller.step(-1)),
            ("play", "▶", "Play / pause (Space)", controller.toggle),
            ("next", "|▶", "Next frame (.)", lambda: controller.step(1)),
            ("end", "⏭", "Jump to end (Ctrl+End)", controller.go_end),
        ):
            self.buttons[name] = tool_button(text, tip, slot)
            bar.addWidget(self.buttons[name])
        self.loop_btn = tool_button(
            "⟲", "Loop playback", lambda v: setattr(controller, "loop", v), controller.loop
        )
        bar.addWidget(self.loop_btn)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(170)
        bar.addWidget(self.time_label)
        bar.addStretch()
        bar.addWidget(QLabel("Duration"))
        self.duration = double_spin(0.1, 3600.0, decimals=2, suffix=" s", tracking=False)
        self.duration.valueChanged.connect(controller.set_duration)
        bar.addWidget(self.duration)
        bar.addWidget(QLabel("FPS"))
        self.fps = QComboBox()
        for fps in FPS_CHOICES:
            self.fps.addItem(f"{fps:g}", fps)
        self.fps.currentIndexChanged.connect(lambda i: controller.set_fps(self.fps.itemData(i)))
        bar.addWidget(self.fps)
        if editor is not None:  # added last, shown before the stretch
            self.auto_key = tool_button(
                "● Auto-key", "Editing an animated property updates its key at the playhead",
                lambda v: setattr(editor, "auto_key", v), editor.auto_key,
            )  # fmt: skip
            bar.insertWidget(bar.count() - 5, self.auto_key)
        return bar

    def _build_views(self, layout: QVBoxLayout) -> None:
        """The ruler alone, or tabs with ruler + dope sheet and the graph editor."""
        from earthling.ui.dope_sheet import LABEL_WIDTH, DopeSheet

        editor = self.editor
        self.ruler = TimeRuler(self.controller, left_inset=LABEL_WIDTH if editor is not None else 0)
        self.dope_sheet = None
        self.graph_editor = None
        self.tabs = None
        if editor is None:
            layout.addWidget(self.ruler)
            return
        from earthling.ui.graph_editor import GraphEditor

        self.dope_sheet = DopeSheet(editor, self.ruler.time_to_x, self.ruler.x_to_time)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.dope_sheet)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self.ruler)
        page_layout.addWidget(scroll, 1)
        self.graph_editor = GraphEditor(editor)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(page, "Dope Sheet")
        self.tabs.addTab(self.graph_editor, "Graph Editor")
        layout.addWidget(self.tabs, 1)
        editor.changed.connect(self.ruler.update)

    def sync_settings(self) -> None:
        self.duration.blockSignals(True)
        self.duration.setValue(self.controller.duration)
        self.duration.blockSignals(False)
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
