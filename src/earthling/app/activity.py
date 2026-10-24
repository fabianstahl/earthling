"""What is still going on in the background, for the status bar: the tiles the view still
needs (with an estimate of the time left), the labels being prepared, servers that cannot be
reached. Polled a few times a second (cheap counters), empty and hidden when all is done."""

from __future__ import annotations

import time
from collections import deque

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel

from earthling.data.net import offline_hosts


def duration_text(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


class ActivityLabel(QLabel):
    def __init__(self, viewport, parent=None) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self._history: deque[tuple[float, int]] = deque(maxlen=16)  # (time, tiles left)
        self.hide()
        self._timer = QTimer(self, interval=250)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    def tiles_left(self) -> int:
        renderer = self.viewport.renderer
        if renderer is None or renderer.terrain.nodes is None:
            return 0
        selection = renderer.terrain.last_selection
        return len(selection.request) if selection is not None else 0

    def time_left(self) -> str:
        """From the pace of the last few seconds (none while the view keeps changing)."""
        if len(self._history) < 4:
            return ""
        (t0, n0), (t1, n1) = self._history[0], self._history[-1]
        if n1 >= n0 or t1 <= t0:
            return ""
        return duration_text(n1 / ((n0 - n1) / (t1 - t0)))

    def refresh(self) -> None:
        parts = []
        tiles = self.tiles_left()
        if tiles:
            self._history.append((time.monotonic(), tiles))
            left = self.time_left()
            parts.append(f"Loading the view: {tiles} tiles" + (f", about {left}" if left else ""))
        else:
            self._history.clear()
        renderer = self.viewport.renderer
        if renderer is not None and renderer.labels.loading:
            parts.append("preparing the labels")
        hosts = offline_hosts()
        if hosts:
            parts.append(f"offline: {', '.join(hosts)} cannot be reached")
        text = " · ".join(parts)
        if text != self.text():
            self.setText(text)
            self.setToolTip("Downloads of an unreachable server pause for 30 s, then are "
                            "tried again" if hosts else "")  # fmt: skip
        self.setVisible(bool(text))
