"""Timeline state and playback (current time, play/pause, looping)."""

from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from earthling.core.animation import Animation, snap_to_frame


class TimelineController(QObject):
    """Owns the current time; applies the animation to the store whenever it changes."""

    time_changed = pyqtSignal(float)
    playing_changed = pyqtSignal(bool)
    settings_changed = pyqtSignal()  # duration / fps

    def __init__(self, animation: Animation, parent=None) -> None:
        super().__init__(parent)
        self.animation = animation
        self.time = 0.0
        self.loop = True
        self._playing = False
        self._last_tick = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(4)
        self._timer.timeout.connect(self._tick)

    # --- state ------------------------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def fps(self) -> float:
        return self.animation.fps

    @property
    def duration(self) -> float:
        return self.animation.duration

    @property
    def frame(self) -> int:
        return int(round(self.time * self.fps))

    def set_time(self, t: float, snap: bool = True) -> None:
        t = min(max(t, 0.0), self.duration)
        if snap:
            t = snap_to_frame(t, self.fps)
        self.time = t
        self.animation.apply(t)
        self.time_changed.emit(t)

    def refresh(self) -> None:
        """Re-apply the animation at the current time (after keys changed)."""
        self.set_time(self.time, snap=False)

    def set_duration(self, seconds: float) -> None:
        self.animation.duration = max(1.0 / self.fps, float(seconds))
        if self.time > self.animation.duration:
            self.set_time(self.animation.duration)
        self.settings_changed.emit()

    def set_fps(self, fps: float) -> None:
        self.animation.fps = float(fps)
        self.settings_changed.emit()
        self.set_time(self.time)

    # --- transport --------------------------------------------------------------------
    def play(self) -> None:
        if self._playing:
            return
        if self.time >= self.duration:
            self.set_time(0.0)
        self._playing = True
        self._last_tick = time.perf_counter()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.set_time(self.time)  # land on a frame
        self.playing_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def stop(self) -> None:
        self.pause()
        self.set_time(0.0)

    def step(self, frames: int) -> None:
        self.pause()
        self.set_time(self.time + frames / self.fps)

    def go_start(self) -> None:
        self.set_time(0.0)

    def go_end(self) -> None:
        self.set_time(self.duration)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        t = self.time + dt  # real time; slow renders drop frames instead of slowing down
        if t >= self.duration:
            if self.loop:
                t = t % self.duration if self.duration > 0 else 0.0
            else:
                self.time = self.duration
                self.pause()
                return
        self.time = t
        self.animation.apply(t)
        self.time_changed.emit(t)


def format_time(t: float, fps: float) -> str:
    minutes = int(t // 60)
    seconds = t - minutes * 60
    frame = int(round(t * fps))
    return f"{minutes:02d}:{seconds:05.2f}  ({frame})"
