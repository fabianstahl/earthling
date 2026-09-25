"""Keyframe editing with undo: adding/removing/moving keys, interpolation, copy & paste."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QUndoCommand, QUndoStack

from earthling.app.timeline import TimelineController
from earthling.core.animation import Animation, Curve, Interp, snap_to_frame

KeyRef = tuple[str, float]  # (property id, key time)


class AnimationSnapshotCommand(QUndoCommand):
    """Undo by restoring the affected curves as they were (JSON snapshots)."""

    def __init__(
        self,
        label: str,
        animation: Animation,
        before: dict[str, list],
        after: dict[str, list],
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(label)
        self.animation = animation
        self.before = before
        self.after = after
        self.on_change = on_change
        self._first = True

    def _restore(self, snapshot: dict[str, list]) -> None:
        registry = self.animation.store.registry
        for pid, data in snapshot.items():
            if data:
                self.animation.curves[pid] = Curve.from_json(registry[pid], data)
            else:
                self.animation.curves.pop(pid, None)
        self.on_change()

    def redo(self) -> None:
        if self._first:  # the edit was already applied when the command was pushed
            self._first = False
            return
        self._restore(self.after)

    def undo(self) -> None:
        self._restore(self.before)


class KeyframeEditor(QObject):
    """All keyframe edits go through here so they are undoable and refresh the views."""

    changed = pyqtSignal()  # keys added/removed/moved (views repaint)

    def __init__(self, animation: Animation, timeline: TimelineController, stack: QUndoStack):
        super().__init__()
        self.animation = animation
        self.timeline = timeline
        self.stack = stack
        self.auto_key = True  # editing an animated property updates its key
        self.clipboard: list[tuple[str, float, Any, Interp]] = []

    # --- snapshots --------------------------------------------------------------------
    def _snapshot(self, pids: Iterable[str]) -> dict[str, list]:
        out = {}
        for pid in pids:
            curve = self.animation.curves.get(pid)
            out[pid] = curve.to_json() if curve is not None else []
        return out

    def _notify(self) -> None:
        self.timeline.refresh()
        self.changed.emit()

    def edit(self, label: str, pids: Iterable[str], fn: Callable[[], None]) -> None:
        pids = list(dict.fromkeys(pids))
        before = self._snapshot(pids)
        fn()
        after = self._snapshot(pids)
        if before == after:
            return
        self.stack.push(
            AnimationSnapshotCommand(label, self.animation, before, after, self._notify)
        )
        self._notify()

    # --- operations -------------------------------------------------------------------
    @property
    def now(self) -> float:
        return snap_to_frame(self.timeline.time, self.animation.fps)

    def has_key_now(self, pid: str) -> bool:
        return self.animation.has_key(pid, self.now)

    def toggle_key(self, pid: str) -> None:
        if self.has_key_now(pid):
            self.edit("Delete key", [pid], lambda: self.animation.remove_key(pid, self.now))
        else:
            self.set_key(pid)

    def set_key(self, pid: str, value: Any = None) -> None:
        self.edit("Insert key", [pid], lambda: self.animation.set_key(pid, self.now, value))

    def on_property_edited(self, pid: str, value: Any, interactive: bool) -> bool:
        """Auto-key: returns True if the edit was turned into a key edit."""
        if not self.auto_key or not self.animation.is_animated(pid) or interactive:
            return False
        self.set_key(pid, value)
        return True

    def delete_keys(self, keys: Iterable[KeyRef]) -> None:
        keys = list(keys)

        def run() -> None:
            for pid, t in keys:
                curve = self.animation.curves.get(pid)
                if curve is not None:
                    curve.remove_key(t)
                    if not curve.keys:
                        del self.animation.curves[pid]

        self.edit("Delete keys", [pid for pid, _ in keys], run)

    def move_keys(self, keys: Iterable[KeyRef], dt: float) -> list[KeyRef]:
        """Shift keys in time (snapped to frames); keys landing on others replace them."""
        keys = list(keys)
        dt = snap_to_frame(dt, self.animation.fps)
        moved: list[KeyRef] = []

        def run() -> None:
            by_pid: dict[str, list[float]] = {}
            for pid, t in keys:
                by_pid.setdefault(pid, []).append(t)
            for pid, times in by_pid.items():
                curve = self.animation.curves.get(pid)
                if curve is None:
                    continue
                picked = [k for k in curve.keys if any(abs(k.time - t) < 1e-6 for t in times)]
                for k in picked:
                    curve.keys.remove(k)
                for k in picked:
                    k.time = max(0.0, snap_to_frame(k.time + dt, self.animation.fps))
                    existing = curve.key_at(k.time)
                    if existing is not None:
                        curve.keys.remove(existing)
                    curve.keys.append(k)
                    moved.append((pid, k.time))
                curve.sort()

        if dt != 0:
            self.edit("Move keys", [pid for pid, _ in keys], run)
            return moved
        return keys

    def set_interpolation(self, keys: Iterable[KeyRef], interp: Interp) -> None:
        keys = list(keys)

        def run() -> None:
            for pid, t in keys:
                curve = self.animation.curves.get(pid)
                key = curve.key_at(t) if curve is not None else None
                if key is not None:
                    key.interp = interp

        self.edit(f"Interpolation: {interp.value}", [pid for pid, _ in keys], run)

    def copy(self, keys: Iterable[KeyRef]) -> None:
        self.clipboard = []
        for pid, t in keys:
            curve = self.animation.curves.get(pid)
            key = curve.key_at(t) if curve is not None else None
            if key is not None:
                self.clipboard.append((pid, key.time, key.value, key.interp))

    def paste(self) -> list[KeyRef]:
        """Paste at the playhead, keeping relative timing."""
        if not self.clipboard:
            return []
        start = min(t for _, t, _, _ in self.clipboard)
        now = self.now
        pasted: list[KeyRef] = []

        def run() -> None:
            for pid, t, value, interp in self.clipboard:
                time = now + (t - start)
                self.animation.set_key(pid, time, value, interp)
                pasted.append((pid, snap_to_frame(time, self.animation.fps)))

        self.edit("Paste keys", [pid for pid, *_ in self.clipboard], run)
        return pasted
