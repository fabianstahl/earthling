"""Undo/redo commands for scene edits."""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QUndoCommand, QUndoStack

from earthling.core.properties import PropertyStore

SET_PROPERTY_ID = 1000


class SetPropertyCommand(QUndoCommand):
    """Sets one property. Consecutive interactive edits (slider drags) merge into one command."""

    def __init__(
        self, store: PropertyStore, pid: str, old: Any, new: Any, interactive: bool = False
    ) -> None:
        label = store.registry[pid].label
        super().__init__(f"Change {label}")
        self.store = store
        self.pid = pid
        self.old = old
        self.new = new
        self.open = interactive  # still accepting merges from the ongoing drag

    def id(self) -> int:
        return SET_PROPERTY_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, SetPropertyCommand) or other.pid != self.pid or not self.open:
            return False
        self.new = other.new
        self.open = other.open
        return True

    def redo(self) -> None:
        self.store.set(self.pid, self.new)

    def undo(self) -> None:
        self.store.set(self.pid, self.old)


def set_property(
    stack: QUndoStack, store: PropertyStore, pid: str, value: Any, interactive: bool = False
) -> None:
    """Undoable property change (no-op if the value does not change)."""
    new = store.registry[pid].coerce(value)
    old = store[pid]
    top = stack.command(stack.index() - 1) if stack.index() > 0 else None
    dragging = isinstance(top, SetPropertyCommand) and top.pid == pid and top.open
    if new == old and not (dragging and not interactive):
        return
    stack.push(SetPropertyCommand(store, pid, old, new, interactive))
