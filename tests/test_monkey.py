"""A monkey in the interface (slow suite): thousands of random steps a user could take -
menu actions, buttons, keys, the scene list, the inspector, the palette, value drags, the
timeline, undo and redo - on an open project. Nothing may raise, crash or freeze the window
for long; the scene stays saveable. The steps come from a fixed seed: a failure repeats."""

import json
import random

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractButton

from earthling.ui.inspector import CATEGORIES
from earthling.ui.property_panel import DragField
from helpers import longest_freeze, menu_choices
from test_ui_actions import close_dialogs, menu_actions, window  # noqa: F401 (the fixture)

pytestmark = pytest.mark.slow
STEPS = 1500
NEVER = ("Quit", "Start Screen", "Open Recent")  # (ends the session under test)
KEYS = [Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Return,
        Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Delete, Qt.Key.Key_Space, Qt.Key.Key_F2,
        Qt.Key.Key_A, Qt.Key.Key_K, Qt.Key.Key_C, Qt.Key.Key_T, Qt.Key.Key_W, Qt.Key.Key_5,
        Qt.Key.Key_Home, Qt.Key.Key_Comma, Qt.Key.Key_Period, Qt.Key.Key_F6]  # fmt: skip
MODS = [Qt.KeyboardModifier.NoModifier] * 4 + [Qt.KeyboardModifier.ShiftModifier,
                                               Qt.KeyboardModifier.ControlModifier,
                                               Qt.KeyboardModifier.AltModifier]  # fmt: skip


def visible(widgets):
    return [w for w in widgets if w.isVisible() and w.isEnabled()]


def step(window, qtbot, rng: random.Random) -> str:  # noqa: F811
    """One random step; returns what it did (for the failure message)."""
    kind = rng.choice(["action", "button", "key", "row", "category", "palette", "drag",
                       "time", "play", "undo", "redo"])  # fmt: skip
    if kind == "action":
        actions = [(p, a) for p, a in menu_actions(window.menuBar())
                   if not any(n in p for n in NEVER)]  # fmt: skip
        path, action = rng.choice(actions)
        action.trigger()
        return f"menu {path}"
    if kind == "button":
        buttons = [b for b in visible(window.findChildren(QAbstractButton))
                   if "Quit" not in b.toolTip()]  # fmt: skip
        button = rng.choice(buttons)
        button.click()
        return f"button {button.text() or button.toolTip()[:40]!r}"
    if kind == "key":
        target = window.focusWidget() or window.viewport
        key, mods = rng.choice(KEYS), rng.choice(MODS)
        if key == Qt.Key.Key_Q or (
            mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_W
        ):
            return "skipped"
        qtbot.keyClick(target, key, mods)
        return f"key {key.name}+{mods.name} on {type(target).__name__}"
    if kind == "row":
        tree = window.scene_list.tree
        items = []
        for i in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(i)
            items += [top] + [top.child(j) for j in range(top.childCount())]
        if items:
            tree.setCurrentItem(rng.choice(items))
        return "scene list row"
    if kind == "category":
        window.inspector.show_category(rng.choice(CATEGORIES).id)
        return "inspector category"
    if kind == "palette":
        window.show_palette(rng.choice(["fog", "sun", "key", "track", "label", "export", ""]))
        entries = [e for e in window.command_palette.shown if "Quit" not in e.title
                   and "Start Screen" not in e.title]  # fmt: skip
        window.command_palette.hide()
        if entries:
            rng.choice(entries).run()
        return "palette entry"
    if kind == "drag":
        fields = visible(window.findChildren(DragField))
        if fields:
            field = rng.choice(fields)
            field.commit(field.value + rng.uniform(-1.0, 1.0) * (field.step or 1.0) * 5)
        return "value drag"
    if kind == "time":
        window.timeline.set_time(rng.uniform(0.0, window.timeline.duration))
        return "timeline time"
    if kind == "play":
        window.timeline.toggle()
        return "play / pause"
    stack = window.undo_stack
    (stack.undo if kind == "undo" else stack.redo)()
    return kind


@pytest.mark.parametrize("seed", [2026])
def test_a_monkey_in_the_interface(window, qtbot, monkeypatch, seed):  # noqa: F811
    menu_choices(monkeypatch)  # (context menus close at once)
    rng = random.Random(seed)
    done = []
    worst = (0.0, "")
    for i in range(STEPS):
        what = []
        try:
            frozen = longest_freeze(qtbot, lambda w=what: w.append(step(window, qtbot, rng)))
        except Exception as exc:  # (which steps led there)
            raise AssertionError(f"step {i}, after: {done[-15:]}") from exc
        close_dialogs()
        done.append(what[0] if what else "?")
        if frozen > worst[0]:
            worst = (frozen, done[-1])
        assert window.isVisible(), done[-20:]
        if i % 100 == 99:
            json.dumps(window.scene.to_json(), default=str)  # (still saveable)
            while window.undo_stack.canUndo():
                window.undo_stack.undo()
    window.timeline.pause()
    print(f"\n{STEPS} steps, the longest freeze {worst[0]:.2f} s: {worst[1]}")
    assert worst[0] < 5.0, worst
