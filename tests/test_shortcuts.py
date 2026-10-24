"""Keyboard: no two shortcuts collide, the F1 list names every one, Ctrl+1 .. Ctrl+0 open the
inspector categories and F6 / Shift+F6 move between the panes."""

from collections import Counter

from PyQt6.QtGui import QAction, QShortcut
from PyQt6.QtWidgets import QMessageBox

from earthling.ui.inspector import CATEGORIES
from test_ui_actions import menu_actions, window  # noqa: F401 (the fixture)


def shortcuts(window) -> list[str]:  # noqa: F811
    keys = [k.toString() for a in window.findChildren(QAction) for k in a.shortcuts()]
    keys += [s.key().toString() for s in window.findChildren(QShortcut)]
    return [k for k in keys if k]


def test_no_two_shortcuts_collide(window):  # noqa: F811
    keys = shortcuts(window)
    assert len(keys) > 30
    assert [k for k, n in Counter(keys).items() if n > 1] == []
    # the standards of such applications are there
    for key in ("Ctrl+N", "Ctrl+O", "Ctrl+S", "Ctrl+Shift+S", "Ctrl+Z", "Ctrl+Y", "Ctrl+E",
                "Ctrl+K", "Ctrl+F", "F1", "F5", "F6", "Space", "Home", "Ctrl+1"):  # fmt: skip
        assert key in keys, key


def test_the_shortcut_list_names_every_shortcut(window, monkeypatch):  # noqa: F811
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    window.show_shortcuts()
    text = shown[0]
    for path, action in menu_actions(window.menuBar()):
        for key in action.shortcuts():
            assert key.toString() in text, path
    for key in ("Space", "Delete", "Esc", "F2"):
        assert key in text


def test_ctrl_digits_open_the_categories(window):  # noqa: F811
    actions = dict(menu_actions(window.view_menu))
    for category, digit in zip(CATEGORIES, "1234567890", strict=False):
        action = actions[f"Inspector Category › {category.title}"]
        assert action.shortcut().toString() == f"Ctrl+{digit}"
        action.trigger()
        assert window.inspector.category == category.id
        assert f"Ctrl+{digit}" in window.inspector.rail.buttons[category.id].toolTip()


def test_f6_moves_between_the_panes(window):  # noqa: F811
    window.scene_list.tree.setFocus()
    seen = []
    for _ in range(5):
        window.focus_pane(1)
        seen.append(window.focusWidget())
    targets = [window.viewport, window.inspector.search,
               window.timeline_widget.dope_sheet or window.timeline_widget,
               window.scene_list.tree, window.viewport]  # fmt: skip
    assert seen == targets
    window.focus_pane(-1)
    assert window.focusWidget() is window.scene_list.tree
    window.timeline_dock.hide()  # (hidden panes are skipped)
    window.inspector.search.setFocus()
    window.focus_pane(1)
    assert window.focusWidget() is window.scene_list.tree
