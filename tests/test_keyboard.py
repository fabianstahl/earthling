"""Keyboard only: every dialog the menus open has the focus inside, Enter presses its own
action (not "Choose…" or "…") and Esc closes it."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QMessageBox, QPushButton

from test_ui_actions import menu_actions, window  # noqa: F401 (the fixture)


def test_every_dialog_works_by_keyboard(window, qtbot, monkeypatch):  # noqa: F811
    report = {}
    current = {}

    def check(dialog) -> int:
        dialog.show()
        qtbot.wait(20)
        focus = dialog.focusWidget()
        buttons = [b for b in dialog.findChildren(QPushButton) if b.isVisible()]
        default = [b.text() for b in buttons if b.isDefault()]
        # (a chooser may be the default while it is the only thing to do: OK still disabled)
        actions = [b.text() for b in buttons if b.isEnabled() and not b.text().endswith("…")
                   and b.text() not in ("Cancel", "Close")]  # fmt: skip
        if default and default[0].endswith("…") and not actions:
            default = ["(the only action)"]
        qtbot.keyClick(focus or dialog, Qt.Key.Key_Escape)
        qtbot.wait(10)
        report[f"{current['path']} ({type(dialog).__name__})"] = (
            focus is not None and dialog.isAncestorOf(focus),
            default,
            not dialog.isVisible(),
        )
        dialog.close()
        return QDialog.DialogCode.Rejected

    window.dialog_closer.stop()
    monkeypatch.setattr(QDialog, "exec", check)
    monkeypatch.setattr(QMessageBox, "exec", check)
    for path, action in menu_actions(window.menuBar()):
        if path in ("File › Quit", "File › Start Screen") or "Open Recent" in path:
            continue
        current["path"] = path
        action.trigger()
        qtbot.wait(10)
    assert len(report) >= 10

    # Enter presses the dialog's own action, not a button that opens another window ("…")
    def fine(focus_inside, default, closed) -> bool:
        return focus_inside and len(default) == 1 and not default[0].endswith("…") and closed

    problems = {name: r for name, r in report.items() if not fine(*r)}
    assert problems == {}, problems
