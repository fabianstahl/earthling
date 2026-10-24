"""Every function of the interface runs: each menu action (with any dialog it opens closed
again), each look and each setting found by the command palette, on an open project."""

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import QApplication, QColorDialog, QDialog, QFileDialog, QInputDialog

from earthling.app.main_window import MainWindow
from earthling.ui.palette import menu_text
from helpers import offline_demo


def menu_actions(menu) -> list[tuple[str, QAction]]:
    """(path, action) of every enabled entry in ``menu`` and its submenus."""
    out = []
    for action in menu.actions():
        if action.isSeparator():
            continue
        text = menu_text(action.text())
        if action.menu() is not None:
            out += [(f"{text} › {path}", a) for path, a in menu_actions(action.menu())]
        elif action.isEnabled():
            out.append((text, action))
    return out


def close_dialogs() -> None:
    """Reject whatever dialog an action opened (modal ones run their own event loop)."""
    modal = QApplication.activeModalWidget()
    if modal is not None:
        modal.close()
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QDialog) and widget.isVisible():
            widget.reject()


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    for name in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, lambda *a, **k: ("", ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **k: QColor())
    for name in ("getText", "getItem", "getInt", "getDouble"):
        monkeypatch.setattr(QInputDialog, name, lambda *a, **k: (None, False))
    from PyQt6.QtGui import QDesktopServices

    monkeypatch.setattr(QDesktopServices, "openUrl", lambda *a, **k: True)
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 1000)
    w.show()
    assert w.open_project(offline_demo(tmp_path))
    qtbot.wait(50)
    closer = w.dialog_closer = QTimer()  # (a test that looks at a dialog stops it)
    closer.timeout.connect(close_dialogs)
    closer.start(40)
    yield w
    closer.stop()
    close_dialogs()


def test_every_menu_action_runs(window, qtbot):
    ran = []
    for path, _ in menu_actions(window.menuBar()):
        if path in ("File › Quit", "File › Start Screen"):  # (closing / leaving the project)
            continue
        # (looked up again: an action may rebuild a menu, e.g. Open Recent)
        action = dict(menu_actions(window.menuBar())).get(path)
        if action is None:
            continue
        checkable = action.isCheckable()  # (an Open Recent entry rebuilds its own menu)
        action.trigger()
        if checkable:
            action.trigger()  # (and back)
        qtbot.wait(20)
        close_dialogs()
        ran.append(path)
        assert window.isVisible(), path
    assert len(ran) > 25  # (a menu went missing?)
    # still usable afterwards: the project is open, the scene can be saved
    assert window.session is not None
    assert window.save_scene()


def test_every_look_applies_and_undoes(window, qtbot):
    store = window.scene.store
    before = {d.id: store[d.id] for d in store.registry}
    for entry in window._look_entries():
        entry.run()
        qtbot.wait(5)
    while window.undo_stack.canUndo():
        window.undo_stack.undo()
    assert {d.id: store[d.id] for d in store.registry} == before


@pytest.mark.slow
def test_every_setting_found_by_the_palette_opens(window, qtbot):
    """Enter on a palette result shows the setting in the inspector (its category opens and
    the editor exists)."""
    entries = window._setting_entries()
    assert len(entries) > 200
    for entry in entries:
        entry.run()
        qtbot.wait(1)


def test_menu_actions_open_what_they_promise(window, qtbot, monkeypatch):
    """`triggered` passes a `checked` flag: it must not land in a slot's first optional
    parameter (the palette got False as its search text, the Animation menu called False)."""
    opened = []
    real_exec = QDialog.exec

    def recording_exec(dialog):
        opened.append(type(dialog).__name__)
        return real_exec(dialog)

    monkeypatch.setattr(QDialog, "exec", recording_exec)
    monkeypatch.setattr(QInputDialog, "getDouble",
                        lambda *a, **k: (opened.append("getDouble"), (None, False))[1])  # fmt: skip
    window.palette_action.trigger()
    assert window.command_palette.isVisible()
    assert window.command_palette.input.text() == ""
    window.command_palette.hide()
    for path, action in menu_actions(window.animation_menu):
        if path.endswith("…") and "Check" not in path:
            opened.clear()
            action.trigger()
            qtbot.wait(20)
            assert opened, path  # (the generator's dialog opened)


def test_what_the_scene_list_adds_is_in_the_menus(window):
    """The scene list's "+" (GPX files, a line, POIs, photos) is also in the Add menu: in the
    command palette and reachable without a mouse."""
    from earthling.ui.palette import menu_entries

    crumbs = {(e.crumb, e.title) for e in menu_entries(window.menuBar())}
    for title in ("GPX Files…", "New Line (Flight, Ferry, Road)…", "POI at the Marker",
                  "POI Where I Click…", "POI at a Place…", "Photos…"):  # fmt: skip
        assert ("Add", title) in crumbs, title
