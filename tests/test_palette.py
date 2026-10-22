from PyQt6.QtCore import Qt

from earthling.ui.inspector import search_terms
from earthling.ui.palette import CommandPalette, Entry, score


def test_score_prefers_title_starts():
    terms = search_terms("fog")
    start = Entry("Settings", "Fog density", "", "x", lambda: None)
    word = Entry("Settings", "Height fog", "", "x", lambda: None)
    other = Entry("Settings", "Aerial perspective", "", "x", lambda: None, words="fog haze")
    none = Entry("Settings", "Width", "", "x", lambda: None)
    assert score(start, terms) > score(word, terms) > score(other, terms) > 0
    assert score(none, terms) == 0
    assert score(start, search_terms("nebel")) > 0  # German


def test_palette_groups_runs_and_keys(qtbot):
    ran = []
    entries = [
        Entry(
            "Settings",
            "Fog density",
            "Weather",
            "cloud-fog",
            lambda: ran.append("show"),
            key=lambda: ran.append("key"),
        ),
        Entry(
            "Actions",
            "Export video…",
            "File",
            "film-strip",
            lambda: ran.append("export"),
            "Ctrl+E",
            recent=True,
        ),
        Entry("Help", "Fog and clouds", "docs", "book-open", lambda: ran.append("help")),
    ]
    palette = CommandPalette([lambda: entries])
    qtbot.addWidget(palette)
    palette.open("")
    assert [e.title for e in palette.shown] == ["Export video…"]  # recent actions
    palette.input.setText("fog")
    assert [e.group for e in palette.shown] == ["Settings", "Help"]
    qtbot.keyClick(palette.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert ran == ["key"]
    palette.open("fog")
    qtbot.keyClick(palette.input, Qt.Key.Key_Down)
    qtbot.keyClick(palette.input, Qt.Key.Key_Return)
    assert ran[-1] == "help"
    palette.open(">exp")
    assert [e.title for e in palette.shown] == ["Export video…"]


def test_window_palette_reveals_a_setting(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show_palette("fog density")
    entry = window.command_palette.current()
    assert entry.title == "Fog density" and entry.right.endswith("/km")
    window.command_palette.run_current()
    assert window.inspector.category == "weather" and "fog.density" in window.inspector.rows
    window.inspector.reveal("weather.layer3.base")  # in a tab of a switched-off section
    assert "weather.layer3.base" in window.inspector.rows
    actions = [e.title for e in window.command_palette.entries if e.group == "Actions"]
    assert "Export Video…" in actions and "Command Palette…" in actions


def test_every_palette_entry_has_an_icon(qtbot):
    from earthling.app.main_window import MainWindow
    from earthling.ui import theme

    window = MainWindow()
    qtbot.addWidget(window)
    window.show_palette("")
    for entry in window.command_palette.entries:  # (raises for an icon that is missing)
        if isinstance(entry.icon, str):
            theme.icon(entry.icon)
    window.command_palette.set_scope("Actions")
    assert window.command_palette.shown
