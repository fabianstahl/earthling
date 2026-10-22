from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_start_screen_until_a_project_opens(qtbot, tmp_path):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show_start(True)
    assert window.central.currentWidget() is window.start_screen
    assert window.toolbar.isHidden() and window.parameters_dock.isHidden()
    assert window.new_project_from_gpx([FIXTURES / "two_tracks.gpx"], tmp_path / "new")
    assert window.central.currentWidget() is window.viewport
    assert not window.parameters_dock.isHidden() and not window.toolbar.isHidden()
    assert (tmp_path / "new" / "earthling.toml").exists() and len(window.tracks) == 2
    window.show_start(True)
    assert str(tmp_path / "new") in [str(Path(p)) for p in window._recent()]
    window.start_screen.continue_requested.emit()
    assert window.central.currentWidget() is window.viewport
    window.undo_stack.setClean()
