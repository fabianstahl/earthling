from earthling.app.main_window import MainWindow
from earthling.cli import build_parser


def test_parser_defaults_to_gui():
    args = build_parser().parse_args([])
    assert args.command is None


def test_main_window_opens(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.windowTitle() == "Earthling"


def test_open_project_updates_title(qtbot, tmp_path):
    from earthling.core.config import Project

    Project.create(tmp_path / "alps")
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(tmp_path / "alps")
    assert window.windowTitle().startswith("alps")


def test_demo_project_shows_tracks(qtbot):
    from pathlib import Path

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(Path(__file__).parents[1] / "examples" / "alps_demo")
    assert len(window.tracks) == 2
    assert window.tracks_dock.tree.topLevelItemCount() == 3  # two tracks + total
