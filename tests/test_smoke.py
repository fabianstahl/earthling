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
