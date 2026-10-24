"""How long the interface freezes during everyday work (slow suite): opening a project, the
inspector categories, the command palette, the scene list, the timeline, the dialogs. The
budgets are generous (machines differ); a freeze above them is a bug to find."""

import time

import pytest
from PyQt6.QtWidgets import QApplication

from earthling.app.main_window import MainWindow
from earthling.ui.inspector import CATEGORIES
from helpers import longest_freeze, offline_demo
from test_ui_actions import close_dialogs

pytestmark = pytest.mark.slow
REPORT: dict[str, float] = {}
# seconds the window may freeze at most (generous: machines differ; measured on the reference
# machine: open 0.3, first category 0.25, kept category 0.02, keystroke 0.04, the rest < 0.12)
BUDGET = {
    "open the demo project": 2.0,
    "show an inspector category the first time (worst)": 0.8,
    "show an inspector category again (worst)": 0.1,
    "type in the palette (worst keystroke)": 0.15,
    "select in the scene list (worst)": 0.3,
    "scrub the timeline (worst step)": 0.15,
    "edit a value (worst step)": 0.15,
    "open the export dialog": 0.5,
    "open the project settings": 0.5,
    "open the download dialog": 0.5,
}


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 1000)
    w.show()
    qtbot.wait(50)
    yield w
    print("\nlongest freezes (ms):")
    for name, seconds in REPORT.items():
        print(f"  {seconds * 1000:7.0f}  {name}")
    REPORT.clear()


def measure(name, qtbot, action, until=None, timeout=20.0) -> float:
    REPORT[name] = longest_freeze(qtbot, action, until, timeout)
    return REPORT[name]


def time_to_dialog(open_dialog) -> float:
    """Seconds until the modal dialog ``open_dialog()`` opens is on screen (then closed)."""
    from PyQt6.QtCore import QTimer

    t0 = time.perf_counter()
    opened: list[float] = []

    def note() -> None:
        if QApplication.activeModalWidget() is not None and not opened:
            opened.append(time.perf_counter() - t0)
            close_dialogs()

    poll = QTimer()
    poll.timeout.connect(note)
    poll.start(5)
    open_dialog()
    poll.stop()
    close_dialogs()
    return opened[0] if opened else float("inf")


def test_everyday_work_does_not_freeze(window, qtbot, tmp_path):
    project = offline_demo(tmp_path)
    measure("open the demo project", qtbot, lambda: window.open_project(project),
            until=lambda: window.session is not None)  # fmt: skip
    qtbot.wait(300)
    for label in ("the first time", "again"):
        worst = 0.0
        for category in CATEGORIES:
            show = window.inspector.show_category
            worst = max(worst, longest_freeze(qtbot, lambda c=category, show=show: show(c.id)))
        REPORT[f"show an inspector category {label} (worst)"] = worst
    window.show_palette()
    palette = window.command_palette
    worst = 0.0
    for k in range(1, 8):
        worst = max(worst, longest_freeze(qtbot, lambda k=k: palette.input.setText("density"[:k])))
    REPORT["type in the palette (worst keystroke)"] = worst
    palette.hide()
    tree = window.scene_list.tree
    worst = 0.0
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        worst = max(worst, longest_freeze(qtbot, lambda it=item: tree.setCurrentItem(it)))
    REPORT["select in the scene list (worst)"] = worst
    timeline = window.timeline
    worst = 0.0
    for k in range(20):
        worst = max(worst, longest_freeze(qtbot, lambda k=k: timeline.set_time(k * 0.7)))
    REPORT["scrub the timeline (worst step)"] = worst
    window.inspector.show_category("weather")
    worst = 0.0
    for k in range(20):
        edit = window.set_property
        worst = max(
            worst, longest_freeze(qtbot, lambda k=k, edit=edit: edit("fog.density", 0.1 * k, True))
        )
    REPORT["edit a value (worst step)"] = worst
    for name, open_dialog in (("export dialog", window.export_video_dialog),
                              ("project settings", window.edit_settings),
                              ("download dialog", window._show_download)):  # fmt: skip
        REPORT[f"open the {name}"] = time_to_dialog(open_dialog)
    over = {name: seconds for name, seconds in REPORT.items() if seconds > BUDGET[name]}
    assert not over, over
