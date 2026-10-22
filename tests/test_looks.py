from datetime import datetime

import pytest

from earthling.app.looks import sun_time
from earthling.core.sun import local_to_aware, solar_position


def test_sun_time_finds_noon_and_the_golden_hour():
    day = datetime(2026, 7, 1, 12, 0)
    noon = sun_time(day, 46.0, 7.0, "Europe/Zurich", None)
    assert 13 <= noon.hour <= 14  # (summer time, 7° east)
    golden = sun_time(day, 46.0, 7.0, "Europe/Zurich", 6.0)
    assert golden > noon
    elevation = solar_position(local_to_aware(golden, "Europe/Zurich"), 46.0, 7.0).elevation
    assert elevation == pytest.approx(6.0, abs=0.5)
    morning = sun_time(day, 46.0, 7.0, "Europe/Zurich", 6.0, evening=False)
    assert morning < noon


def test_looks_are_one_undo_step(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    storm = next(lk for lk in window.looks["weather"] if lk.name == "Storm")
    storm.apply()
    store = window.scene.store
    assert store["rain.enabled"] and store["weather.enabled"] and store["lightning.rate"] > 0
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    assert not store["rain.enabled"] and not store["weather.enabled"]
    night = next(lk for lk in window.looks["light"] if lk.name == "Night")
    night.apply()
    assert store["sun.datetime"].hour >= 21 or store["sun.datetime"].hour <= 3
    window.inspector.show_category("light")
    from PyQt6.QtWidgets import QToolButton

    buttons = window.inspector.scroll.widget().findChildren(QToolButton, "look")
    assert [b.text() for b in buttons] == ["Noon", "Golden hour", "Blue hour", "Night"]
    window.undo_stack.setClean()
