from datetime import datetime

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from earthling.app.timeline import TimelineController, format_time
from earthling.core.scene import Scene
from earthling.ui.timeline_widget import TimelineWidget, nice_tick_step


def make(qapp):
    scene = Scene()
    anim = scene.animation
    anim.duration, anim.fps = 10.0, 25.0
    anim.set_key("sun.datetime", 0.0, datetime(2026, 7, 1, 12, 0))
    anim.set_key("sun.datetime", 10.0, datetime(2026, 7, 1, 22, 0))
    return scene, TimelineController(anim)


def test_set_time_snaps_and_applies(qapp):
    scene, tl = make(qapp)
    seen = []
    tl.time_changed.connect(seen.append)
    tl.set_time(5.013)
    assert tl.time == pytest.approx(5.0) and tl.frame == 125
    assert scene.store["sun.datetime"] == datetime(2026, 7, 1, 17, 0)
    tl.set_time(99)
    assert tl.time == 10.0 and seen[-1] == 10.0


def test_playback_uses_real_time_and_loops(qapp, monkeypatch):
    import earthling.app.timeline as timeline_module

    scene, tl = make(qapp)
    clock = [100.0]
    monkeypatch.setattr(timeline_module.time, "perf_counter", lambda: clock[0])
    tl.play()
    assert tl.playing
    clock[0] += 2.5
    tl._tick()
    assert tl.time == pytest.approx(2.5)
    clock[0] += 9.0  # past the end -> wraps around
    tl._tick()
    assert tl.time == pytest.approx(1.5)
    tl.loop = False
    clock[0] += 20.0
    tl._tick()
    assert not tl.playing and tl.time == 10.0
    tl.step(-2)
    assert tl.time == pytest.approx(10.0 - 2 / 25)


def test_settings(qapp):
    _, tl = make(qapp)
    tl.set_time(8.0)
    tl.set_duration(5.0)
    assert tl.duration == 5.0 and tl.time == 5.0
    tl.set_fps(30.0)
    assert tl.fps == 30.0
    assert format_time(65.5, 30) == "01:05.50  (1965)"
    assert nice_tick_step(100.0) == 1 and nice_tick_step(5.0) == 15


def test_ruler_scrubbing(qtbot):
    _, tl = make(None)
    widget = TimelineWidget(tl)
    qtbot.addWidget(widget)
    widget.resize(600, 120)
    widget.show()
    ruler = widget.ruler
    x = ruler.time_to_x(7.0)
    assert ruler.x_to_time(x) == pytest.approx(7.0)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(x, 10), QPointF(x, 10),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )  # fmt: skip
    ruler.mousePressEvent(event)
    assert tl.time == pytest.approx(7.0, abs=0.05)
    assert "00:07" in widget.time_label.text()
    widget.buttons["play"].click()
    assert tl.playing and widget.buttons["play"].text() == "⏸"
    tl.pause()


def test_window_has_timeline_and_scene_stays_clean_during_playback(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.scene.animation.set_key("view.fov", 0.0, 30.0)
    window.scene.animation.set_key("view.fov", 1.0, 60.0)
    window.timeline.set_time(0.5)
    assert window.scene.store["view.fov"] == pytest.approx(45.0)
    assert not window.scene.dirty
