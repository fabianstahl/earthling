import datetime as dt

import numpy as np
import pytest

from earthling import scenekit as sk
from test_scenekit import journey  # noqa: F401 (fixture: the demo journey on flat terrain)


@pytest.fixture
def scene(journey):  # noqa: F811
    scene = sk.new_scene(journey, 20.0)
    scene.store.set("progress.mode", "time")
    return scene


def test_draw_keys(journey, scene):  # noqa: F811
    d0, d1 = journey.path.track_range(1)
    head_of = sk.draw_keys(scene, journey, d0, d1, 2.0, 12.0, realtime=False, soft=False)
    assert head_of(2.0) == pytest.approx(d0) and head_of(7.0) == pytest.approx((d0 + d1) / 2)
    curve = scene.animation.curves["progress.head"]
    assert curve.keys[0].time == 2.0 and curve.keys[-1].time == pytest.approx(12.0)
    reached = sk.head_distance_of(scene, journey.path)
    assert reached(7.0) == pytest.approx((d0 + d1) / 2, rel=1e-3)  # (the keys say the same)
    timed = sk.draw_keys(scene, journey, d0, d1, 2.0, 12.0, realtime=True)
    assert timed(2.0) == pytest.approx(d0, abs=1.0) and timed(12.0) == pytest.approx(d1, abs=1.0)


def test_sun_keys(journey, scene):  # noqa: F811
    d0, d1 = journey.path.track_range(0)
    head_of = sk.draw_keys(scene, journey, d0, d1, 0.0, 10.0, realtime=False)
    sk.sun_keys(scene, journey, np.arange(0.0, 10.01, 0.5), head_of, earliest=10.0, latest=14.0)
    keys = scene.animation.curves["sun.datetime"].keys
    assert not scene.store["sun.follow_track"] and len(keys) == 21
    hours = [k.value.hour + k.value.minute / 60 for k in keys]
    assert min(hours) >= 10.0 and max(hours) <= 14.0
    assert isinstance(keys[0].value, dt.datetime)


def test_orbit_and_line_shots(journey):  # noqa: F811
    times = np.arange(0.0, 4.01, 1.0 / 30)
    centre = journey.start(1)
    shots = sk.orbit_shots(journey, times, lambda t: centre, distance=1000.0, pitch=-30.0,
                           heading=0.0, turn=90.0)  # fmt: skip
    assert np.linalg.norm((shots[0][1] - centre)[:2]) == pytest.approx(1000 * np.cos(np.pi / 6))
    assert sk.heading_of(shots[0][1], centre) == pytest.approx(0.0, abs=1e-6)
    assert sk.heading_of(shots[-1][1], centre) == pytest.approx(90.0, abs=1e-6)
    d0, d1 = journey.path.track_range(0)
    line = sk.line_shots(journey, times, lambda t: d0 + (d1 - d0) * t / 4.0, near=1000.0,
                         far=8000.0, pitch=-40.0, side=90.0)  # fmt: skip
    gaps = [np.linalg.norm(e - c) for _, e, c in line]
    assert gaps[0] == pytest.approx(1000.0, rel=0.05) and max(gaps) > 6000.0


def test_dialogs_generate(qtbot, journey, scene):  # noqa: F811
    from earthling.app.generators import CameraDialog, DrawDialog, SunDialog

    draw = DrawDialog(20.0, journey.path.tracks)
    qtbot.addWidget(draw)
    draw.t0.setValue(1.0)
    draw.t1.setValue(19.0)
    draw.generate(journey, scene)
    assert scene.animation.is_animated("progress.head")
    for kind in ("follow", "map", "orbit", "line"):
        camera = CameraDialog(20.0)
        qtbot.addWidget(camera)
        camera.kind.setCurrentIndex(camera.kind.findData(kind))
        camera.generate(journey, scene)
        keys = scene.animation.curves["camera.pose"].keys
        assert keys[0].time == 0.0 and keys[-1].time == pytest.approx(20.0), kind
    sun = SunDialog(20.0)
    qtbot.addWidget(sun)
    sun.smooth.setChecked(True)
    sun.generate(journey, scene)
    assert scene.animation.is_animated("sun.datetime")


def test_generators_are_one_undo_step(qtbot):
    from earthling.app import generators
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    anim = window.scene.animation
    anim.duration = 10.0
    anim.set_key("camera.pose", 0.0, (0.0, 0.0, 1000.0, 0.0, -30.0, 0.0))
    anim.set_key("camera.pose", 10.0, (500.0, 0.0, 1000.0, 0.0, -30.0, 0.0))
    assert generators.run(
        window, "Add still seconds", lambda _j, scene: sk.add_holds(scene, 1.0), needs_tracks=False
    )
    assert anim.duration == 12.0
    assert not generators.run(window, "Draw", lambda _j, _s: None)  # (no tracks)
    window.undo_stack.undo()
    assert anim.duration == 10.0 and len(anim.curves["camera.pose"].keys) == 2
    window.undo_stack.redo()
    assert anim.duration == 12.0


def test_scene_frame_origin_with_undo(qtbot):
    from earthling.app.main_window import MainWindow
    from helpers import DEMO

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(DEMO)
    pose = (1000.0, -3000.0, 2500.0, 20.0, -25.0, 0.0)
    window.scene.animation.set_key("camera.pose", 0.0, pose)
    window.set_scene_origin((46.2, 7.1))
    assert window.scene.origin == (46.2, 7.1) and window.session.frame.lat == 46.2
    moved = window.scene.animation.curves["camera.pose"].keys[0].value
    assert moved[0] != pytest.approx(pose[0])
    window.undo_stack.undo()
    assert window.scene.origin is None
    assert window.scene.animation.curves["camera.pose"].keys[0].value == pytest.approx(
        pose, abs=1e-4
    )
