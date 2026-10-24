"""The 3D viewport as the user drives it: orbit drags, panning, the wheel, the fly keys, Tab,
boost, picking a point and the on-demand repaint; data set before and after the GL context."""

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from earthling.app.viewport import Viewport, unproject_log_depth
from earthling.core.scene import Scene
from earthling.render.camera import Camera, FlyController
from helpers import FRAME

L, R, M = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton
NONE = Qt.KeyboardModifier.NoModifier
SHIFT, CTRL = Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.ControlModifier


def press(widget, pos, button=L, mods=NONE, kind=QEvent.Type.MouseButtonPress):
    point = QPointF(*pos)
    QApplication.sendEvent(widget, QMouseEvent(kind, point, point, button, button, mods))


def move(widget, pos, buttons=L, mods=NONE):
    point = QPointF(*pos)
    event = QMouseEvent(QEvent.Type.MouseMove, point, point, Qt.MouseButton.NoButton, buttons,
                        mods)  # fmt: skip
    QApplication.sendEvent(widget, event)


def release(widget, pos, button=L):
    press(widget, pos, button, kind=QEvent.Type.MouseButtonRelease)


def drag(widget, start, end, button=L):
    press(widget, start, button)
    move(widget, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2), button)
    move(widget, end, button)
    release(widget, end, button)


def key(widget, k, mods=NONE, kind=QEvent.Type.KeyPress, auto_repeat=False):
    QApplication.sendEvent(widget, QKeyEvent(kind, k, mods, "", auto_repeat))


def wheel(widget, steps, mods=NONE):
    pos = QPointF(100, 100)
    event = QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, int(120 * steps)),
                        Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase,
                        False)  # fmt: skip
    QApplication.sendEvent(widget, event)


@pytest.fixture
def view(qtbot):
    """A viewport that is never shown: no GL context, the input handling alone."""
    v = Viewport()
    qtbot.addWidget(v)
    v.resize(400, 300)
    v._timer.stop()  # (the tests tick by hand)
    return v


def test_orbit_drag_rotates_right_and_middle_drag_pan(view):
    orbit = view.orbit
    heading, pitch = orbit.heading, orbit.pitch
    drag(view, (100, 100), (160, 130))
    assert orbit.heading != heading and orbit.pitch != pitch
    assert view.camera.heading == pytest.approx(orbit.heading)  # the camera follows
    target = orbit.target.copy()
    drag(view, (100, 100), (150, 100), R)
    panned = orbit.target.copy()
    assert not np.allclose(panned, target)
    drag(view, (100, 100), (100, 160), M)
    assert not np.allclose(orbit.target, panned)
    # moving without a pressed button does nothing
    before = orbit.target.copy()
    move(view, (10, 10), Qt.MouseButton.NoButton)
    assert np.allclose(orbit.target, before)


def test_wheel_zooms_the_orbit_and_sets_the_fly_speed(view):
    distance = view.orbit.distance
    wheel(view, 2)
    assert view.orbit.distance < distance
    wheel(view, -4)
    assert view.orbit.distance > distance
    view.set_mode(view.FLY)
    wheel(view, 1)
    assert view.fly.speed_multiplier == pytest.approx(1.25)
    wheel(view, 100)
    assert view.fly.speed_multiplier == 50.0  # clamped
    wheel(view, -1000)
    assert view.fly.speed_multiplier == 0.05


def test_fly_keys_switch_to_fly_mode_and_steer(view):
    modes = []
    view.mode_changed.connect(modes.append)
    key(view, Qt.Key.Key_W)
    assert view.mode == view.FLY and modes == [view.FLY]
    assert view.fly.pressed == {FlyController.FORWARD}
    for k, direction in ((Qt.Key.Key_A, FlyController.LEFT), (Qt.Key.Key_Down, FlyController.BACK),
                         (Qt.Key.Key_PageUp, FlyController.UP),
                         (Qt.Key.Key_Q, FlyController.DOWN)):  # fmt: skip
        key(view, k)
        assert direction in view.fly.pressed
    key(view, Qt.Key.Key_W, kind=QEvent.Type.KeyRelease, auto_repeat=True)  # (repeat: ignored)
    assert FlyController.FORWARD in view.fly.pressed
    key(view, Qt.Key.Key_W, kind=QEvent.Type.KeyRelease)
    assert FlyController.FORWARD not in view.fly.pressed
    QApplication.sendEvent(view, QEvent(QEvent.Type.FocusOut))  # (leaving: nothing keeps flying)
    assert view.fly.pressed == set()
    # Shift boosts, Ctrl slows down, both only while held
    key(view, Qt.Key.Key_D, SHIFT)
    assert view.fly.boost == 4.0
    key(view, Qt.Key.Key_D, CTRL)
    assert view.fly.boost == 0.25
    key(view, Qt.Key.Key_D, kind=QEvent.Type.KeyRelease)
    assert view.fly.boost == 1.0
    # in fly mode a drag looks around (left or right button)
    heading = view.camera.heading
    drag(view, (100, 100), (140, 100), R)
    assert view.camera.heading == pytest.approx((heading + 40 * view.fly.look_sensitivity) % 360)


def test_tab_switches_between_orbit_and_fly(view):
    assert view.mode == view.ORBIT
    key(view, Qt.Key.Key_Tab)  # (Tab does not move the focus away)
    assert view.mode == view.FLY
    view.fly.pressed.add(FlyController.FORWARD)
    key(view, Qt.Key.Key_Tab)
    assert view.mode == view.ORBIT and view.fly.pressed == set()
    # without a picked surface the orbit centre is 2 km ahead
    assert view.orbit.distance == pytest.approx(2000.0)
    view.set_mode(view.ORBIT)  # (no change, no signal)
    assert view.mode == view.ORBIT


def test_manual_navigation_leaves_the_animated_camera(view):
    scene = Scene()
    view.set_store(scene.store)
    pose = (100.0, 200.0, 3000.0, 45.0, -20.0, 0.0)
    scene.store.set("camera.pose", pose)
    seen = []
    view.animated_camera_changed.connect(seen.append)
    view.set_animated_camera(True)
    assert view.camera_pose() == pytest.approx(pose) and seen == [True]
    # a key of the camera animation moves the camera along
    scene.store.set("camera.pose", (0.0, 0.0, 2000.0, 90.0, -10.0, 0.0))
    assert view.camera.heading == 90.0
    press(view, (50, 50))  # grabbing the view: back to the free camera (fly)
    release(view, (50, 50))
    assert not view.animated_camera and seen == [True, False] and view.mode == view.FLY
    view.set_animated_camera(True)
    wheel(view, 1)  # the wheel too
    assert not view.animated_camera
    view.set_animated_camera(True)
    key(view, Qt.Key.Key_S)  # and the fly keys
    assert not view.animated_camera


def test_camera_state_round_trip_and_bad_states(view):
    view.camera.position = np.array([1.0, 2.0, 3.0])
    view.camera.heading, view.camera.pitch = 33.0, -12.0
    view.set_mode(view.FLY)
    state = view.camera_state()
    other = Viewport()
    other._timer.stop()
    assert other.restore_camera_state(state)
    assert other.mode == other.FLY and other.camera_pose() == pytest.approx(view.camera_pose())
    view.set_mode(view.ORBIT)
    orbit_state = view.camera_state()
    assert other.restore_camera_state(orbit_state) and other.mode == other.ORBIT
    assert other.orbit.target == pytest.approx(view.orbit.target)
    assert not other.restore_camera_state({"position": [1, 2]})  # (no heading)
    assert not other.restore_camera_state({"position": "x", "heading": 0, "pitch": 0})
    other.deleteLater()


def test_go_to_and_camera_text_need_a_frame(view):
    assert view.camera_geodetic() is None and view.camera_text() == ""
    view.go_to(46.0, 7.0, 500.0)  # (no scene yet: nothing happens)
    assert view.mode == view.ORBIT
    view.set_scene(FRAME, [])  # (kept for initializeGL)
    assert view._pending_scene is not None
    view.go_to(46.01, 7.02, 500.0)
    lat, lon, h = view.camera_geodetic()
    assert (lat, lon, h) == (pytest.approx(46.01), pytest.approx(7.02), pytest.approx(500.0))
    assert view.mode == view.FLY and view.camera.pitch == -30.0
    text = view.camera_text()
    assert text.startswith("fly") and "46.01000°N" in text and "7.02000°E" in text
    assert "above ground" not in text  # (no terrain yet)


def test_setters_before_the_gl_context_are_kept(view):
    view.set_terrain_source("data", "nodes", None, "© credits")
    view.set_outlines(["outline"])
    view.set_camera_path(["path"])
    view.set_pois(["poi"], None, "anim")
    view.set_photos(["photo"], None)
    view.set_on_demand(False)  # (no renderer: ignored)
    assert view._pending_terrain == ("data", "nodes", None, "© credits")
    assert view._pending_outlines == ["outline"] and view._pending_camera_path == ["path"]
    assert view._pending_pois == (["poi"], None, "anim")
    assert view._pending_photos == (["photo"], None)
    assert view.pick(10, 10) is None and view.clamp_pose((0, 0, 5, 0, 0, 0)) is not None
    view.shutdown()  # (nothing to stop)


def test_on_demand_rendering_repaints_only_after_changes(view, monkeypatch):
    repaints = []
    monkeypatch.setattr(view, "update", lambda: repaints.append(1))
    view._tick()  # the first frame
    assert len(repaints) == 1
    view._tick()
    assert len(repaints) == 1  # idle: no repaint
    view.request_render()
    view._tick()
    assert len(repaints) == 2
    view.camera.heading += 5.0  # the camera moved
    view._tick()
    view._tick()
    assert len(repaints) == 3
    view.fly.pressed.add(FlyController.FORWARD)  # flying: every tick
    view._tick()
    view._tick()
    assert len(repaints) == 5
    view.fly.pressed.clear()
    view.suspended = True  # (exporting a video)
    view.request_render()
    view._tick()
    assert len(repaints) == 5
    view.suspended = False
    scene = Scene()
    view.set_store(scene.store)  # a store change repaints
    scene.store.set("view.fov", 40.0)
    view._tick()
    assert len(repaints) == 6
    scene.store.set("progress.head", 0.5)  # the pulsing marker is animated
    view._tick()
    view._tick()
    assert len(repaints) == 8


def project(camera, point, width, height):
    """Pixel (GL convention) of a world point and its logarithmic depth."""
    from pyglm import glm

    offset = np.asarray(point) - camera.position  # (the view matrix has no translation)
    clip = camera.view_projection(width / height) * glm.vec4(*offset, 1.0)
    px = (clip.x / clip.w + 1.0) / 2.0 * width - 0.5
    py = (clip.y / clip.w + 1.0) / 2.0 * height - 0.5
    distance = float(np.dot(np.asarray(point) - camera.position, camera.forward))
    return px, py, camera.log_depth_coef * np.log2(distance + 1.0)


def test_unproject_log_depth_finds_the_point_again():
    camera = Camera(np.array([100.0, -2000.0, 1500.0]), heading=20.0, pitch=-25.0)
    for point in ([300.0, 0.0, 400.0], [-500.0, 800.0, 0.0], [120.0, -1500.0, 1400.0]):
        px, py, depth = project(camera, point, 800, 450)
        found = unproject_log_depth(camera, px, py, 800, 450, depth)
        assert found == pytest.approx(point, abs=0.05)


def test_click_picks_a_point_and_double_click_orbits_around_it(view):
    """Picking reads the depth under the cursor (the scene target may be smaller than the
    widget); the next click after request_pick() places something there."""
    ratio = view.devicePixelRatio()
    width, height = int(view.width() * ratio), int(view.height() * ratio)
    tw, th = width // 2, height // 2  # (preview quality: half the size)
    depth = view.camera.log_depth_coef * np.log2(1500.0 + 1.0)  # 1.5 km ahead
    read = []
    depths = {"value": depth}
    view.ctx = object()

    def read_depth(x, y):
        read.append((x, y))
        return depths["value"]

    view.renderer = SimpleNamespace(
        target=SimpleNamespace(size=(tw, th)),
        read_depth=read_depth,
        terrain=SimpleNamespace(data=None, exaggeration=1.0),
    )
    x, y = round(view.width() * 0.7), round(view.height() * 0.25)  # (y down)
    px, py = int(x * ratio), int(height - 1 - y * ratio)  # the device pixel (GL: y up)
    tx, ty = int(px * tw / width), int(py * th / height)
    expected = unproject_log_depth(view.camera, tx, ty, tw, th, depth)
    assert view.pick(x, y) == pytest.approx(expected) and read[-1] == (tx, ty)
    ahead = np.dot(expected - view.camera.position, view.camera.forward)
    assert ahead == pytest.approx(1500.0, rel=1e-6)
    assert view.pick(-5, y) is None and view.pick(x, view.height() + 5) is None  # outside
    picked = []
    view.request_pick(picked.append)
    assert view.cursor().shape() == Qt.CursorShape.CrossCursor
    press(view, (x, y))
    release(view, (x, y))
    assert len(picked) == 1 and picked[0] == pytest.approx(expected)
    assert view.cursor().shape() != Qt.CursorShape.CrossCursor
    press(view, (x, y))  # (the next click navigates again)
    release(view, (x, y))
    assert len(picked) == 1
    view.set_mode(view.FLY)
    press(view, (x, y), kind=QEvent.Type.MouseButtonDblClick)
    assert view.mode == view.ORBIT and view.orbit.target == pytest.approx(expected)
    depths["value"] = 1.0  # the sky: nothing to pick
    view.request_pick(picked.append)
    press(view, (x, y))
    assert len(picked) == 1 and view._pick_callback is None
    view.ctx = view.renderer = None


def test_gl_context_takes_the_pending_data(qtbot):
    """Data set before the widget is shown reaches the renderer in initializeGL; later setters
    upload at once; painting flies the camera while a key is held."""
    view = Viewport()
    qtbot.addWidget(view)
    scene = Scene()
    view.set_store(scene.store)
    view.set_scene(FRAME, [], timezone="Europe/Zurich")
    view.set_camera_path([])
    view.resize(320, 200)
    view.show()
    try:
        qtbot.waitUntil(lambda: view.renderer is not None, timeout=5000)
    except AssertionError:
        pytest.skip("no OpenGL 4.3 context for the viewport")
    assert view._pending_scene is None and view._pending_camera_path is None
    assert view.renderer.timezone == "Europe/Zurich" and view.renderer.store is scene.store
    view.set_outlines([])
    assert view._pending_outlines is None  # (uploaded at once)
    view.set_scene(FRAME, [], timezone="UTC")
    assert view.renderer.timezone == "UTC"
    view.set_animated_camera(True)
    assert not view.renderer.camera_path.visible
    view.set_animated_camera(False)
    view.set_mode(view.FLY)
    start = view.camera.position.copy()
    view.fly.pressed.add(FlyController.FORWARD)
    scene.store.set("view.preview", True)  # (half resolution)
    view.export_size = (1080, 1920)  # (portrait export frame drawn into the view)
    qtbot.waitUntil(lambda: not np.allclose(view.camera.position, start), timeout=3000)
    view.fly.pressed.clear()
    stats = []
    view.stats_changed.connect(stats.append)
    view._fps_timer -= 1.0  # (the next frame reports the frame rate)
    view.request_render()
    qtbot.waitUntil(lambda: bool(stats), timeout=3000)
    assert stats[-1].endswith("fps")
    view.shutdown()
    view.shutdown()  # (idempotent)
    view.close()
