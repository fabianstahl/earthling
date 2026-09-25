import numpy as np
import pytest

from earthling.core.animation import Curve, Interp
from earthling.core.camera_path import (
    CameraSpline,
    camera_path_lines,
    orientation_matrix,
    orientation_to_quat,
    quat_to_orientation,
)
from earthling.core.properties import PropertyDef, PType

CAM = PropertyDef("camera.pose", "Camera", PType.CAMERA, (0, 0, 0, 0, 0, 0))


@pytest.mark.parametrize("hpr", [(0, 0, 0), (90, -30, 0), (215, 20, 10), (359, -80, -25)])
def test_orientation_roundtrip(hpr):
    h, p, r = quat_to_orientation(orientation_to_quat(*hpr))
    assert np.cos(np.radians(h - hpr[0])) == pytest.approx(1.0, abs=1e-9)
    assert (p, r) == pytest.approx(hpr[1:], abs=1e-6)


def test_orientation_matrix_forward():
    m = orientation_matrix(90.0, 0.0, 0.0)
    assert m[:, 1] == pytest.approx([1, 0, 0], abs=1e-12)  # looking east
    assert m[:, 2] == pytest.approx([0, 0, 1], abs=1e-12)


POSES = [
    (0.0, 0.0, 1000.0, 0.0, -20.0, 0.0),
    (1000.0, 500.0, 1200.0, 45.0, -10.0, 0.0),
    (2000.0, 0.0, 1500.0, 120.0, -30.0, 5.0),
    (3000.0, 800.0, 900.0, 170.0, -5.0, 0.0),
    (4000.0, 0.0, 1100.0, 250.0, -15.0, 0.0),
]


def camera_curve(poses=POSES, dt=5.0, interp=Interp.LINEAR):
    c = Curve(CAM)
    for i, pose in enumerate(poses):
        c.set_key(i * dt, pose, interp)
    return c


def test_spline_passes_through_keys_and_is_continuous():
    c = camera_curve()
    for i, pose in enumerate(POSES):
        got = c.evaluate(i * 5.0)
        assert got[:3] == pytest.approx(pose[:3], abs=1e-6)
    # continuity: small time steps produce small position steps everywhere
    times = np.linspace(0, 20, 2001)
    pts = np.array([c.evaluate(t)[:3] for t in times])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() < 10.0
    # rotations are continuous too (no heading flips)
    headings = np.array([c.evaluate(t)[3] for t in times])
    dh = np.abs((np.diff(headings) + 180) % 360 - 180)
    assert dh.max() < 2.0
    # a mid-segment key time value is honoured by orientation too
    assert c.evaluate(10.0)[4] == pytest.approx(-30.0, abs=1e-6)


def test_constant_speed():
    poses = [(0, 0, 0, 0, 0, 0), (100, 0, 0, 0, 0, 0), (1000, 0, 0, 0, 0, 0)]
    c = camera_curve(poses, dt=10.0)
    c.constant_speed = True
    xs = np.array([c.evaluate(t)[0] for t in np.linspace(0, 20, 41)])
    speeds = np.diff(xs)
    assert speeds.std() / speeds.mean() < 0.05
    c.constant_speed = False
    assert c.evaluate(10.0)[0] == pytest.approx(100.0)


def test_single_key_and_gizmo_lines():
    c = camera_curve(POSES[:1])
    assert c.evaluate(3.0) == pytest.approx(POSES[0])
    assert len(camera_path_lines(c)) == 1 + 4  # frustum rectangle + 4 edges
    lines = camera_path_lines(camera_curve())
    assert lines[0][0].shape[1] == 3 and len(lines) == 1 + 5 * 5
    assert camera_path_lines(None) == []


def test_spline_direct():
    s = CameraSpline(POSES)
    assert s.pose(1, 0.0)[:3] == pytest.approx(POSES[1][:3])
    g, lengths = s.arc_table(16)
    assert g[-1] == pytest.approx(4.0) and np.all(np.diff(lengths) >= 0)


def test_window_camera_keys_and_animated_view(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    vp = window.viewport
    vp.camera.position = np.array([0.0, 0.0, 1000.0])
    vp.camera.heading, vp.camera.pitch = 0.0, -20.0
    window.timeline.set_time(0.0)
    window.add_camera_key()
    vp.camera.position = np.array([1000.0, 0.0, 2000.0])
    vp.camera.heading = 90.0
    window.timeline.set_time(4.0)
    window.add_camera_key()
    window.animated_camera_action.setChecked(True)
    window.timeline.set_time(2.0)
    assert 400 < vp.camera.position[0] < 600
    assert vp.camera.heading == pytest.approx(45.0, abs=5.0)
    vp._take_manual_control()
    assert not window.animated_camera_action.isChecked()
    window.timeline.set_time(0.0)
    assert vp.camera.position[0] != 0.0  # the free camera no longer follows
    window.undo_stack.setClean()
