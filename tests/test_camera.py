import numpy as np
import pytest
from pyglm import glm

from earthling.render.camera import Camera, OrbitController


def test_look_at_sets_heading_and_pitch():
    cam = Camera(position=np.array([0.0, 0.0, 0.0]))
    cam.look_at([100.0, 0.0, 0.0])
    assert cam.heading == pytest.approx(90.0)
    assert cam.pitch == pytest.approx(0.0)
    cam.look_at([0.0, 100.0, -100.0])
    assert cam.heading == pytest.approx(0.0)
    assert cam.pitch == pytest.approx(-45.0)


def test_view_projection_puts_target_in_center():
    cam = Camera()
    orbit = OrbitController(cam)
    orbit.target = np.array([1000.0, 2000.0, 1500.0])
    orbit.distance = 5000
    orbit.apply()
    rel = cam.relative(orbit.target)
    clip = cam.view_projection(1.5) * glm.vec4(*rel, 1.0)
    ndc = clip.xyz / clip.w
    assert abs(ndc.x) < 1e-4 and abs(ndc.y) < 1e-4
    assert clip.w == pytest.approx(5000, rel=1e-4)


def test_frame_bounds_contains_box():
    cam = Camera()
    orbit = OrbitController(cam)
    orbit.frame_bounds(np.array([-1000.0, -1000.0, 0.0]), np.array([1000.0, 1000.0, 500.0]))
    for corner in ([-1000, -1000, 0], [1000, 1000, 500], [1000, -1000, 0]):
        clip = cam.view_projection(1.0) * glm.vec4(*cam.relative(corner), 1.0)
        ndc = clip.xyz / clip.w
        assert abs(ndc.x) <= 1 and abs(ndc.y) <= 1


def test_fly_controller_moves_forward_and_scales_speed():
    from earthling.render.camera import FlyController

    cam = Camera(position=np.array([0.0, 0.0, 1000.0]), heading=90.0, pitch=0.0)
    fly = FlyController(cam)
    fly.pressed.add(fly.FORWARD)
    assert fly.step(0.05, height_above_ground=1000.0)
    assert cam.position[0] > 40 and abs(cam.position[1]) < 1e-6
    assert fly.speed(10_000) > fly.speed(100)
    fly.pressed = {fly.UP}
    z = cam.position[2]
    fly.step(0.05, 1000.0)
    assert cam.position[2] > z
    fly.look(100, 0)
    assert cam.heading == pytest.approx(105.0)


def test_orbit_sync_keeps_camera_position():
    cam = Camera()
    orbit = OrbitController(cam)  # moves the camera to its default orbit
    cam.position = np.array([100.0, -500.0, 800.0])
    orbit.sync_from_camera([0.0, 0.0, 0.0])
    assert cam.position == pytest.approx([100.0, -500.0, 800.0])


def test_unproject_log_depth_roundtrip():
    from earthling.app.viewport import unproject_log_depth

    cam = Camera(position=np.array([0.0, 0.0, 0.0]), heading=30.0, pitch=-10.0)
    target = np.array([500.0, 900.0, -150.0])
    w_, h_ = 800, 600
    clip = cam.view_projection(w_ / h_) * glm.vec4(*cam.relative(target), 1.0)
    ndc = clip.xyz / clip.w
    px = (ndc.x + 1) / 2 * w_ - 0.5
    py = (ndc.y + 1) / 2 * h_ - 0.5
    depth = np.log2(1 + clip.w) * cam.log_depth_coef
    assert unproject_log_depth(cam, px, py, w_, h_, depth) == pytest.approx(target, rel=1e-3)
