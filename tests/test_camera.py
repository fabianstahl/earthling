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
