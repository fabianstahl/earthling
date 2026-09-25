import numpy as np
import pytest

from earthling.core.animation import Interp
from earthling.core.scene import Scene
from earthling.render.camera import forward_vector
from earthling.render.camera_rig import CameraRig, blend_poses, look_orientation


def test_look_orientation():
    h, p, r = look_orientation(np.zeros(3), np.array([100.0, 0.0, -100.0]))
    assert h == pytest.approx(90.0) and p == pytest.approx(-45.0)


def test_blend_poses_endpoints():
    a = (0, 0, 0, 10, -5, 0)
    b = (100, 0, 50, 350, -25, 0)
    assert blend_poses(a, b, 0.0) == pytest.approx(a, abs=1e-6)
    mid = blend_poses(a, b, 0.5)
    assert mid[0] == pytest.approx(50) and (mid[3] < 1 or mid[3] > 359)  # shortest way round


class StraightPath:
    """Minimal ProgressPath stand-in: a 10 km line to the north at 1000 m."""

    total_m = 10_000.0

    def distance_for(self, progress, mode="distance"):
        return progress * self.total_m

    def position_at(self, d):
        d = min(max(d, 0.0), self.total_m)
        return np.array([0.0, d, 1000.0])


def make_rig():
    scene = Scene()
    rig = CameraRig(scene.animation, scene.store)
    rig.path = StraightPath()
    anim = scene.animation
    anim.set_key("camera.pose", 0.0, (0, -1000, 2000, 0, -20, 0))
    anim.set_key("camera.pose", 10.0, (5000, -1000, 2000, 0, -20, 0))
    return scene, rig


def test_look_at_mode_points_at_the_target():
    scene, rig = make_rig()
    scene.store.set("camera.mode", "look_at")
    scene.store.set("camera.target", (2500.0, 3000.0, 1000.0))
    pose = rig.pose(5.0)
    f = forward_vector(pose[3], pose[4])
    to_target = np.array([2500.0, 3000.0, 1000.0]) - np.array(pose[:3])
    assert np.dot(f, to_target / np.linalg.norm(to_target)) == pytest.approx(1.0, abs=1e-6)


def test_follow_mode_stays_behind_and_looks_ahead():
    scene, rig = make_rig()
    scene.store.set("camera.mode", "follow")
    scene.store.set("follow.lag", 0.0)
    scene.animation.set_key("progress.head", 0.0, 0.0)
    scene.animation.set_key("progress.head", 10.0, 1.0)
    pose = rig.pose(5.0)  # hiker at y = 5000
    assert pose[1] == pytest.approx(5000 - 600, abs=1)  # default distance behind
    assert pose[2] == pytest.approx(1250, abs=1)  # default height above
    assert pose[3] == pytest.approx(0.0, abs=1e-6) or pose[3] == pytest.approx(360.0)
    assert pose[4] < 0  # looking down at the track ahead


def test_mode_switch_blends_over_the_transition():
    scene, rig = make_rig()
    scene.store.set("camera.transition", 2.0)
    anim = scene.animation
    anim.set_key("camera.mode", 0.0, "keys", Interp.STEP)
    anim.set_key("camera.mode", 4.0, "look_at", Interp.STEP)
    scene.store.set("camera.target", (2000.0, 0.0, 0.0))

    def pose_at(t):
        anim.apply(t)
        return np.array(rig.pose(t))

    before, at_switch, after = pose_at(3.99), pose_at(4.0), pose_at(6.5)
    assert np.abs(at_switch[3:5] - before[3:5]).max() < 1.0  # no jump at the switch
    headings = [pose_at(t)[3] for t in np.linspace(4.0, 6.0, 21)]
    steps = np.abs((np.diff(headings) + 180) % 360 - 180)
    assert steps.max() < 15.0  # smooth turn towards the target
    target_dir = np.array([2000.0, 0, 0]) - after[:3]
    f = forward_vector(after[3], after[4])
    assert np.dot(f, target_dir / np.linalg.norm(target_dir)) == pytest.approx(1.0, abs=1e-6)
