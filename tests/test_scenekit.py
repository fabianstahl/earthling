import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from earthling import scenekit as sk
from earthling.session import Session

DEMO = Path(__file__).parents[1] / "examples" / "alps_demo"


class FlatTerrain:
    """Terrain stub: everything at 1000 m (no downloads in tests)."""

    def height_at(self, lon, lat, z=None):
        return 1000.0

    def heights_at(self, lon, lat, z=None):
        return np.full(np.shape(lon), 1000.0)


@pytest.fixture
def journey(monkeypatch):
    monkeypatch.setattr(Session, "terrain_data", lambda self, on_demand=True: FlatTerrain())
    return sk.load_journey(DEMO, tz=dt.timezone(dt.timedelta(hours=2)))


def test_load_journey_without_gpu(journey):
    assert sk.current() is journey
    assert len(journey.path.tracks) == 2  # the two demo days, in time order
    assert journey.path.total_m > 10_000
    assert journey.gpu[0].vbo is None  # headless: no GPU buffers
    first, last = journey.day_times(1)
    assert first < last <= journey.day_times(2)[0]
    assert journey.day_index("day02_trient_champex") == 1
    assert abs(journey.ground_z(journey.start(1)) - journey.start(1)[2]) < 10.0  # on the ground
    assert sk.local(first).tzinfo is None
    assert (
        sk.epoch_at(journey, 1, "12:00")
        == dt.datetime.fromtimestamp(first, journey.tz)
        .replace(hour=12, minute=0, second=0, microsecond=0)
        .timestamp()
    )


def test_scene_from_shots_and_holds(journey, tmp_path, monkeypatch):
    scene = sk.new_scene(journey, 10.0)
    assert scene.origin == (journey.frame.lat, journey.frame.lon)
    head_of = sk.head_keys_for(scene, journey, 1, 0.0, 10.0, realtime=False)
    times = np.arange(0.0, 10.0 + 1e-9, 0.25)
    shots = sk.follow_shots(journey, times, head_of, distance=2000.0, pitch=-35.0, side=30.0)
    sk.key_shots(scene, shots)
    pin = sk.poi("start", journey.start(1), "flag")
    scene.set_pois([pin])
    sk.show(scene, "start", None, 6.0)
    monkeypatch.setattr(journey.session.project, "folder", tmp_path)
    path = sk.save(scene, "test", holds=True)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["animation"]["duration"] == 12.0  # a still second at both ends
    cam = scene.animation.curves["camera.pose"]
    assert cam.evaluate(0.0) == pytest.approx(cam.evaluate(1.0))  # the first second holds still
    assert cam.evaluate(11.0) == pytest.approx(cam.evaluate(12.0))
    head = scene.animation.curves["progress.head"]
    day1_end = journey.head_at_time(journey.day_times(1)[1])  # (progress in time mode)
    assert head.evaluate(1.0) < 0.001 and head.evaluate(11.0) == pytest.approx(day1_end, abs=1e-3)


def test_pose_levels_the_horizon_only_when_asked(journey):
    eye, target = np.array([0.0, -5000.0, 3000.0]), np.zeros(3)
    x, y, z, heading, pitch, roll = sk.pose(eye, target)
    assert (x, y, z) == (0.0, -5000.0, 3000.0) and heading == pytest.approx(0.0)
    assert pitch < 0.0 and abs(roll) < 0.5  # near the origin the ENU up is the local up
    assert sk.orbit_eye(target, heading, pitch, np.linalg.norm(eye)) == pytest.approx(eye)
    journey.level_horizon = False
    assert sk.level_roll(eye, target, heading, pitch) == 0.0
