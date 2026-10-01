"""Step 18.1: whole-world pyramid, scene origins and far planes for views from space."""

import shutil
from pathlib import Path

import pytest

from earthling.core.aoi import compute_aoi, plan_tiles_multi
from earthling.core.config import Config, ConfigError, Project
from earthling.core.scene import Scene
from earthling.core.session import Session
from earthling.render.camera import far_for_height
from test_aoi import AREA, straight_track

DEMO = Path(__file__).parents[1] / "examples" / "alps_demo"


def test_world_pyramid_covers_the_globe_from_zoom_0():
    aoi = compute_aoi([straight_track()], AREA)
    plan = plan_tiles_multi([(aoi, AREA)], world=(4, 3))
    assert plan.zooms("imagery")[0] == 0
    for z in range(5):
        assert plan.count("imagery", z) == 4**z  # every tile of the world
    assert plan.count("dem", 3) == 64 and plan.count("dem", 4) < 64
    assert plan.zooms("imagery")[-1] == 15  # the track zones are unchanged
    without = plan_tiles_multi([(aoi, AREA)])
    assert plan.count("imagery", 15) == without.count("imagery", 15)


def test_world_section_is_optional_and_validated():
    assert Config.from_toml("").world is None
    world = Config.from_toml("[world]\nimagery_zoom = 3\n").world
    assert world.imagery_zoom == 3 and world.dem_zoom == 3
    with pytest.raises(ConfigError):
        Config.from_toml("[world]\nimagery_zoom = 9\n")


def test_scene_origin_roundtrip(tmp_path):
    scene = Scene()
    assert "origin" not in scene.to_json()
    scene.origin = (13.75, 100.5)
    path = scene.save(tmp_path / "s.json")
    other = Scene()
    other.load(path)
    assert other.origin == (13.75, 100.5)
    other.reset()
    assert other.origin is None


def test_session_origin_override(tmp_path):
    shutil.copytree(DEMO, tmp_path / "demo")
    session = Session(Project.load(tmp_path / "demo"))
    default = session.frame
    assert not session.set_origin(None)
    assert session.set_origin((45.9, 6.9))
    assert (session.frame.lat, session.frame.lon) == (45.9, 6.9)
    assert not session.set_origin((45.9, 6.9))
    assert session.set_origin(None)
    assert session.frame == default


def test_far_plane_reaches_the_horizon_from_space():
    assert far_for_height(2000.0) == 2_000_000.0
    # from 10 000 km the horizon is ~15 000 km away
    assert far_for_height(10_000e3) > 15_000e3
