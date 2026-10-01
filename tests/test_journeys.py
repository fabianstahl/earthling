"""Step 18.1: whole-world pyramid, scene origins and far planes for views from space."""

import shutil
from pathlib import Path

import numpy as np
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


FLIGHT_GPX = """<?xml version="1.0"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><name>Flight</name><trkseg>
  <trkpt lat="45.80" lon="6.70"><ele>0</ele><time>2020-01-01T08:00:00Z</time></trkpt>
  <trkpt lat="45.90" lon="6.85"><ele>9000</ele><time>2020-01-01T08:30:00Z</time></trkpt>
  <trkpt lat="46.00" lon="7.00"><ele>0</ele><time>2020-01-01T09:00:00Z</time></trkpt>
 </trkseg></trk>
</gpx>
"""


def test_walked_groups_form_one_journey_in_time_order(gl_ctx, tmp_path):
    from earthling.render.renderer import Renderer
    from earthling.render.tracks import TrackGeometryOptions

    (tmp_path / "flight").mkdir()
    (tmp_path / "flight" / "f.gpx").write_text(FLIGHT_GPX, encoding="utf-8")
    (tmp_path / "earthling.toml").write_text(f"""
[project]
cache_dir = "{(tmp_path / "cache").as_posix()}"
[[tracks]]
name = "own"
gpx = "{(DEMO / "gpx").as_posix()}"
[[tracks]]
name = "flight"
gpx = "flight"
elevation = "gpx"
labels = false
""", encoding="utf-8")  # fmt: skip
    session = Session(Project.load(tmp_path))
    renderer = Renderer(gl_ctx)
    renderer.set_scene(session.frame, session.all_tracks)
    renderer.tracks.ensure_built(TrackGeometryOptions(smoothing=1))
    path = renderer.tracks.path
    # the flight (2020) comes before the demo hike although its group is listed second
    assert path.tracks[0].track.group == "flight"
    assert path.has_time
    assert path.tracks[1].offset_m == pytest.approx(path.tracks[0].length_m)
    # heights from the GPX file: 9 km up in the middle
    assert np.nanmax(path.tracks[0].ele) == pytest.approx(9000.0, abs=1.0)
    assert "trackgroup.flight.dots" in {d.id for d in session.track_group_properties()}


def test_sun_time_follows_the_head_time_between_tracks(gl_ctx, tmp_path):
    from earthling.core.scene import Scene
    from earthling.render.renderer import Renderer
    from earthling.render.tracks import TrackGeometryOptions

    (tmp_path / "flight").mkdir()
    (tmp_path / "flight" / "f.gpx").write_text(FLIGHT_GPX, encoding="utf-8")
    (tmp_path / "earthling.toml").write_text(f"""
[project]
cache_dir = "{(tmp_path / "cache").as_posix()}"
[[tracks]]
name = "own"
gpx = "{(DEMO / "gpx").as_posix()}"
[[tracks]]
name = "flight"
gpx = "flight"
elevation = "gpx"
labels = false
""", encoding="utf-8")  # fmt: skip
    session = Session(Project.load(tmp_path))
    renderer = Renderer(gl_ctx)
    renderer.set_scene(session.frame, session.all_tracks)
    renderer.tracks.ensure_built(TrackGeometryOptions(smoothing=1))
    scene = Scene()
    scene.set_dynamic("track_groups", session.track_group_properties())
    renderer.store = scene.store
    path = renderer.tracks.path
    t = path.time[np.isfinite(path.time)]
    scene.store.set("sun.follow_track", True)
    scene.store.set("progress.mode", "time")
    flight_end = path.tracks[0].time[-1]
    scene.store.set("progress.head", float((flight_end - t[0]) / (t[-1] - t[0])))
    # at the end of the flight: its own time, not the start of the next track there
    assert renderer.track_time().timestamp() == pytest.approx(flight_end, abs=1.0)
