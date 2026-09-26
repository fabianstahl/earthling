from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from earthling.core.config import Config, ConfigError, Project
from earthling.core.gpx import load_gpx_file
from earthling.core.scene import Scene
from earthling.core.session import Session

DEMO_GPX = Path("examples/alps_demo/gpx").resolve()

PLANNED_GPX = """<?xml version="1.0"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><name>Stage A</name><trkseg>
  <trkpt lat="45.90" lon="6.80"><ele>1000</ele></trkpt>
  <trkpt lat="46.00" lon="6.95"><ele>1200</ele></trkpt>
  <trkpt lat="46.20" lon="7.20"><ele>1500</ele></trkpt>
 </trkseg></trk>
 <trk><name>Stage B</name><trkseg>
  <trkpt lat="46.20" lon="7.20"></trkpt>
  <trkpt lat="46.60" lon="7.80"></trkpt>
 </trkseg></trk>
</gpx>
"""


def toml(tmp_path):
    return f"""
[project]
cache_dir = "{(tmp_path / "cache").as_posix()}"

[area]
border_km = 3.0
zones = [{{ within_km = 1.0, imagery_zoom = 15, dem_zoom = 12 }},
         {{ within_km = 3.0, imagery_zoom = 13, dem_zoom = 11 }}]

[[tracks]]
name = "own"
gpx = "{DEMO_GPX.as_posix()}"

[[tracks]]
name = "trail"
label = "The trail"
gpx = "planned/*.gpx"
role = "planned"
[tracks.area]
border_km = 60.0
zones = [{{ within_km = 60.0, imagery_zoom = 8, dem_zoom = 8 }}]
"""


@pytest.fixture
def session(tmp_path):
    (tmp_path / "planned").mkdir()
    (tmp_path / "planned" / "trail.gpx").write_text(PLANNED_GPX, encoding="utf-8")
    return Session(Project(tmp_path, Config.from_toml(toml(tmp_path))))


def test_config_track_groups():
    with pytest.raises(ConfigError, match="unique"):
        Config.from_toml('[[tracks]]\nname="a"\ngpx="x"\n[[tracks]]\nname="a"\ngpx="y"\n')
    with pytest.raises(ConfigError):
        Config.from_toml('[[tracks]]\nname="a"\ngpx="x"\nrole="dreamt"\n')


def test_session_groups_areas_and_plan(session):
    own, trail = session.groups
    assert own.role == "walked" and trail.role == "planned" and trail.label == "The trail"
    assert len(own.tracks) == 2 and [t.name for t in trail.tracks] == ["Stage A", "Stage B"]
    assert session.tracks == own.tracks  # the hike = walked tracks
    assert len(session.all_tracks) == 4 and all(t.group == "trail" for t in trail.tracks)
    # the planned group reaches much further, but only coarsely
    assert trail.aoi.aoi.area > 20 * own.aoi.aoi.area
    plan = session.plan
    assert max(plan.zooms("imagery")) == 15 and max(plan.zooms("dem")) == 12
    fine = plan.levels["imagery"][15]
    coarse = plan.levels["imagery"][8]
    assert len(coarse) >= 4 and len(fine) > 0
    # labels and the local frame use the walked area only
    assert session.detail_aoi is own.aoi
    assert abs(session.frame.lat - np.mean(own.aoi.bounds[1::2])) < 0.01


def test_group_properties_are_dynamic_scene_properties(session):
    scene = Scene()
    rebuilt = []
    scene.dynamic_listeners.append(lambda: rebuilt.append(1))
    scene.set_dynamic("track_groups", session.track_group_properties())
    assert rebuilt and scene.store["trackgroup.trail.dash"] == 10.0
    assert scene.store["trackgroup.own.color_mode"] == "auto"
    options = [v for v, _ in scene.registry["trackgroup.trail.highlight"].options]
    assert options == ["none", "Stage A", "Stage B"]
    scene.animation.set_key("trackgroup.trail.opacity", 0.0, 0.2)
    scene.store.set("trackgroup.trail.highlight", "Stage B")
    doc = scene.to_json()
    # a new scene without the groups reports them as unknown; with them it loads cleanly
    assert Scene().load_json(doc)
    other = Scene()
    other.set_dynamic("track_groups", session.track_group_properties())
    assert other.load_json(doc) == []
    assert other.store["trackgroup.trail.highlight"] == "Stage B"
    assert other.animation.is_animated("trackgroup.trail.opacity")
    # replacing the set drops ids that disappeared
    other.set_dynamic("track_groups", [])
    assert "trackgroup.trail.dash" not in other.registry
    assert not other.animation.is_animated("trackgroup.trail.opacity")


def test_group_styles_and_highlight(session):
    from earthling.render.tracks import TrackLayer, track_color

    scene = Scene()
    scene.set_dynamic("track_groups", session.track_group_properties())
    s = scene.store
    stage_a, stage_b = session.group("trail").tracks

    def style(track, index):
        return TrackLayer._group_style(s, SimpleNamespace(track=track, index=index), False)

    color, opacity, width, glow, dash = style(stage_a, 0)
    assert color == track_color(0) and dash == 10.0 and width == 0.6
    s.set("trackgroup.trail.highlight", "Stage B")
    assert style(stage_b, 1)[0] == s["trackgroup.trail.highlight_color"]
    assert style(stage_a, 0)[1] == pytest.approx(0.9 * 0.35)
    s.set("trackgroup.trail.visible", False)
    assert style(stage_a, 0) is None
    own = session.tracks[0]
    assert style(own, 0)[0] == track_color(0)  # "auto": the global Tracks section decides


def test_progress_follows_walked_tracks_only(gl_ctx, session):
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    renderer.set_scene(session.frame, session.all_tracks)
    scene = Scene()
    scene.set_dynamic("track_groups", session.track_group_properties())
    renderer.store = scene.store
    renderer.timezone = "UTC"
    fbo = gl_ctx.simple_framebuffer((64, 64))
    renderer.render(fbo, 64, 64, Camera(position=np.array([0.0, -20000.0, 20000.0]), pitch=-40))
    path = renderer.tracks.path
    walked = [g for g in renderer.tracks._gpu if g.track.role == "walked"]
    planned = [g for g in renderer.tracks._gpu if g.track.role == "planned"]
    assert len(walked) == 2 and len(planned) == 2
    assert path.total_m == pytest.approx(sum(g.length_m for g in walked))
    assert all(g.offset_m == 0.0 for g in planned)
    assert renderer.tracks.walked() == session.tracks


def test_fast_parser_matches_gpxpy(tmp_path):
    from earthling.core.gpx import _load_with_gpxpy

    for path in sorted(DEMO_GPX.glob("*.gpx")):
        fast, slow = load_gpx_file(path), _load_with_gpxpy(path)
        assert [t.name for t in fast] == [t.name for t in slow]
        for a, b in zip(fast, slow, strict=True):
            for sa, sb in zip(a.segments, b.segments, strict=True):
                assert np.allclose(sa.lat, sb.lat) and np.allclose(sa.lon, sb.lon)
                assert np.allclose(sa.ele, sb.ele, equal_nan=True)
                assert (sa.time == sb.time).all()
    planned = tmp_path / "p.gpx"
    planned.write_text(PLANNED_GPX, encoding="utf-8")
    tracks = load_gpx_file(planned)
    assert np.isnan(tracks[1].segments[0].ele).all() and np.isnat(tracks[0].segments[0].time).all()
    assert load_gpx_file(planned)[0].name == "Stage A"  # from the parse cache
