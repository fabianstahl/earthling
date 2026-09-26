from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from earthling.core.config import Project
from earthling.core.scene import Scene
from earthling.core.session import Session
from earthling.render.camera import Camera
from earthling.render.renderer import Renderer


def test_sun_follows_the_gpx_time(gl_ctx):
    session = Session(Project.load(Path("examples/alps_demo")))
    renderer = Renderer(gl_ctx)
    renderer.set_scene(session.frame, session.all_tracks)
    scene = Scene()
    renderer.store = scene.store
    renderer.timezone = "Europe/Paris"
    fbo = gl_ctx.simple_framebuffer((32, 32))
    camera = Camera(position=np.array([0.0, -20000.0, 15000.0]), pitch=-30.0)
    renderer.render(fbo, 32, 32, camera)  # builds the track geometry
    path = renderer.tracks.path
    assert path.has_time
    s = scene.store
    s.set("progress.head", 0.25)
    renderer.render(fbo, 32, 32, camera)
    assert renderer.lighting.when == datetime(2026, 7, 1, 12, tzinfo=renderer.lighting.when.tzinfo)
    s.set("sun.follow_track", True)
    renderer.render(fbo, 32, 32, camera)
    head = path.distance_for(0.25)
    expected = datetime.fromtimestamp(path.time_at(head), tz=UTC)
    assert renderer.lighting.when == expected
    s.set("sun.track_offset", 1.5)
    renderer.render(fbo, 32, 32, camera)
    assert renderer.lighting.when == expected + timedelta(hours=1.5)
    # later on the track = later in the day
    s.set("progress.head", 0.4)
    renderer.render(fbo, 32, 32, camera)
    assert renderer.lighting.when > expected + timedelta(hours=1.5)


def test_progress_path_inverse_helpers(gl_ctx):
    session = Session(Project.load(Path("examples/alps_demo")))
    renderer = Renderer(gl_ctx)
    renderer.set_scene(session.frame, session.all_tracks)
    renderer.store = Scene().store
    renderer.render(gl_ctx.simple_framebuffer((16, 16)), 16, 16, Camera())
    path = renderer.tracks.path
    for mode in ("distance", "time"):  # (time mode: several progress values per night pause)
        for d in (500.0, 0.5 * path.total_m, 0.9 * path.total_m):
            p = path.progress_for_distance(d, mode)
            assert path.distance_for(p, mode) == pytest.approx(d, abs=0.5)
    start, end = path.track_range(1)
    assert start == pytest.approx(path.track_range(0)[1]) and end == pytest.approx(path.total_m)
    t = path.time_at(1234.0)
    assert path.distance_for_time(t) == pytest.approx(1234.0, abs=1.0)
