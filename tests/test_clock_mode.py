"""Step 19.9: several heads at once (progress by clock, e.g. two runs as a race)."""

from pathlib import Path

import numpy as np
import pytest

from earthling.core.gpx import Segment, Track
from test_tracks import FRAME

N = 101


def run(group: str, start: str, seconds: float) -> Track:
    """1 km to the north in ``seconds``, starting at ``start``."""
    lat = np.linspace(46.0, 46.009, N)
    lon = np.full(N, 7.0)
    t0 = np.datetime64(start, "ms")
    times = t0 + (np.linspace(0.0, seconds, N) * 1000).astype("timedelta64[ms]")
    track = Track(group, Path(f"{group}.gpx"), [Segment(lat, lon, np.full(N, 1500.0), times)])
    track.group = group
    return track


def setup(gl_ctx):
    from earthling.core.scene import Scene
    from earthling.render.parameters import flt
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    # this year's run (5 min) and last year's (6 min, one year earlier)
    renderer.set_scene(FRAME, [run("now", "2026-09-27T09:00", 300.0),
                               run("then", "2025-09-28T09:00", 360.0)])  # fmt: skip
    scene = Scene()
    scene.set_dynamic("groups", [
        flt("trackgroup.now.time_offset", "shift", 0.0, -1e6, 1e6),
        flt("trackgroup.then.time_offset", "shift", 0.0, -1e6, 1e6)])  # fmt: skip
    store = scene.store
    store.set("trackgroup.then.time_offset", 364 * 24.0)  # the same start time
    store.set("tracks.elevation", "gpx")
    store.set("progress.mode", "clock")
    renderer.store = store
    return renderer, store


def test_every_track_has_its_own_head(gl_ctx):
    from earthling.render.camera import Camera

    renderer, store = setup(gl_ctx)
    fbo = gl_ctx.simple_framebuffer((64, 64))
    camera = Camera(np.array([0.0, 500.0, 4000.0]), 0.0, -89.0, 0.0)
    store.set("progress.head", 0.5)  # 3 of 6 minutes
    renderer.render(fbo, 64, 64, camera)
    path = renderer.tracks.path
    heads = {g.track.group: h - g.offset_m for g in path.tracks
             for h in [renderer.tracks.track_heads[id(g)]]}  # fmt: skip
    length = path.tracks[0].length_m
    assert heads["now"] == pytest.approx(length * 180 / 300, rel=0.02)  # 60 % of the way
    assert heads["then"] == pytest.approx(length * 180 / 360, rel=0.02)  # 50 %
    assert len(renderer.marker.extra) == 2 and renderer.marker.position is None
    # the lead group (the first) drives the camera / stats; the other one is a minute behind
    assert path.distance_for(0.5, "clock") == pytest.approx(
        path.lead().offset_m + heads["now"], rel=1e-6)  # fmt: skip
    assert path.gap_at(path.clock_at(0.5)) == pytest.approx(36.0, abs=2.0)
    store.set("progress.head", 0.0)
    renderer.render(fbo, 64, 64, camera)
    assert len(renderer.marker.extra) == 2  # both at the start


def test_gap_in_the_stats(gl_ctx):
    from earthling.render.camera import Camera
    from earthling.render.hud import format_gap

    assert format_gap(83.0) == "+1:23" and format_gap(-45.0) == "−0:45"
    assert format_gap(3730.0) == "+1:02:10"
    renderer, store = setup(gl_ctx)
    store.set("progress.head", 0.5)
    store.set("stats.visible", True)
    store.set("stats.show_gap", True)
    fbo = gl_ctx.simple_framebuffer((320, 180))
    renderer.render(fbo, 320, 180, Camera(np.array([0.0, 500.0, 4000.0]), 0.0, -89.0, 0.0))
    assert renderer.hud.gap_s == pytest.approx(36.0, abs=2.0)
    # the stats show the lead run alone (not both runs added up)
    assert renderer.hud.last_stats.distance_m == pytest.approx(600.0, rel=0.05)
