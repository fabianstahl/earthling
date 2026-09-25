from pathlib import Path

import numpy as np
import pytest

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Segment, Track
from earthling.render.tracks import (
    PRIMITIVE_RESTART,
    TrackGeometryOptions,
    build_track_positions,
    resample,
    ribbon_vertices,
    smooth,
)

FRAME = LocalFrame(46.0, 7.0, 0.0)


def straight_track(n=200, ele=1500.0):
    lat = np.linspace(46.0, 46.01, n)  # ~1.1 km north
    lon = np.full(n, 7.0)
    seg = Segment(lat, lon, np.full(n, ele), np.full(n, np.datetime64("NaT", "ms")))
    return Track("t", Path("t.gpx"), [seg])


def test_resample_keeps_ends_and_spacing():
    t = straight_track()
    lat, lon, ele = resample(t.segments[0].lat, t.segments[0].lon, t.segments[0].ele, 50.0)
    assert lat[0] == t.segments[0].lat[0] and lat[-1] == t.segments[0].lat[-1]
    assert 20 <= len(lat) <= 25


def test_smooth_preserves_linear_data():
    x = np.linspace(0, 10, 50)
    assert smooth(x, 5)[5:-5] == pytest.approx(x[5:-5])


def test_positions_use_dem_heights_and_offset():
    t = straight_track()
    options = TrackGeometryOptions(elevation_source="dem", height_offset_m=3.0, spacing_m=10.0)
    segs = build_track_positions(
        t, FRAME, options, heights_at=lambda lon, lat: np.full(lon.shape, 2000.0)
    )
    enu, dist = segs[0]
    # z includes curvature drop (tiny at 1 km) -> ~2003 m
    assert enu[:, 2] == pytest.approx(2003.0, abs=0.2)
    assert dist[-1] == pytest.approx(1112, rel=0.02)  # smoothing shortens the ends slightly
    gpx = build_track_positions(
        t, FRAME, TrackGeometryOptions(elevation_source="gpx", height_offset_m=0.0)
    )
    assert gpx[0][0][:, 2] == pytest.approx(1500.0, abs=0.2)


def test_ribbon_layout():
    enu = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    verts, idx = ribbon_vertices([(enu, np.array([0.0, 10, 20]))], enu[0])
    assert verts.shape == (6, 11)
    assert list(verts[:, 9]) == [-1, 1, -1, 1, -1, 1]
    assert idx[-1] == PRIMITIVE_RESTART and len(idx) == 7
    # mirrored neighbours at the ends
    assert verts[0, 3:6] == pytest.approx([-10, 0, 0])


def test_ribbon_renders_with_pixel_width(gl_ctx):
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [straight_track()])
    renderer.store = Scene().store
    renderer.store.set("tracks.elevation", "gpx")
    renderer.store.set("tracks.height_offset", 0.0)
    renderer.store.set("tracks.width", 6.0)
    renderer.store.set("glow.enabled", False)  # the halo would widen the measured line
    renderer.store.set("haze.aerial", 0.0)
    renderer.timezone = "UTC"
    # look straight down on the track from 3 km above its middle
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 3000.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))
    renderer.render(fbo, 200, 200, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3).astype(int)
    red = (img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80)
    row = red[100]
    assert 4 <= row.sum() <= 9  # ~6 px wide
    assert red[:, 100].sum() > 60  # 1.1 km seen from 3 km with 50 deg fov ~ 78 px


def test_glow_adds_a_halo_and_leaves_clean_state(gl_ctx):
    import moderngl

    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [straight_track()])
    renderer.store = Scene().store
    for pid, value in (
        ("tracks.elevation", "gpx"),
        ("tracks.height_offset", 0.0),
        ("tracks.width", 6.0),
        ("haze.aerial", 0.0),
        ("tracks.glow", 5.0),
    ):
        renderer.store.set(pid, value)  # fmt: skip
    renderer.timezone = "UTC"
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 3000.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))

    def red_width(glow):
        renderer.store.set("glow.enabled", glow)
        # simulate a compositor that leaves additive blending enabled between frames
        gl_ctx.enable(moderngl.BLEND)
        gl_ctx.blend_func = moderngl.ONE, moderngl.ONE
        renderer.render(fbo, 200, 200, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3)
        img = img.astype(int)
        return int(((img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80))[100].sum())

    plain = red_width(False)
    assert 4 <= plain <= 9  # the leaked blend state did not wash out the frame
    assert red_width(True) > plain + 4  # halo around the line
