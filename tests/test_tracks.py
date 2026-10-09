from pathlib import Path

import numpy as np
import pytest

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Segment, Track
from earthling.render.tracks import (
    PRIMITIVE_RESTART,
    TrackGeometryOptions,
    TrackSegmentGeometry,
    build_track_positions,
    resample,
    ribbon_vertices,
    smooth,
)

FRAME = LocalFrame(46.0, 7.0, 0.0)


# pixel sizes are given for 1080p and scale with the image height; the test images are 200 px
PX = 1080 / 200


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
    enu, dist = segs[0].enu, segs[0].dist
    # z includes curvature drop (tiny at 1 km) -> ~2003 m
    assert enu[:, 2] == pytest.approx(2003.0, abs=0.2)
    assert dist[-1] == pytest.approx(1112, rel=0.02)  # smoothing shortens the ends slightly
    gpx = build_track_positions(
        t, FRAME, TrackGeometryOptions(elevation_source="gpx", height_offset_m=0.0)
    )
    assert gpx[0].enu[:, 2] == pytest.approx(1500.0, abs=0.2)


def test_ribbon_layout():
    enu = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    geometry = TrackSegmentGeometry(enu, np.array([0.0, 10, 20]), np.full(3, np.nan))
    verts, idx = ribbon_vertices([geometry], enu[0])
    assert verts.shape == (6, 17)
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
    # widths are given for 1080p and scale with the image height (step 18.5)
    for size, low, high in ((1080, 4, 9), (540, 2, 5)):
        fbo = gl_ctx.simple_framebuffer((size, size))
        renderer.render(fbo, size, size, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
        img = img.reshape(size, size, 3).astype(int)
        red = (img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80)
        assert low <= red[size // 2].sum() <= high  # ~6 px wide at 1080p, ~3 px at 540
        # 1.1 km seen from 3 km with 50 deg fov ~ 0.39 of the image height
        assert red[:, size // 2].sum() > 0.3 * size
        fbo.release()


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
        ("tracks.width", 6.0 * PX),
        ("tracks.outline", PX),
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


def timed_track(name, start, n=100, lat0=46.0, minutes=60):
    lat = np.linspace(lat0, lat0 + 0.01, n)
    lon = np.full(n, 7.0)
    t0 = np.datetime64(start, "ms")
    times = t0 + (np.arange(n) * (minutes * 60_000 // (n - 1))).astype("timedelta64[ms]")
    seg = Segment(lat, lon, np.full(n, 1500.0), times)
    return Track(name, Path(f"{name}.gpx"), [seg])


def build_path(tracks):
    from earthling.render.tracks import ProgressPath, _GpuTrack

    gpus, offset = [], 0.0
    for i, t in enumerate(tracks):
        segs = build_track_positions(t, FRAME, TrackGeometryOptions(elevation_source="gpx"))
        enu = np.vstack([s.enu for s in segs])
        dist = np.concatenate([s.dist for s in segs])
        time = np.concatenate([s.time for s in segs])
        g = _GpuTrack(t, i, enu[0], None, None, float(dist[-1]), {}, offset, enu, dist, time)
        offset += g.length_m
        gpus.append(g)
    return ProgressPath(gpus)


def test_progress_by_distance_and_time():
    # day 1: 08:00-09:00, day 2 the next morning 08:00-09:00 (same length each)
    path = build_path([
        timed_track("d1", "2026-07-01T08:00", lat0=46.0),
        timed_track("d2", "2026-07-02T08:00", lat0=46.01),
    ])  # fmt: skip
    total = path.total_m
    assert total == pytest.approx(2 * 1100, rel=0.03)
    assert path.distance_for(0.5, "distance") == pytest.approx(total / 2)
    # half of the elapsed time (the night) -> still at the end of day 1
    d1_end = path.tracks[0].length_m
    assert path.distance_for(0.5, "time") == pytest.approx(d1_end, rel=0.01)
    assert path.distance_for(1.0, "time") == pytest.approx(total)
    pos = path.position_at(d1_end + 1.0)
    assert pos[1] > path.tracks[0].enu[-1][1]  # into the second track (further north)


def test_progress_without_times_falls_back_to_distance():
    path = build_path([straight_track()])
    assert not path.has_time
    assert path.distance_for(0.25, "time") == pytest.approx(path.total_m / 4)


def test_head_hides_the_rest_of_the_track(gl_ctx):
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [straight_track()])
    renderer.store = Scene().store
    for pid, value in (
        ("tracks.elevation", "gpx"),
        ("tracks.height_offset", 0.0),
        ("haze.aerial", 0.0),
        ("glow.enabled", False),
        ("marker.visible", False),
        ("progress.head", 0.5),
    ):
        renderer.store.set(pid, value)  # fmt: skip
    renderer.timezone = "UTC"
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 3000.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))
    renderer.render(fbo, 200, 200, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3).astype(int)
    red = (img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80)
    rows = np.where(red.any(axis=1))[0]  # image rows are bottom-up: row 0 = south
    assert rows.max() < 105 and rows.min() < 80  # only the southern (first) half is drawn


def _top_down(gl_ctx, tracks, **props):
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, tracks)
    renderer.store = Scene().store
    for pid, value in {"tracks.elevation": "gpx", "tracks.height_offset": 0.0,
                       "tracks.width": 6.0 * PX, "tracks.outline": PX, "glow.enabled": False,
                       "haze.aerial": 0.0, "marker.visible": False,
                       **props}.items():  # fmt: skip
        renderer.store.set(pid, value)
    renderer.timezone = "UTC"
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 3000.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))
    renderer.render(fbo, 200, 200, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3).astype(int)
    return img, renderer


def test_casing_darkens_around_the_line(gl_ctx):
    plain, _ = _top_down(gl_ctx, [straight_track()])
    cased, _ = _top_down(gl_ctx, [straight_track()], **{"tracks.casing": 4.0 * PX})
    beside = (slice(90, 110), slice(104, 107))  # just right of the ~6 px line
    assert cased[beside].mean() < plain[beside].mean() - 20
    red = lambda img: (img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80)  # noqa: E731
    assert abs(int(red(cased)[100].sum()) - int(red(plain)[100].sum())) <= 1  # same line


def test_tracks_do_not_write_depth(gl_ctx):
    """A (nearly transparent) line must not hide tracks below it or cut the clouds / rain,
    which march up to the scene depth: only the terrain writes depth."""
    img, renderer = _top_down(gl_ctx, [straight_track()])
    red = (img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80)
    assert red[100].any()  # the line is drawn (no terrain in this test)
    x = int(np.flatnonzero(red[100])[0])
    assert renderer.read_depth(x, 99) == pytest.approx(1.0)  # still the cleared depth


def test_overlapping_tracks_do_not_add_up_their_glow(gl_ctx):
    def glow_max(tracks):
        _, renderer = _top_down(
            gl_ctx, tracks, **{"glow.enabled": True, "tracks.color_mode": "single"}
        )
        glow = renderer.target.glow
        return np.frombuffer(glow.read(), dtype=np.float16).astype(np.float32).max()

    one = glow_max([straight_track()])
    twice = glow_max([straight_track(), straight_track()])  # the same line drawn twice
    assert one > 0.0 and twice == pytest.approx(one, rel=0.05)


def test_tracks_on_top_of_each_other_do_not_mix_colours(gl_ctx):
    """The upper track replaces the glow of the one below; no third colour appears."""
    from earthling.render.tracks import track_color

    below, above = straight_track(), straight_track()
    below.name, above.name = "below", "above"
    _, renderer = _top_down(gl_ctx, [below, above], **{"glow.enabled": True})
    glow = np.frombuffer(renderer.target.glow.read(), dtype=np.float16).astype(np.float32)
    glow = glow.reshape(200, 200, 4)
    rgb = glow[..., :3].reshape(-1, 3)
    brightest = rgb[np.argmax(rgb.sum(axis=1))]
    top = np.array(track_color(1)) ** 2.2
    assert np.allclose(brightest / brightest.max(), top / top.max(), atol=0.05)


def test_depth_bias_keeps_tracks_above_coarse_terrain(gl_ctx):
    from earthling.core.geo import lonlat_to_tile
    from earthling.core.scene import Scene
    from earthling.render import lod
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer
    from test_terrain import FakeTerrainData

    def red_pixels(bias):
        renderer = Renderer(gl_ctx)
        renderer.set_scene(FRAME, [straight_track(ele=1490.0)])  # 10 m under the plateau
        tx, ty = lonlat_to_tile(7.0, 46.0, 10)
        renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
        renderer.store = Scene().store
        for pid, value in {"tracks.elevation": "gpx", "tracks.height_offset": 0.0,
                           "tracks.width": 6.0 * PX, "glow.enabled": False, "haze.aerial": 0.0,
                           "marker.visible": False, "tracks.depth_bias": bias}.items():  # fmt: skip
            renderer.store.set(pid, value)
        renderer.timezone = "UTC"
        mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
        camera = Camera(position=mid + np.array([0.0, -2500.0, 1500.0]), heading=0.0, pitch=-31.0)
        renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 200)
        fbo = gl_ctx.simple_framebuffer((200, 200))
        renderer.render(fbo, 200, 200, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3)
        img = img.astype(int)
        return ((img[..., 0] > img[..., 2] + 60) & (img[..., 0] > img[..., 1] + 40)).sum()

    assert red_pixels(0.0) < 5 and red_pixels(0.01) > 50


def test_hairpins_have_no_miter_spikes(gl_ctx):
    """A zig-zag track stays within a narrow band around its centre line (no long spikes)."""
    n = 60
    lat = 46.0 + np.linspace(0.0, 0.01, n)
    lon = 7.0 + np.where(np.arange(n) % 2 == 0, 0.0, 0.0012)  # hairpins every point
    seg = Segment(lat, lon, np.full(n, 1500.0), np.full(n, np.datetime64("NaT", "ms")))
    img, _ = _top_down(gl_ctx, [Track("zz", Path("zz.gpx"), [seg])], **{"tracks.smoothing": 1,
                       "tracks.width": 10.0 * PX})  # fmt: skip
    red = (img[..., 0] > img[..., 2] + 40) & (img[..., 0] > 80)
    cols = np.flatnonzero(red.any(axis=0))
    width = cols.max() - cols.min()
    # the zig-zag spans ~90 m = ~24 px at 3 km with a 50 degree fov, plus the 10 px line
    assert width < 24 + 10 + 16


def test_walked_tracks_are_drawn_on_top_of_planned_routes(gl_ctx):
    from earthling.render.tracks import track_color

    walked, planned = straight_track(), straight_track()
    walked.name, planned.name, planned.role = "walked", "planned", "planned"
    # the planned route comes later in the list, but must end up below
    img, _ = _top_down(gl_ctx, [walked, planned])
    centre = img[90:110, 99:101].reshape(-1, 3).mean(axis=0)
    expected = np.array(track_color(0)) * 255
    assert np.argmax(centre) == np.argmax(expected) == 0  # the walked (orange-red) colour


def test_motion_values():
    from earthling.render.tracks import motion_values

    ground = np.arange(0.0, 1001.0, 10.0)
    ele = 1000.0 + ground * 0.08  # 8 % climb
    time = 1.7e9 + ground / 2.5  # 2.5 m/s = 9 km/h
    hr = np.full(len(ground), 150.0)
    v = motion_values(ground, ele, time, hr)
    assert v[50] == pytest.approx([1040.0, 8.0, 9.0, 150.0])
    assert v[0, 1] == pytest.approx(8.0)  # at the ends too
    no_time = motion_values(ground, ele, np.full(len(ground), np.nan))
    assert np.isnan(no_time[:, 2]).all() and np.isnan(no_time[:, 3]).all()


def test_tracks_coloured_by_gradient(gl_ctx):
    """A climbing track takes the colour of the ramp at its gradient."""
    from earthling.core.gpx import Segment, Track
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    n = 200
    lat = np.full(n, FRAME.lat)
    lon = FRAME.lon + np.linspace(-0.02, 0.02, n)
    ele = np.linspace(1000.0, 1300.0, n)  # 300 m over 3 km: 10 %
    times = np.datetime64("2026-07-01T08:00") + (np.arange(n) * 10).astype("timedelta64[s]")
    track = Track("climb", Path("climb.gpx"), [Segment(lat, lon, ele, times)])
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [track])
    scene = Scene()
    store = scene.store
    renderer.store = store
    store.set("tracks.elevation", "gpx")
    store.set("tracks.width", 12.0)
    store.set("glow.enabled", False)
    store.set("tracks.casing", 0.0)
    store.set("tracks.outline", 0.0)
    store.set("tracks.emissive", 1.0)
    store.set("haze.aerial", 0.0)
    camera = Camera(np.array([0.0, 0.0, 6000.0]), 0.0, -89.9, 0.0)
    fbo = gl_ctx.simple_framebuffer((320, 180))

    def centre_colour():
        renderer.render(fbo, 320, 180, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(180, 320, 3)
        column = img[60:120, 140:180].astype(int)  # the line crosses the middle (sky around)
        sky = column[0, 0]
        far = np.abs(column - sky).sum(axis=2)
        return column[far > 0.8 * far.max()].mean(axis=0)

    store.set("tracks.color_mode", "single")
    store.set("tracks.color", (0.1, 0.9, 0.1))
    green = centre_colour()
    assert green[1] > green[0] + 40 and green[1] > green[2] + 40
    store.set("tracks.color_by", "gradient")
    store.set("tracks.color_min", 0.0)
    store.set("tracks.color_max", 10.0)  # 10 % = the red end of turbo
    red = centre_colour()
    assert red[0] > red[1] + 80 and red[0] > red[2] + 80


def test_tracks_split_by_water(gl_ctx):
    """Step 19.11: the parts of a track on water take the group's water style."""
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.parameters import boolean, color, flt
    from earthling.render.renderer import Renderer

    track = straight_track()  # 46.0 .. 46.01 N
    track.split_water = True
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [track])
    scene = Scene()
    p = "trackgroup.tracks."
    scene.set_dynamic("groups", [color(p + "water_color", "w", (0.1, 0.3, 1.0)),
                                 flt(p + "water_dash", "d", 0.0, 0.0, 100.0),
                                 boolean(p + "water_dots", "o", False)])  # fmt: skip
    store = scene.store
    renderer.store = store
    for pid, value in (("tracks.elevation", "gpx"), ("tracks.width", 8.0), ("haze.aerial", 0.0),
                       ("glow.enabled", False), ("tracks.color_mode", "single"),
                       ("tracks.color", (1.0, 0.2, 0.1)), ("tracks.emissive", 1.0),
                       ("tracks.outline", 0.0), ("marker.visible", False)):  # fmt: skip
        store.set(pid, value)
    renderer.tracks.water_at = lambda lon, lat: (np.asarray(lat) > 46.005).astype(float)
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 3000.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))
    renderer.render(fbo, 200, 200, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3)
    img = img[::-1].astype(int)  # top row first: north at the top
    column = img[:, 95:105]
    blue = (column[..., 2] > column[..., 0] + 60).any(axis=1)
    red = (column[..., 0] > column[..., 2] + 60).any(axis=1)
    # the track runs from row ~61 (north, water) to ~139 (south, land)
    assert blue[65:95].mean() > 0.8 and red[105:135].mean() > 0.8
