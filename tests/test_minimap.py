"""Mini-map geometry (render/minimap.py) and how the HUD draws it."""

from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from earthling.core.scene import Scene
from earthling.render.minimap import MARGIN, build
from helpers import FRAME

FULL = [(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]  # a whole tile (exterior ring)
ISLAND = [(1024, 1024), (1024, 3072), (3072, 3072), (3072, 1024), (1024, 1024)]  # a hole


def track(east, north, index=0, offset_m=0.0, n=50):
    enu = np.column_stack([np.linspace(*east, n), np.linspace(*north, n), np.zeros(n)])
    dist = np.r_[0.0, np.cumsum(np.hypot(*np.diff(enu[:, :2], axis=0).T))]
    return SimpleNamespace(enu=enu, dist=dist, offset_m=offset_m, index=index,
                           length_m=float(dist[-1]))  # fmt: skip


def journey(*tracks):
    return SimpleNamespace(tracks=list(tracks),
                           total_m=max(t.offset_m + t.length_m for t in tracks))  # fmt: skip


class Water:
    """WaterData stand-in: every vector tile is water (optionally with an island)."""

    def __init__(self, island=False):
        self.rings = [FULL, ISLAND] if island else [FULL]
        self.asked = []

    def polygons(self, z, x, y):
        self.asked.append((z, x, y))
        return [(4096, self.rings)]


def area(triangles):
    u, v = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    return float(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum() / 2.0)


def test_route_is_framed_with_a_margin_and_a_limited_aspect():
    wide = build(journey(track((-5000, 5000), (0, 0))), FRAME)
    assert wide.aspect == pytest.approx(0.5)  # never flatter than 2:1
    _, xy, dist = wide.routes[0]
    assert xy[0, 0] == pytest.approx(MARGIN / (1 + 2 * MARGIN), abs=1e-3)
    assert xy[-1, 0] == pytest.approx(1 - MARGIN / (1 + 2 * MARGIN), abs=1e-3)
    assert np.allclose(xy[:, 1], 0.5, atol=1e-3) and dist[-1] == pytest.approx(10_000.0)
    tall = build(journey(track((0, 0), (-5000, 5000))), FRAME)
    assert tall.aspect == pytest.approx(1.4)  # never taller than 1.4
    _, xy, _ = tall.routes[0]
    assert xy[0, 1] > xy[-1, 1]  # north is up (y down on the map)
    assert np.allclose(xy[:, 0], 0.5, atol=1e-3)
    assert len(tall.water) == 0 and len(tall.borders) == 0  # no sources: no background


def test_tracks_with_offsets_and_degenerate_input():
    day1, day2 = track((0, 3000), (0, 0)), track((3000, 3000), (0, 2000), index=1, offset_m=3000)
    geo = build(journey(day1, day2), FRAME)
    assert [r[0] for r in geo.routes] == [0, 1]
    assert geo.routes[1][2][0] == pytest.approx(3000.0)  # global distance
    assert np.allclose(geo.routes[0][1][-1], geo.routes[1][1][0], atol=1e-6)  # they join
    # a single point (a start before walking): no division by zero
    still = build(journey(track((100, 100), (50, 50), n=3)), FRAME)
    assert np.isfinite(still.routes[0][1]).all() and np.allclose(still.routes[0][1], 0.5)
    # long tracks are thinned out, the distances along with them
    long = build(journey(track((-5000, 5000), (0, 0), n=6000)), FRAME)
    _, xy, dist = long.routes[0]
    assert len(xy) == len(dist) <= 1500


def test_water_is_clipped_to_the_map_and_keeps_islands():
    path = journey(track((-20_000, 20_000), (-8000, 8000)))
    sea = Water()
    geo = build(path, FRAME, water=sea)
    assert 2 <= len({t[1] for t in sea.asked}) <= 5  # a zoom of a few tiles across the map
    assert geo.water.shape[1:] == (3, 2) and area(geo.water) == pytest.approx(1.0, abs=1e-3)
    assert geo.water.min() >= -1e-9 and geo.water.max() <= 1 + 1e-9
    islands = build(path, FRAME, water=Water(island=True))
    assert 0.3 < area(islands.water) < 0.95  # each tile's island is land
    # one feature of two polygons (west and east half) and a degenerate ring: all water
    halves = Water()
    halves.rings = [[(0, 0), (2048, 0), (2048, 4096), (0, 4096), (0, 0)], [(5, 5), (9, 9)],
                    [(2048, 0), (4096, 0), (4096, 4096), (2048, 4096), (2048, 0)]]  # fmt: skip
    assert area(build(path, FRAME, water=halves).water) == pytest.approx(1.0, abs=1e-3)
    dry = SimpleNamespace(polygons=lambda z, x, y: None)  # no vector tiles (offline)
    assert len(build(path, FRAME, water=dry).water) == 0


class Borders:
    def __init__(self, lines):
        self.lines = lines

    def lines_in(self, name, bounds):
        assert name == "countries"
        return np.array(self.lines, dtype=object)


def test_border_segments_are_clipped_to_the_map():
    path = journey(track((-5000, 5000), (0, 0)))  # around 7.0 E, 46.0 N
    border = shapely.LineString([(7.0, 45.0), (7.0, 47.0)])  # north-south through the map
    geo = build(path, FRAME, borders=Borders([border]))
    assert len(geo.borders) >= 1 and geo.borders.shape[1:] == (2, 2)
    assert np.allclose(geo.borders[..., 0], 0.5, atol=1e-3)  # through the middle
    ys = geo.borders[..., 1]  # from the top edge to the bottom edge, not beyond
    assert ys.min() == pytest.approx(0.0, abs=1e-9) and ys.max() == pytest.approx(1.0)
    assert len(build(path, FRAME, borders=Borders([])).borders) == 0
    far = shapely.LineString([(20.0, 10.0), (21.0, 10.0)])
    assert len(build(path, FRAME, borders=Borders([far])).borders) == 0


# --- drawn by the HUD ------------------------------------------------------------------------
W, H = 960, 540


def draw_hud(gl_ctx, path, heads, sources=(None, None), **props):
    from earthling.render.hud import HudLayer
    from earthling.render.labels import FontAtlas
    from earthling.render.shader_library import ShaderLibrary

    store = Scene().store
    store.set("minimap.visible", True)
    store.set("minimap.opacity", 1.0)
    for key, value in props.items():
        store.set(f"minimap.{key}", value)
    atlas = FontAtlas(gl_ctx)
    hud = HudLayer(gl_ctx, ShaderLibrary(gl_ctx), lambda: atlas)
    hud.frame, hud.map_path, hud.map_heads = FRAME, path, heads
    hud.map_colors = {t.index: (1.0, 0.0, 0.0) for t in path.tracks}
    hud.map_sources = sources
    fbo = gl_ctx.simple_framebuffer((W, H))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    hud.render(fbo, W, H, store, None, 0.0, "UTC")
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]
    return img.astype(int), store, hud


def test_hud_draws_water_the_head_and_the_corner(gl_ctx):
    from earthling.render.scale import pixel_scale

    path = journey(track((-20_000, 20_000), (-8000, 8000)))
    img, store, hud = draw_hud(gl_ctx, path, [path.total_m / 2], (Water(), None),
                               water_color=(0.0, 0.0, 1.0))  # fmt: skip
    geo = hud._map_geometry
    k = pixel_scale(W, H)
    w = store["minimap.size"] * k
    x0, y0 = W - 28.0 * k - w, 28.0 * k  # top right
    _, xy, _ = geo.routes[0]
    # the map is water (blue) away from the route
    corner = img[int(y0 + 3), int(x0 + 3)]
    assert corner[2] > 150 and corner[0] < 60 and corner[1] < 60
    # the head: a white dot at the middle of the route, the walked part red before it
    hx, hy = x0 + np.mean(xy[[0, -1], 0]) * w, y0 + np.mean(xy[[0, -1], 1]) * w * geo.aspect
    assert (img[int(hy), int(hx)] > 220).all()
    start = img[int(y0 + xy[3, 1] * w * geo.aspect), int(x0 + xy[3, 0] * w)]
    assert start[0] > 150 and start[2] < 100  # walked: the track colour
    # the rest of the picture stays empty
    assert img[H // 2 :, : W // 2].max() == 0


def test_hud_map_corner_and_broken_sources(gl_ctx):
    class Broken:
        def polygons(self, z, x, y):
            raise OSError("no water today")

    path = journey(track((-5000, 5000), (-1000, 1000)))
    img, _, hud = draw_hud(gl_ctx, path, [], (Broken(), None), position="bottom_left")
    lit = img.sum(axis=2) > 30
    ys, xs = np.nonzero(lit)
    assert xs.max() < W // 2 and ys.min() > H // 2  # bottom left
    assert len(hud._map_geometry.water) == 0  # the map without its background
    # nothing walked: no track colour, no head
    red = (img[..., 0] > 150) & (img[..., 1] < 80)
    assert not red.any() and not (img > 240).all(axis=2).any()
