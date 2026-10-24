"""Step 19.10: the water mask (vector tiles) and the water surface."""

import json
import os
import time

import httpx
import numpy as np
import pytest

from earthling.data import net
from earthling.data.cache import TileCache
from earthling.data.vector_tiles import decode_tile, ring_area
from earthling.data.water import PROVIDER_ID, TILEJSON_URL, WaterData
from helpers import mock_http


# --- a minimal MVT encoder (only what the tests need) -------------------------------------------
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((number << 3) | wire)
    if wire == 0:
        return key + _varint(payload)
    return key + _varint(len(payload)) + payload


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _polygon(rings) -> bytes:
    cmds, x, y = [], 0, 0
    for ring in rings:
        (x0, y0), rest = ring[0], ring[1:]
        cmds += [(1 << 3) | 1, _zigzag(x0 - x), _zigzag(y0 - y)]
        x, y = x0, y0
        cmds.append((len(rest) << 3) | 2)
        for px, py in rest:
            cmds += [_zigzag(px - x), _zigzag(py - y)]
            x, y = px, py
        cmds.append((1 << 3) | 7)
    return b"".join(_varint(c) for c in cmds)


def water_tile(polygons) -> bytes:
    """A tile with a "water" layer of class "ocean" polygons (lists of rings)."""
    value = _field(1, 2, b"ocean")
    features = b"".join(
        _field(2, 2, _field(2, 2, _varint(0) + _varint(0)) + _field(3, 0, 3)
               + _field(4, 2, _polygon(rings)))
        for rings in polygons
    )  # fmt: skip
    layer = (_field(15, 0, 2) + _field(1, 2, b"water") + features + _field(3, 2, b"class")
             + _field(4, 2, value) + _field(5, 0, 4096))  # fmt: skip
    return _field(3, 2, layer)


# the left half of the tile is sea, with a square island (a hole) in it
SEA = [[(-64, -64), (2048, -64), (2048, 4160), (-64, 4160), (-64, -64)],
       [(512, 1536), (512, 2560), (1536, 2560), (1536, 1536), (512, 1536)]]  # fmt: skip


def test_decode_polygons():
    layers = decode_tile(water_tile([SEA]))
    feature = layers["water"].features[0]
    assert feature.tags == {"class": "ocean"} and feature.is_polygon
    assert ring_area(feature.rings[0]) > 0 > ring_area(feature.rings[1])  # exterior, hole


def test_mask_with_islands(tmp_path):
    cache = TileCache(tmp_path)
    cache.write(PROVIDER_ID, 14, 100, 200, "pbf", water_tile([SEA]))
    (tmp_path / "_vector").mkdir()
    (tmp_path / "_vector" / "openfreemap.json").write_text(
        '{"tiles": ["https://example.invalid/{z}/{x}/{y}.pbf"]}')  # fmt: skip
    water = WaterData(cache)
    water.allow_download = False
    mask = water.mask(14, 100, 200, size=128)
    assert mask.shape == (128, 128, 2)
    cover = mask[..., 0]
    assert cover[64, 10] == pytest.approx(1.0)  # sea
    assert cover[64, 120] == pytest.approx(0.0)  # land
    assert cover[64, 32] == pytest.approx(0.0)  # the island
    assert mask[64, 10, 1] > mask[64, 60, 1]  # open water away from the shore
    # nodes beyond zoom 14 crop the zoom 14 tile (the top-left quarter: sea and the island)
    deeper = water.coverage(15, 200, 400, size=64)
    assert deeper[5, 5] == pytest.approx(1.0) and deeper[56, 32] == pytest.approx(0.0)
    assert water.is_water([0.0], [0.0])[0] == 0.0  # unknown tiles: no water
    assert water.mask(12, 1, 1) is None  # not cached, no download


def test_water_surface_reflects_and_glints(gl_ctx):
    import datetime

    from earthling.core.geo import lonlat_to_tile
    from earthling.core.scene import Scene
    from earthling.render import lod
    from earthling.render.camera import Camera, OrbitController
    from earthling.render.renderer import Renderer
    from helpers import FRAME, FakeTerrainData

    class Sea(FakeTerrainData):
        def texture_for(self, key, source="imagery"):
            if source == "water":
                return np.ones((64, 64, 2), dtype=np.float32)  # all water
            return super().texture_for(key, source)

    def render(**props):
        renderer = Renderer(gl_ctx)
        renderer.set_scene(FRAME, [])
        tx, ty = lonlat_to_tile(7.0, 46.0, 10)
        renderer.set_terrain_source(Sea(), lod.NodeSet({10: [(int(tx), int(ty))]}))
        store = Scene().store
        store.set("sun.datetime", datetime.datetime(2026, 7, 1, 12, 0))
        store.set("haze.aerial", 0.0)
        for pid, value in props.items():
            store.set(pid.replace("__", "."), value)
        renderer.store = store
        renderer.timezone = "UTC"
        camera = Camera()
        renderer.apply_properties(camera)
        renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
        orbit = OrbitController(camera)
        orbit.frame_bounds(*renderer.scene_bounds())
        orbit.pitch = -8.0  # a grazing view: the sea reflects the sky
        orbit.apply()
        renderer.apply_properties(camera)
        renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
        fbo = gl_ctx.simple_framebuffer((64, 64))
        renderer.render(fbo, 64, 64, camera)
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).astype(float)

    plain = render()
    water = render(water__enabled=True)
    assert np.abs(water - plain).mean() > 2.0  # sky reflection on the (green) imagery
    assert np.abs(render(water__enabled=True, water__reflection=0.0) - plain).max() <= 1.0
    # step 19.22: open water in one even colour instead of the imagery
    even = render(water__enabled=True, water__reflection=0.0, water__uniform=1.0,
                  water__color=(0.0, 0.0, 1.0))  # fmt: skip
    rgb = even.reshape(-1, 3)
    assert rgb[:, 2].mean() > rgb[:, 1].mean() + 5.0  # blue sea, not the green imagery


# --- the TileJSON, on-demand downloads, polygon filtering ------------------------------------
TILES_URL = "https://tiles.example/water/{z}/{x}/{y}.pbf"


def tagged_tile(features, extent=4096) -> bytes:
    """A "water" layer of (tags, rings, geometry type) features; string and int tags."""
    keys, values, encoded = [], [], []
    for tags, rings, gtype in features:
        pairs = b""
        for key, value in tags.items():
            if key not in keys:
                keys.append(key)
            if value not in values:
                values.append(value)
            pairs += _varint(keys.index(key)) + _varint(values.index(value))
        encoded.append(_field(2, 2, _field(2, 2, pairs) + _field(3, 0, gtype)
                                    + _field(4, 2, _polygon(rings))))  # fmt: skip
    value_bytes = b"".join(
        _field(4, 2, _field(4, 0, v) if isinstance(v, int) else _field(1, 2, v.encode()))
        for v in values
    )
    layer = (_field(15, 0, 2) + _field(1, 2, b"water") + b"".join(encoded)
             + b"".join(_field(3, 2, k.encode()) for k in keys) + value_bytes
             + _field(5, 0, extent))  # fmt: skip
    return _field(3, 2, layer)


def write_tilejson(root, doc=None):
    path = root / "_vector" / "openfreemap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc if doc is not None else {"tiles": [TILES_URL]}))
    return path


def test_tilejson_downloaded_cached_and_refreshed(tmp_path, monkeypatch):
    answers = []
    requests = mock_http(monkeypatch, lambda r: answers.pop(0))
    cache = TileCache(tmp_path)
    answers.append(httpx.Response(200, json={"tiles": [TILES_URL]}))
    assert WaterData(cache).provider().url_template == TILES_URL
    assert str(requests[0].url) == TILEJSON_URL
    offline = WaterData(cache)
    offline.allow_download = False
    assert offline.provider().url_template == TILES_URL and len(requests) == 1  # from the file
    # older than a month: fetched again, the old one kept when that fails
    path = write_tilejson(tmp_path, {"tiles": ["https://old.example/{z}/{x}/{y}.pbf"]})
    month_ago = time.time() - 31 * 86400
    os.utime(path, (month_ago, month_ago))
    answers.extend(httpx.Response(500) for _ in range(net.MAX_RETRIES + 1))  # (retried)
    assert WaterData(cache).provider().url_template.startswith("https://old.example")
    answers.append(httpx.Response(200, json={"tiles": [TILES_URL]}))
    assert WaterData(cache).provider().url_template == TILES_URL
    assert json.loads(path.read_text())["tiles"] == [TILES_URL]


def test_no_tile_url_no_water(tmp_path, monkeypatch):
    requests = mock_http(monkeypatch, lambda r: httpx.Response(200, json={"tiles": []}))
    water = WaterData(TileCache(tmp_path))
    assert water.provider() is None and water.mask(14, 100, 200) is None
    offline = WaterData(TileCache(tmp_path / "offline"))
    offline.allow_download = False
    assert offline.provider() is None and offline.tile_bytes(14, 1, 1) is None
    assert len(requests) == 1  # only the online instance asked


def test_tiles_downloaded_on_demand(tmp_path, monkeypatch):
    def server(request):
        if request.url.path == "/water/14/100/200.pbf":
            return httpx.Response(200, content=water_tile([SEA]))
        if request.url.path == "/water/14/2/2.pbf":
            raise OSError("connection reset")  # (not an httpx error: not retried)
        return httpx.Response(404)

    requests = mock_http(monkeypatch, server)
    cache = TileCache(tmp_path)
    write_tilejson(tmp_path)
    water = WaterData(cache)
    assert water.polygons(14, 100, 200)[0][0] == 4096  # (extent, rings)
    assert cache.read(PROVIDER_ID, 14, 100, 200, "pbf") is not None
    assert water.tile_bytes(14, 1, 1) is None and cache.is_missing(PROVIDER_ID, 14, 1, 1)
    assert water.tile_bytes(14, 2, 2) is None  # network trouble never breaks rendering
    paths = [r.url.path for r in requests]
    assert water.tile_bytes(14, 1, 1) is None and water.tile_bytes(14, 100, 200) is not None
    assert [r.url.path for r in requests] == paths  # known missing / cached: no new requests
    water.allow_download = False
    assert water.tile_bytes(14, 3, 3) is None and [r.url.path for r in requests] == paths


def test_polygons_only_permanent_water_surfaces(tmp_path):
    square = [[(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]]
    tile = tagged_tile([
        ({"class": "ocean"}, square, 3),
        ({"class": "swimming_pool"}, square, 3),
        ({"class": "lake", "intermittent": 1}, square, 3),  # dry in summer
        ({"class": "lake"}, square, 3),
        ({"class": "river"}, [[(0, 0), (4096, 4096)]], 2),  # a line, not an area
    ])  # fmt: skip
    cache = TileCache(tmp_path)
    write_tilejson(tmp_path)
    cache.write(PROVIDER_ID, 14, 100, 200, "pbf", tile)
    cache.write(PROVIDER_ID, 14, 101, 200, "pbf", b"\x1a\xff\xff")  # truncated
    water = WaterData(cache)
    water.allow_download = False
    assert len(water.polygons(14, 100, 200)) == 2  # the sea and the permanent lake
    sea_only = WaterData(cache, classes={"ocean"})
    sea_only.allow_download = False
    assert len(sea_only.polygons(14, 100, 200)) == 1
    assert water.polygons(14, 101, 200) == []  # undecodable: no water, but no error
    assert water.coverage(14, 101, 200, size=16).max() == 0.0
    for x in range(300):  # the polygon cache stays bounded
        water.polygons(10, x, 0)
    assert len(water._polygons) <= 257


def test_is_water_and_raster(tmp_path):
    from earthling.core.geo import mercator_to_lonlat, tile_bounds_mercator

    cache = TileCache(tmp_path)
    write_tilejson(tmp_path)
    cache.write(PROVIDER_ID, 14, 8500, 5800, "pbf", water_tile([SEA]))
    water = WaterData(cache)
    water.allow_download = False
    x0, y0, x1, y1 = tile_bounds_mercator(14, 8500, 5800)

    def at(fx, fy):  # fractions of the tile, y down
        lon, lat = mercator_to_lonlat(x0 + fx * (x1 - x0), y1 - fy * (y1 - y0))
        return float(lon), float(lat)

    lon, lat = zip(at(0.05, 0.5), at(0.9, 0.5), at(0.25, 0.5), strict=True)
    assert water.is_water(lon, lat).tolist() == [1.0, 0.0, 0.0]  # sea, land, the island
    # two tiles side by side: the cached one with the sea, the other one unknown
    (w, s), (e, n) = at(0.2, 0.9), at(1.5, 0.1)
    raster, origin = water.raster((w, s, e, n), 14, size=32)
    assert origin == (8500, 5800) and raster.shape == (32, 64)
    assert raster[16, 2] == pytest.approx(1.0) and raster[:, 32:].max() == 0.0
