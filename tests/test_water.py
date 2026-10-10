"""Step 19.10: the water mask (vector tiles) and the water surface."""

import numpy as np
import pytest

from earthling.data.cache import TileCache
from earthling.data.vector_tiles import decode_tile, ring_area
from earthling.data.water import PROVIDER_ID, WaterData


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
    from test_terrain import FRAME, FakeTerrainData

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
