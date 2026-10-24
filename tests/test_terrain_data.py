"""TerrainData: heightmaps and textures per LOD node, height queries, on-demand downloads."""

from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import shapely

from earthling.core.aoi import TilePlan
from earthling.core.geo import lonlat_to_tile, tile_bounds_lonlat
from earthling.data.cache import TileCache
from earthling.data.coverage import register_coverage
from earthling.data.dem import (
    DEM_SOURCES,
    HEIGHTMAP_SAMPLES,
    DemSource,
    encode_heightmap,
    sample_positions_mercator,
)
from earthling.data.providers import TileProvider
from earthling.data.terrain_data import TerrainData
from helpers import mock_http, png

IMAGERY = TileProvider("test_td_imagery", "Imagery", "imagery",
                       "https://img.example/{z}/{x}/{y}.png", ext="png", max_zoom=14,
                       max_requests_per_second=0)  # fmt: skip
Z = 11
TX, TY = (int(v) for v in lonlat_to_tile(7.0, 46.0, Z))
KEY = (Z, TX, TY)
N = HEIGHTMAP_SAMPLES
register_coverage("test_td_square", shapely.box(7.0, 46.0, 7.2, 46.2), 300.0)
register_coverage("test_td_west", shapely.box(6.0, 45.0, 7.0, 47.0), 300.0)


def plan(dem=None, imagery=None):
    return TilePlan({"dem": dem or {}, "imagery": imagery or {}})


def test_heights_join_at_tile_borders(tmp_path):
    cache = TileCache(tmp_path)
    right = (Z, TX + 1, TY)
    base = sample_positions_mercator(*KEY)[0][0]
    for key in (KEY, right):  # 1 m per 100 m eastwards: continuous across the border
        mx, _ = sample_positions_mercator(*key)
        cache.write("dem", *key, "png", encode_heightmap(np.tile(1000.0 + (mx - base) * 0.01,
                                                                 (N, 1))))  # fmt: skip
    data = TerrainData(cache, "dem", IMAGERY, plan({Z: np.array([KEY[1:], right[1:]])}))
    west, south, east, north = tile_bounds_lonlat(*KEY)
    lat = (south + north) / 2
    on_border = data.height_at(east, lat)
    left_of, right_of = data.height_at(east - 1e-7, lat), data.height_at(east + 1e-7, lat)
    assert on_border == pytest.approx(left_of, abs=0.05) == pytest.approx(right_of, abs=0.05)
    # the ramp across both tiles, sampled densely, has no step at the border
    lons = np.linspace(west + 0.01, east + (east - west) - 0.01, 400)
    h = data.heights_at(lons, np.full(400, lat))
    assert np.isfinite(h).all() and (np.diff(h) > 0).all()
    assert np.abs(np.diff(h, 2)).max() < 0.25  # (0.2 m height steps of the encoding)
    # the shape of the input is kept
    grid = data.heights_at(lons[:6].reshape(2, 3), np.full((2, 3), lat))
    assert grid.shape == (2, 3) and np.allclose(grid.ravel(), h[:6])


def test_gaps_are_filled_from_the_parent(tmp_path):
    """A national DEM ending at the border: the coarser parent fills the gap."""
    cache = TileCache(tmp_path)
    child = np.full((N, N), 800.0, dtype=np.float32)
    child[:, :130] = np.nan  # no data in the western half
    cache.write("dem", *KEY, "png", encode_heightmap(child))
    lone = (Z, TX + 4, TY)  # the same gap, but no parent heightmap
    cache.write("dem", *lone, "png", encode_heightmap(child))
    cache.write("dem", Z - 1, TX >> 1, TY >> 1, "png", encode_heightmap(np.full((N, N), 500.0)))
    data = TerrainData(cache, "dem", IMAGERY, plan({Z: np.array([KEY[1:]])}))
    heights = data.heightmap_for(KEY).heights
    assert np.isfinite(heights).all()
    assert heights[100, 10] == pytest.approx(500.0, abs=0.2)
    assert heights[100, 200] == pytest.approx(800.0, abs=0.2)
    # without a parent the gap stays: no height there (None / NaN), heights elsewhere
    west, south, east, north = tile_bounds_lonlat(*lone)
    lat = (south + north) / 2
    gap, solid = west + 0.1 * (east - west), west + 0.9 * (east - west)
    assert data.height_at(gap, lat, z=Z) is None
    assert data.height_at(solid, lat, z=Z) == pytest.approx(800.0, abs=0.2)
    h = data.heights_at(np.array([gap, solid]), np.array([lat, lat]), z=Z)
    assert np.isnan(h[0]) and h[1] == pytest.approx(800.0, abs=0.2)


def test_default_zoom_and_outside_the_data(tmp_path):
    cache = TileCache(tmp_path)
    z12 = (12, *(int(v) for v in lonlat_to_tile(7.0, 46.0, 12)))
    cache.write("dem", *z12, "png", encode_heightmap(np.full((N, N), 321.0)))
    data = TerrainData(cache, "dem", IMAGERY, plan())  # no DEM levels: zoom 12
    assert data.height_at(7.0, 46.0) == pytest.approx(321.0, abs=0.2)
    assert data.heights_at(np.array([7.0]), np.array([46.0]))[0] == pytest.approx(321.0, abs=0.2)
    assert data.height_at(-120.0, -60.0) is None  # nothing cached there
    assert data.height_at(7.0, 89.99) is None and data.height_at(7.0, -89.99) is None
    assert np.isnan(data.heights_at(np.array([180.0, -180.0]), np.array([0.0, 0.0]))).all()


class _Dem(DemSource):
    def __init__(self, sid, coverage=None):
        self.id = self.name = sid
        self.coverage = coverage


def test_dem_sources_skipped_outside_coverage_and_where_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(DEM_SOURCES, "test_td_regional", _Dem("test_td_regional", "test_td_square"))
    cache = TileCache(tmp_path)
    far = (Z, *(int(v) for v in lonlat_to_tile(9.0, 47.0, Z)))  # outside the coverage
    for key in (KEY, far):
        cache.write("test_td_regional", *key, "png", encode_heightmap(np.full((N, N), 1.0)))
        cache.write("test_td_world", *key, "png", encode_heightmap(np.full((N, N), 2.0)))
    cache.mark_missing("test_td_first", *far)
    ids = ["test_td_first", "test_td_regional", "test_td_world"]
    data = TerrainData(cache, ids, IMAGERY, plan())
    # (the regional heightmap would be wrong there: the coverage decides, not the cache)
    assert np.nanmax(data.heightmap_for(far).heights) == pytest.approx(2.0, abs=0.2)
    assert data.heightmap_for(far).source == far
    assert data.heightmap_for((Z + 1, far[1] * 2, far[2] * 2)).source == far  # an ancestor


# --- textures --------------------------------------------------------------------------------
def node_children(key, d=2):
    z, x, y = key
    return [((x << d) + i, (y << d) + j) for j in range(1 << d) for i in range(1 << d)]


def test_texture_sources(tmp_path):
    cache = TileCache(tmp_path)
    for cx, cy in node_children(KEY):
        cache.write(IMAGERY.id, Z + 2, cx, cy, "png", png((10, 200, 30)))
    data = TerrainData(cache, "dem", IMAGERY, plan(imagery={Z + 2: np.array(node_children(KEY))}))
    tex = data.texture_for(KEY)
    assert tex.shape == (1024, 1024, 3) and tuple(tex[500, 500]) == (10, 200, 30)
    assert data.texture_for(KEY, "topo") is None  # not configured
    assert data.texture_for(KEY, "water") is None and data.texture_for(KEY, "borders") is None
    assert data.source_ready("borders") and data.source_ready("water")
    calls = []
    data.water = SimpleNamespace(mask=lambda *k: calls.append(("water", k)) or "mask",
                                 allow_download=True)  # fmt: skip
    data.borders = SimpleNamespace(distance_field=lambda *k: calls.append(("borders", k)) or "df",
                                   allow_download=True, ready=False)  # fmt: skip
    assert data.texture_for(KEY, "water") == "mask" and data.texture_for(KEY, "borders") == "df"
    assert calls == [("water", KEY), ("borders", KEY)]
    assert not data.water.allow_download and not data.borders.allow_download  # not on demand
    assert not data.source_ready("borders")  # still loading


def test_texture_zoom_follows_the_plan_and_the_provider(tmp_path):
    coarse = TileProvider("test_td_coarse", "Coarse", "imagery", ext="png", max_zoom=Z + 1)
    one_finer = {Z + 1: np.array([(TX * 2, TY * 2)])}
    data = TerrainData(TileCache(tmp_path), "dem", IMAGERY, plan(imagery=one_finer))
    assert data.texture_zoom_for(KEY, IMAGERY) == Z + 1  # where the plan has imagery
    assert data.texture_zoom_for((Z, TX + 9, TY), IMAGERY) == Z + 2  # elsewhere: the default
    assert data.texture_zoom_for((Z, TX + 9, TY), coarse) == Z + 1  # capped by the provider


def test_composite_texture_resizes_and_skips_providers(tmp_path):
    """Regional imagery (finer) over a coarser worldwide provider: different texture sizes."""
    regional = TileProvider("test_td_reg", "Reg", "imagery", ext="png", max_zoom=14,
                            coverage="test_td_west")  # fmt: skip
    world = TileProvider("test_td_world_img", "World", "imagery", ext="png", max_zoom=Z + 1)
    cache = TileCache(tmp_path)
    tx, ty = (int(v) for v in lonlat_to_tile(7.0, 46.0, Z))  # straddles the coverage edge
    key = (Z, tx, ty)
    for cx, cy in node_children(key):
        cache.write(regional.id, Z + 2, cx, cy, "png", png((255, 0, 0)))
    for cx, cy in node_children(key, 1):
        cache.write(world.id, Z + 1, cx, cy, "png", png((0, 0, 255)))
    data = TerrainData(cache, "dem", {"imagery": [regional, world]}, plan())
    tex = data.texture_for(key)
    assert tex.shape == (1024, 1024, 3)  # the regional size; the world imagery is scaled up
    west, _, east, _ = tile_bounds_lonlat(*key)
    edge = int((7.0 - west) / (east - west) * 1024)
    assert tuple(tex[512, max(edge - 60, 0)]) == (255, 0, 0)
    assert tuple(tex[512, min(edge + 60, 1023)]) == (0, 0, 255)
    # outside the regional coverage only the world provider is asked
    far = (Z, tx + 20, ty)
    assert data.texture_for(far) is None  # (and nothing cached there at all)
    for cx, cy in node_children(far, 1):
        cache.write(world.id, Z + 1, cx, cy, "png", png((0, 0, 255)))
    assert tuple(data.texture_for(far)[10, 10]) == (0, 0, 255)


def test_on_demand_downloads_only_planned_tiles_until_closed(tmp_path, monkeypatch):
    requests = mock_http(monkeypatch, lambda r: httpx.Response(200, content=png((9, 9, 9))))
    children = node_children(KEY)
    planned = {Z + 2: np.array(children[:2])}
    data = TerrainData(TileCache(tmp_path), "dem", IMAGERY, plan(imagery=planned), on_demand=True)
    tex = data.texture_for(KEY)
    wanted = {f"/{Z + 2}/{x}/{y}.png" for x, y in children[:2]}
    assert {r.url.path for r in requests} == wanted  # only the planned tiles
    assert tuple(tex[5, 5]) == (9, 9, 9)
    data.close()
    n = len(requests)
    data.texture_for((Z, TX + 1, TY))
    assert len(requests) == n and all(d.cancelled for d in data.downloaders.values())


def test_colour_reference_harmonises_the_imagery(tmp_path, monkeypatch):
    """The imagery's large-scale colour follows the reference mosaic (data/harmonize.py)."""
    mock_http(monkeypatch, lambda r: httpx.Response(404))
    source = TileProvider("test_td_src", "Src", "imagery", ext="png", max_requests_per_second=0)
    reference = TileProvider("test_td_ref", "Ref", "imagery", ext="png", max_zoom=4,
                             max_requests_per_second=0)  # fmt: skip
    cache = TileCache(tmp_path)
    rng = np.random.default_rng(1)
    for x in range(8, 14):
        for y in range(8, 14):
            tile = np.clip(200.0 + rng.normal(0.0, 6.0, (256, 256, 3)), 0, 255)
            cache.write(source.id, 5, x, y, "png", png(tile))
    for x in range(3, 8):
        for y in range(3, 8):
            cache.write(reference.id, 4, x, y, "png", png((35, 70, 95)))
    tiles = np.array([(x, y) for x in range(8, 14) for y in range(8, 14)])
    data = TerrainData(cache, "dem", source, plan(imagery={5: tiles}), on_demand=True,
                       colour_reference=(reference, 4, 1.0))  # fmt: skip
    assert reference.id in data.downloaders  # the reference is fetched on demand too
    tex = data.texture_for((4, 5, 5))
    assert np.allclose(tex.reshape(-1, 3).mean(axis=0), (35, 70, 95), atol=5.0)
    assert data.texture_for((4, 5, 5), "topo") is None
