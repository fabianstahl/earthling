import io
from pathlib import Path

import numpy as np
import pytest
import shapely
from PIL import Image

from earthling.core.aoi import TilePlan
from earthling.core.geo import lonlat_to_mercator, lonlat_to_tile, tile_bounds_mercator
from earthling.data.cache import TileCache
from earthling.data.coverage import get_coverage, needed_tiles, register_coverage
from earthling.data.dem import DEM_SOURCES, DemSource, encode_heightmap
from earthling.data.providers import TileProvider
from earthling.data.terrain_data import TerrainData

SQUARE = shapely.box(7.0, 46.0, 7.2, 46.2)
register_coverage("test_square", SQUARE, feather_m=300.0)

REGIONAL = TileProvider("test_regional", "Regional", "imagery", ext="png", max_zoom=14,
                        coverage="test_square")  # fmt: skip
GLOBAL = TileProvider("test_global", "Global", "imagery", ext="png", max_zoom=14)
RED, BLUE = (255, 0, 0), (0, 0, 255)


def png(color):
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buf, format="PNG")
    return buf.getvalue()


def metres_east(lon, lat, m):
    return lon + m / (111_320.0 * np.cos(np.radians(lat)))


def test_weights_follow_distance_to_the_edge():
    cov = get_coverage("test_square")
    lat = 46.1
    lons = [6.99, 7.0, metres_east(7.0, lat, 150.0), metres_east(7.0, lat, 300.0), 7.1]
    mx, my = lonlat_to_mercator(np.array(lons), np.full(5, lat))
    w = [cov.weights(np.array([x, x + 1.0]), np.array([my[0], my[0] - 1.0]))[0, 0] for x in mx]
    assert w[0] == 0.0 and w[1] == pytest.approx(0.0, abs=0.01)
    assert w[2] == pytest.approx(0.5, abs=0.03)
    assert w[3] == pytest.approx(1.0, abs=0.01) and w[4] == 1.0


def test_weights_agree_across_lod_levels():
    """Weights come from world-space distances: parent and child tiles agree."""
    cov = get_coverage("test_square")
    tx, ty = lonlat_to_tile(7.0, 46.1, 12)
    parent = (12, int(tx), int(ty))
    child = (14, parent[1] * 4 + 2, parent[2] * 4 + 1)
    wp = cov.pixel_weights(tile_bounds_mercator(*parent), 1024, 1024)
    wc = cov.pixel_weights(tile_bounds_mercator(*child), 256, 256)
    # child pixel (i, j) centre == parent pixel (2*256 + i, 1*256 + j) centre
    assert np.abs(wp[256:512, 512:768] - wc).max() < 0.03


def test_classification_and_needed_tiles():
    cov = get_coverage("test_square")
    z = 12
    inside = lonlat_to_tile(7.1, 46.1, z)
    edge = lonlat_to_tile(7.0, 46.1, z)
    outside = lonlat_to_tile(6.7, 46.1, z)
    keys = [(int(t[0]), int(t[1])) for t in (inside, edge, outside)]
    kinds = [cov.classify(tile_bounds_mercator(z, *k)) for k in keys]
    assert kinds == ["full", "partial", "none"]
    regional, world = needed_tiles([cov, None], {z: np.array(keys)})
    assert {tuple(t) for t in regional[z]} == set(keys[:2])
    assert {tuple(t) for t in world[z]} == set(keys[1:])  # not needed where fully covered


def node_texture_setup(tmp_path, skip_regional=()):
    cache = TileCache(tmp_path)
    tx, ty = lonlat_to_tile(7.0, 46.1, 12)
    key = (12, int(tx), int(ty))
    children = [(key[1] * 4 + i, key[2] * 4 + j) for j in range(4) for i in range(4)]
    for cx, cy in children:
        cache.write(GLOBAL.id, 14, cx, cy, "png", png(BLUE))
        if (cx, cy) not in skip_regional:
            cache.write(REGIONAL.id, 14, cx, cy, "png", png(RED))
    plan = TilePlan({"imagery": {14: np.array(children)}, "dem": {12: np.array([key[1:]])}})
    data = TerrainData(cache, "none", {"imagery": [REGIONAL, GLOBAL]}, plan)
    return data, key, children


def column_of(key, lon, lat, width=1024):
    min_x, _, max_x, _ = tile_bounds_mercator(*key)
    mx, _ = lonlat_to_mercator(lon, lat)
    return int((mx - min_x) / (max_x - min_x) * width)


def row_of(key, lat, height=1024):
    _, min_y, _, max_y = tile_bounds_mercator(*key)
    _, my = lonlat_to_mercator(7.0, lat)
    return int((max_y - my) / (max_y - min_y) * height)


def test_imagery_is_composited_with_feathered_edges(tmp_path):
    data, key, _ = node_texture_setup(tmp_path)
    tex = data.texture_for(key)
    assert tex.shape == (1024, 1024, 3)
    lat = 46.1
    row = row_of(key, lat)
    outside = tex[row, column_of(key, 6.97, lat)]
    inside = tex[row, column_of(key, 7.02, lat)]
    half = tex[row, column_of(key, metres_east(7.0, lat, 150.0), lat)]
    assert tuple(outside) == BLUE and tuple(inside) == RED
    assert 90 < half[0] < 165 and 90 < half[2] < 165
    # the red/blue profile along the row is monotonic (no seams between imagery tiles)
    reds = tex[row, :, 0].astype(int)
    assert (np.diff(reds) >= -1).all()


def test_missing_regional_tiles_fall_back(tmp_path):
    tx, ty = lonlat_to_tile(7.02, 46.1, 14)
    hole = (int(tx), int(ty))
    data, key, children = node_texture_setup(tmp_path, skip_regional={hole})
    assert hole in children
    tex = data.texture_for(key)
    assert tuple(tex[row_of(key, 46.1), column_of(key, 7.02, 46.1)]) == BLUE


class _TestDem(DemSource):
    def __init__(self, sid, coverage=None):
        self.id = self.name = sid
        self.coverage = coverage


@pytest.fixture
def dem_sources():
    DEM_SOURCES["test_dem_regional"] = _TestDem("test_dem_regional", "test_square")
    DEM_SOURCES["test_dem_global"] = _TestDem("test_dem_global")
    yield
    del DEM_SOURCES["test_dem_regional"], DEM_SOURCES["test_dem_global"]


def test_dem_sources_blend_per_sample(tmp_path, dem_sources):
    cache = TileCache(tmp_path)
    tx, ty = lonlat_to_tile(7.0, 46.1, 12)
    key = (12, int(tx), int(ty))
    regional = np.full((259, 259), 1000.0)
    regional[200:, 200:] = np.nan  # a hole inside the coverage
    cache.write("test_dem_regional", *key, "png", encode_heightmap(regional))
    cache.write("test_dem_global", *key, "png", encode_heightmap(np.full((259, 259), 500.0)))
    plan = TilePlan({"imagery": {}, "dem": {12: np.array([key[1:]])}})
    data = TerrainData(cache, ["test_dem_regional", "test_dem_global"], GLOBAL, plan)
    h = data.heightmap_for(key).heights
    row = row_of(key, 46.1, 256) + 1
    cols = [column_of(key, lon, 46.1, 256) + 1 for lon in (6.97, 7.0, 7.02)]
    assert h[row, cols[0]] == pytest.approx(500.0, abs=0.2)
    assert h[row, cols[2]] == pytest.approx(1000.0, abs=0.2)
    profile = h[row, cols[0] : cols[2] + 1]
    assert (np.diff(profile) >= -0.3).all()  # smooth ramp, no step
    assert h[230, 230] == pytest.approx(500.0, abs=0.2)  # hole filled from the global DEM
    assert np.isfinite(h).all()


def test_shipped_coverages():
    france, swiss = get_coverage("france"), get_coverage("switzerland")
    assert france.contains_lonlat(2.35, 48.85) and france.contains_lonlat(6.87, 45.92)
    assert not france.contains_lonlat(6.14, 46.21)  # Geneva
    assert swiss.contains_lonlat(7.75, 46.02) and swiss.contains_lonlat(6.14, 46.21)
    assert not swiss.contains_lonlat(6.87, 45.92)  # Chamonix


def test_session_plans_tiles_per_provider(tmp_path):
    from earthling.core.config import Config, Project, ProjectSection, SourcesSection
    from earthling.core.session import Session
    from earthling.data.providers import register

    register(TileProvider("test_ch", "Test CH", "imagery", max_zoom=19, coverage="switzerland"))
    config = Config(
        project=ProjectSection(cache_dir=tmp_path),
        sources=SourcesSection(imagery=("test_ch", "esri_world_imagery")),
    )
    session = Session(Project(Path("examples/alps_demo").resolve(), config))
    (ch, ch_tiles), (esri, esri_tiles) = session.provider_tiles("imagery")
    total = session.plan.count("imagery")
    n_ch = sum(len(t) for t in ch_tiles.values())
    n_esri = sum(len(t) for t in esri_tiles.values())
    assert 0 < n_ch < total and 0 < n_esri < total  # the demo hike crosses the border
    assert n_ch + n_esri > total  # tiles along the border are needed from both
    assert "Test CH" in session.provider_report()
    stacks = session.tile_sources()["imagery"]
    assert [p.id for p in stacks] == ["test_ch", "esri_world_imagery"]


def test_needed_tiles_skips_sources_not_used_at_a_zoom():
    keys = np.array([[533, 360], [534, 360]])
    # source 0 (worldwide) is not used at zoom 10: source 1 gets everything there
    fine, fallback = needed_tiles([None, None], {10: keys, 14: keys}, lambda i, z: i == 1 or z > 12)
    assert 10 not in fine and 14 in fine
    assert len(fallback[10]) == 2 and 14 not in fallback
