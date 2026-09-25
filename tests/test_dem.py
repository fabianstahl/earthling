import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from earthling.core.geo import lonlat_to_tile, mercator_to_lonlat
from earthling.data.cache import TileCache
from earthling.data.dem import (
    HEIGHTMAP_SAMPLES,
    DemBaker,
    DemSource,
    SourceFile,
    decode_heightmap,
    encode_heightmap,
    read_heightmap,
    sample_positions_mercator,
    source_dir,
)


def height_fn(lon, lat):
    return 1000.0 + 300.0 * (lon - 6.0) + 200.0 * (lat - 45.0)


class FakeSource(DemSource):
    id = "fake"
    name = "Fake"
    native_resolution_m = 90.0

    def files_for_bounds(self, bounds):
        west, south, east, north = bounds
        out = []
        for lon in range(int(np.floor(west)), int(np.floor(east)) + 1):
            if lon in (6, 7):
                out.append(SourceFile("", f"t{lon}.tif", (lon, 45, lon + 1, 46)))
        return out


@pytest.fixture
def baker(tmp_path):
    cache = TileCache(tmp_path)
    folder = source_dir(cache, FakeSource())
    folder.mkdir(parents=True)
    n = 1200  # 3 arcsec
    res = 1.0 / n
    for lon0 in (6, 7):
        centres_lon = lon0 + (np.arange(n) + 0.5) * res
        centres_lat = 46 - (np.arange(n) + 0.5) * res
        glon, glat = np.meshgrid(centres_lon, centres_lat)
        data = height_fn(glon, glat).astype(np.float32)
        profile = dict(driver="GTiff", width=n, height=n, count=1, dtype="float32")
        profile.update(crs="EPSG:4326", transform=from_origin(lon0, 46, res, res), nodata=-9999)
        with rasterio.open(folder / f"t{lon0}.tif", "w", **profile) as ds:
            ds.write(data, 1)
    return DemBaker(FakeSource(), cache)


def test_encode_decode_roundtrip():
    h = np.linspace(-500, 8000, HEIGHTMAP_SAMPLES**2).reshape(HEIGHTMAP_SAMPLES, -1)
    h[0, 0] = np.nan
    out = decode_heightmap(encode_heightmap(h))
    assert np.isnan(out[0, 0])
    assert np.nanmax(np.abs(out - h)) <= 0.1 + 2e-3  # float32 at 8 km


def test_bake_matches_source_and_crosses_file_seam(baker):
    z = 12
    tx, ty = lonlat_to_tile(7.0, 45.5, z)  # tile containing the 7 deg seam
    x, y = int(tx), int(ty)
    heights = baker.bake_tile(z, x, y)
    assert heights.shape == (HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES)
    assert not np.isnan(heights).any()
    mx, my = sample_positions_mercator(z, x, y)
    lon, lat = mercator_to_lonlat(*np.meshgrid(mx, my))
    assert np.abs(heights - height_fn(lon, lat)).max() < 0.5


def test_neighbouring_tiles_share_edges(baker):
    z = 11
    tx, ty = lonlat_to_tile(6.5, 45.5, z)
    x, y = int(tx), int(ty)
    progress = baker.bake([(z, x, y), (z, x + 1, y), (z, x, y + 1)])
    assert progress.downloaded == 3
    a = read_heightmap(baker.cache, "fake", z, x, y)
    right = read_heightmap(baker.cache, "fake", z, x + 1, y)
    below = read_heightmap(baker.cache, "fake", z, x, y + 1)
    # sample index i is stored at array index i + 1
    assert np.allclose(a[:, 257], right[:, 1], atol=0.21)
    assert np.allclose(a[257, :], below[1, :], atol=0.21)


def test_low_zoom_uses_decimation_and_outside_is_missing(baker):
    assert baker._decimation_for(8, 45.5) > 1
    progress = baker.bake([(12, 0, 0)])  # far away from the sources
    assert progress.missing == 1
    assert baker.cache.is_missing("fake", 12, 0, 0)
    # skipped on second run
    assert baker.bake([(12, 0, 0)]).skipped == 1


def test_copernicus_file_names():
    from earthling.data.dem import CopernicusGlo30

    files = CopernicusGlo30().files_for_bounds((6.5, 45.5, 7.2, 46.1))
    names = sorted(f.filename for f in files)
    assert names[0] == "Copernicus_DSM_COG_10_N45_00_E006_00_DEM.tif"
    assert len(names) == 4


def test_terrain_data_height_query(baker):
    from earthling.core.aoi import TilePlan
    from earthling.data.providers import TileProvider
    from earthling.data.terrain_data import TerrainData

    z = 11
    tx, ty = lonlat_to_tile(6.5, 45.5, z)
    baker.bake([(z, int(tx), int(ty))])
    plan = TilePlan({"dem": {z: np.array([[int(tx), int(ty)]])}, "imagery": {}})
    data = TerrainData(baker.cache, "fake", TileProvider(id="i", name="i", kind="imagery"), plan)
    assert data.height_at(6.5, 45.5) == pytest.approx(height_fn(6.5, 45.5), abs=0.5)
    # deeper zoom falls back to the ancestor heightmap
    assert data.height_at(6.5, 45.5, z=14) == pytest.approx(height_fn(6.5, 45.5), abs=0.5)
    assert data.height_at(20.0, 10.0) is None


def test_vectorised_heights_match_scalar(baker):
    from earthling.core.aoi import TilePlan
    from earthling.data.providers import TileProvider
    from earthling.data.terrain_data import TerrainData

    z = 11
    tiles = []
    for lon in (6.4, 6.6):
        tx, ty = lonlat_to_tile(lon, 45.5, z)
        tiles.append((int(tx), int(ty)))
    baker.bake([(z, x, y) for x, y in tiles])
    plan = TilePlan({"dem": {z: np.array(tiles)}, "imagery": {}})
    data = TerrainData(baker.cache, "fake", TileProvider(id="i", name="i", kind="imagery"), plan)
    lon = np.array([6.4, 6.45, 6.6, 20.0])
    lat = np.array([45.5, 45.52, 45.49, 10.0])
    h = data.heights_at(lon, lat)
    for i in range(3):
        assert h[i] == pytest.approx(data.height_at(lon[i], lat[i]), abs=1e-3)
    assert np.isnan(h[3])
