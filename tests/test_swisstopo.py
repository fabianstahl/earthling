import httpx
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from earthling.core.geo import lonlat_to_tile, mercator_to_lonlat
from earthling.data.cache import TileCache
from earthling.data.dem import (
    DEM_SOURCES,
    DemBaker,
    SourceFile,
    StacDemSource,
    download_sources,
    read_index,
    sample_positions_mercator,
    source_dir,
)

SUFFIX = "_2_2056_5728.tif"


class FakeStac(StacDemSource):
    id = name = "test_stac"
    items_url = "https://stac.example/collections/dem/items"
    asset_suffix = SUFFIX
    native_resolution_m = 10.0


def item(year, sheet, bbox):
    stem = f"swissalti3d_{year}_{sheet}"
    return {
        "id": stem,
        "bbox": bbox,
        "assets": {
            f"{stem}{SUFFIX}": {"href": f"https://files.example/{stem}{SUFFIX}"},
            f"{stem}_0.5_2056_5728.tif": {"href": f"https://files.example/{stem}_0.5.tif"},
        },
    }


def stac_handler(request):
    if "page=2" in str(request.url):
        return httpx.Response(
            200,
            json={"features": [item(2024, "2570-1100", [7.05, 46.05, 7.06, 46.06])], "links": []},
        )
    assert request.url.params["bbox"] == "7.000000,46.000000,7.100000,46.100000"
    return httpx.Response(200, json={
        "features": [item(2019, "2570-1100", [7.05, 46.05, 7.06, 46.06]),
                     item(2019, "2571-1100", [7.06, 46.05, 7.07, 46.06])],
        "links": [{"rel": "next", "href": "https://stac.example/collections/dem/items?page=2"}],
    })  # fmt: skip


def test_stac_listing_pages_and_keeps_newest_edition():
    client = httpx.Client(transport=httpx.MockTransport(stac_handler))
    files = FakeStac().files_for_bounds((7.0, 46.0, 7.1, 46.1), client=client)
    names = sorted(f.filename for f in files)
    assert names == [f"swissalti3d_2019_2571-1100{SUFFIX}", f"swissalti3d_2024_2570-1100{SUFFIX}"]
    assert all(f.url.startswith("https://files.example/") for f in files)


def test_download_keeps_a_local_index(tmp_path):
    source = FakeStac()
    cache = TileCache(tmp_path)
    box = (7.05, 46.05, 7.06, 46.06)
    files = [
        SourceFile("https://files.example/a", f"swissalti3d_2019_2570-1100{SUFFIX}", box),
        SourceFile("https://files.example/b", f"swissalti3d_2024_2570-1100{SUFFIX}", box),
        SourceFile(
            "https://files.example/c",
            f"swissalti3d_2024_2590-1100{SUFFIX}",
            (7.3, 46.05, 7.31, 46.06),
        ),
    ]
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"tif"))
    )
    progress = download_sources(source, files, cache, client=client)
    assert progress.downloaded == 3
    folder = source_dir(cache, source)
    assert {f.filename for f in read_index(folder)} == {f.filename for f in files}
    local = source.local_files(folder, (7.0, 46.0, 7.1, 46.1))
    assert [f.filename for f in local] == [f"swissalti3d_2024_2570-1100{SUFFIX}"]  # newest only


def plane(e, n):
    return 1500.0 + 0.01 * (e - 2_570_000.0) + 0.02 * (n - 1_100_000.0)


@pytest.fixture
def lv95_raster(tmp_path):
    """A 4 km x 4 km 10 m raster in LV95 with a linear elevation plane."""
    cache = TileCache(tmp_path)
    source = FakeStac()
    folder = source_dir(cache, source)
    folder.mkdir(parents=True)
    west, north, res, n = 2_568_000.0, 1_102_000.0, 10.0, 400
    e = west + (np.arange(n) + 0.5) * res
    nn = north - (np.arange(n) + 0.5) * res
    ge, gn = np.meshgrid(e, nn)
    name = f"swissalti3d_2024_2570-1100{SUFFIX}"
    with rasterio.open(
        folder / name, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
        crs="EPSG:2056", transform=from_origin(west, north, res, res), nodata=-9999.0,
    ) as ds:  # fmt: skip
        ds.write(plane(ge, gn).astype(np.float32), 1)
    to_ll = Transformer.from_crs(2056, 4326, always_xy=True)
    lon0, lat0 = to_ll.transform(west, north - n * res)
    lon1, lat1 = to_ll.transform(west + n * res, north)
    from earthling.data.dem import update_index

    update_index(folder, [SourceFile("x", name, (lon0, lat0, lon1, lat1))])
    return cache, source


def test_baking_reprojects_lv95_rasters(lv95_raster):
    cache, source = lv95_raster
    DEM_SOURCES[source.id] = source
    try:
        to_lv95 = Transformer.from_crs(4326, 2056, always_xy=True)
        lon, lat = Transformer.from_crs(2056, 4326, always_xy=True).transform(2_570_000, 1_100_000)
        tx, ty = lonlat_to_tile(lon, lat, 15)
        z, x, y = 15, int(tx), int(ty)
        heights = DemBaker(source, cache).bake_tile(z, x, y)
        assert heights is not None and np.isfinite(heights).all()
        mx, my = sample_positions_mercator(z, x, y)
        gx, gy = np.meshgrid(mx, my)
        slon, slat = mercator_to_lonlat(gx, gy)
        e, n = to_lv95.transform(slon, slat)
        assert np.abs(heights - plane(e, n)).max() < 0.05
    finally:
        del DEM_SOURCES[source.id]


def test_swisstopo_sources_registered():
    from earthling.data.providers import PROVIDERS

    image = PROVIDERS["swisstopo_swissimage"]
    assert image.coverage == "switzerland" and image.tile_url(15, 1, 2).endswith("/15/1/2.jpeg")
    alti = DEM_SOURCES["swisstopo_alti3d"]
    assert alti.listed_remotely and not alti.direct and alti.asset_suffix == SUFFIX
