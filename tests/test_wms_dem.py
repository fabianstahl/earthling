from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pytest

from earthling.core.geo import lonlat_to_tile
from earthling.data.cache import TileCache
from earthling.data.dem import (
    HEIGHTMAP_SAMPLES,
    DemBaker,
    WmsDemSource,
    read_heightmap,
    sample_positions_mercator,
)


class RampWms(WmsDemSource):
    id = name = "test_wms"
    wms_url = "https://wms.example/wms"
    layer = "DEM"
    oversample = 2
    concurrency = 3
    max_requests_per_second = 0.0


def ramp(x, y, x0, y0):
    return 1000.0 + (x - x0) * 0.01 - (y - y0) * 0.02


def ramp_server(calls, fail_first=0, nodata_west=False):
    state = {"fails": fail_first}

    def handler(request: httpx.Request) -> httpx.Response:
        q = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        calls.append(q)
        if state["fails"] > 0:
            state["fails"] -= 1
            return httpx.Response(503)
        if q["LAYERS"] != "DEM" or q["CRS"] != "EPSG:3857":
            return httpx.Response(400)
        x0, y0, x1, y1 = map(float, q["BBOX"].split(","))
        w, h = int(q["WIDTH"]), int(q["HEIGHT"])
        xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w  # pixel centres
        ys = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h
        gx, gy = np.meshgrid(xs, ys)
        values = ramp(gx, gy, 770_000.0, 5_770_000.0).astype("<f4")
        if nodata_west:
            values[:, : w // 4] = -99999.0
        return httpx.Response(
            200, content=values.tobytes(), headers={"content-type": "image/x-bil;bits=32"}
        )

    return handler


def key_near_chamonix(z=12):
    tx, ty = lonlat_to_tile(6.87, 45.92, z)
    return z, int(tx), int(ty)


def test_wms_pixels_are_averaged_onto_the_sample_grid(tmp_path):
    calls = []
    client = httpx.Client(transport=httpx.MockTransport(ramp_server(calls)))
    baker = DemBaker(RampWms(), TileCache(tmp_path), client=client)
    z, x, y = key_near_chamonix()
    heights = baker.bake_tile(z, x, y)
    assert heights.shape == (HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES)
    assert calls[0]["WIDTH"] == str(2 * HEIGHTMAP_SAMPLES)
    xs, ys = sample_positions_mercator(z, x, y)
    gx, gy = np.meshgrid(xs, ys)
    expected = ramp(gx, gy, 770_000.0, 5_770_000.0)
    assert np.abs(heights - expected).max() < 0.01  # exact sample positions (incl. border)


def test_nodata_retries_and_threaded_bake(tmp_path):
    calls = []
    client = httpx.Client(
        transport=httpx.MockTransport(ramp_server(calls, fail_first=2, nodata_west=True))
    )
    cache = TileCache(tmp_path)
    baker = DemBaker(RampWms(), cache, client=client)
    z, x, y = key_near_chamonix()
    tiles = [(z, x + i, y) for i in range(4)]
    import earthling.data.dem as dem

    dem.time.sleep, sleep = (lambda s: None), dem.time.sleep  # no real back-off in tests
    try:
        progress = baker.bake(tiles)
    finally:
        dem.time.sleep = sleep
    assert progress.downloaded == 4 and progress.failed == 0 and progress.done == 4
    assert len(calls) == 6  # two 503s were retried
    heights = read_heightmap(cache, "test_wms", *tiles[0])
    assert np.isnan(heights[:, :60]).all() and np.isfinite(heights[:, 70:]).all()
    assert baker.bake(tiles).skipped == 4  # cached now


def test_not_found_marks_missing(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    cache = TileCache(tmp_path)
    progress = DemBaker(RampWms(), cache, client=client).bake([key_near_chamonix()])
    assert progress.missing == 1
    assert cache.is_missing("test_wms", *key_near_chamonix())


def test_unexpected_answer_is_an_error(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, text="<ServiceException/>", headers={"content-type": "text/xml"}
            )
        )
    )
    baker = DemBaker(RampWms(), TileCache(tmp_path), client=client)
    with pytest.raises(ValueError):
        baker.bake_tile(*key_near_chamonix())
    assert baker.bake([key_near_chamonix()]).failed == 1


def test_ign_sources_registered():
    from earthling.data.dem import DEM_SOURCES
    from earthling.data.providers import PROVIDERS

    ign = PROVIDERS["ign_bdortho"]
    assert ign.coverage == "france" and ign.coverage_area.contains_lonlat(6.87, 45.92)
    assert "TILEMATRIX=15&TILEROW=2&TILECOL=1" in ign.tile_url(15, 1, 2)
    alti = DEM_SOURCES["ign_rgealti"]
    assert alti.direct and alti.coverage == "france" and alti.oversample == 2
