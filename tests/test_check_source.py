"""``earthling check-source``: what the reports say about broken, shifted or missing data."""

import hashlib
import io
import math
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import shapely
from PIL import Image

from earthling.core.geo import MERCATOR_HALF
from earthling.data.cache import TileCache
from earthling.data.check import check_dem, check_imagery, default_location
from earthling.data.coverage import register_coverage
from earthling.data.dem import DemSource, SourceFile, WmsDemSource
from earthling.data.providers import TileProvider
from test_jobs import geotiff

LON, LAT = 6.87, 45.92


def image_bytes(size: int) -> bytes:
    rng = np.random.default_rng(2)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_imagery_failures_are_problems():
    placeholder = frozenset({hashlib.md5(b"no data").hexdigest()})
    provider = TileProvider("test_check_img", "Check", "imagery",
                            "https://tiles.example/{z}/{x}/{y}.png", ext="png",
                            placeholder_md5=placeholder)  # fmt: skip

    def handler(request):
        z = int(request.url.path.split("/")[1])
        if z == 10:
            raise httpx.ConnectTimeout("connect timed out", request=request)
        if z == 13:
            return httpx.Response(200, content=b"no data")
        if z == 16:
            return httpx.Response(200, content=b"<html>quota exceeded</html>")
        return httpx.Response(200, content=image_bytes(512))  # z19: the wrong tile size

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = check_imagery(provider, LON, LAT, client=client)  # zooms 10, 13, 16, 19
    assert not report.ok and len(report.problems) == 4
    timeout, placeholder, not_image, size = report.problems
    assert timeout.startswith("z10") and "ConnectTimeout" in timeout
    assert "known placeholder" in placeholder
    assert "not an image" in not_image and "quota exceeded" in not_image
    assert "tile size 512 px, expected 256" in size
    assert "coverage: worldwide" in report.text() and report.text().endswith("4 problem(s)")


# --- DEM: a bumpy WMS terrain, optionally shifted, biased per zoom or failing ----------------
def bumps(mx, my):
    return 1500.0 + 40.0 * np.sin(mx / 90.0) + 30.0 * np.cos(my / 70.0)


class BumpyWms(WmsDemSource):
    concurrency = 1
    max_requests_per_second = 0.0
    layer = "DEM"

    def __init__(self, sid, shift_m=0.0, bias_at=None, status_at=None):
        self.id = self.name = sid
        self.wms_url = f"https://{sid}.example/wms"
        self.shift_m = shift_m
        self.bias_at = bias_at or {}  # zoom -> metres added
        self.status_at = status_at or {}  # zoom -> HTTP status

    def handle(self, q):
        x0, y0, x1, y1 = map(float, q["BBOX"].split(","))
        z = round(math.log2(2 * MERCATOR_HALF / (x1 - x0)))
        if z in self.status_at:
            return httpx.Response(self.status_at[z])
        w, h = int(q["WIDTH"]), int(q["HEIGHT"])
        xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
        ys = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h
        gx, gy = np.meshgrid(xs, ys)
        values = bumps(gx - self.shift_m, gy) + self.bias_at.get(z, 0.0)
        return httpx.Response(200, content=values.astype("<f4").tobytes(),
                              headers={"content-type": "image/x-bil;bits=32"})  # fmt: skip


def wms_client(*sources):
    by_host = {urlparse(s.wms_url).hostname: s for s in sources}

    def handler(request):
        q = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        return by_host[request.url.host].handle(q)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_shifted_dem_is_reported_as_misaligned(tmp_path):
    reference = BumpyWms("test_check_ref")
    shifted = BumpyWms("test_check_shifted", shift_m=30.0)  # ~1.6 samples at z13
    client = wms_client(shifted, reference)
    report = check_dem(shifted, LON, LAT, TileCache(tmp_path), client=client, zooms=(13,),
                       reference=reference)  # fmt: skip
    assert report.ok, report.text()
    [warning] = report.warnings
    assert warning.startswith("z13: misaligned by ~1.") and "samples" in warning
    assert "vs test_check_ref: shift south +0.00 east -1." in report.text()
    # the same data is aligned
    same = BumpyWms("test_check_same")
    report = check_dem(same, LON, LAT, TileCache(tmp_path), client=wms_client(same, reference),
                       zooms=(13,), reference=reference)  # fmt: skip
    assert not report.warnings and "east +0.00" in report.text() and "(r=1.00)" in report.text()


def test_lod_bias_failures_and_gaps(tmp_path):
    biased = BumpyWms("test_check_biased", bias_at={14: 5.0})
    report = check_dem(biased, LON, LAT, TileCache(tmp_path), client=wms_client(biased),
                       zooms=(12, 13), reference=False)  # fmt: skip
    assert "LOD z13 vs z14: mean -5.00 m" in report.text()
    assert any("biased by -5.0 m" in w for w in report.warnings)
    # z12 refused, z13 has no data: two problems and no LOD comparison
    broken = BumpyWms("test_check_broken", status_at={12: 400, 13: 404})
    report = check_dem(broken, LON, LAT, TileCache(tmp_path), client=wms_client(broken),
                       zooms=(12, 13), reference=False)  # fmt: skip
    assert report.problems[0].startswith("z12: HTTPStatusError")
    assert report.problems[1] == "z13: no data at the location"
    assert "LOD" not in report.text()
    # the children missing: no comparison either
    childless = BumpyWms("test_check_childless", status_at={14: 404})
    report = check_dem(childless, LON, LAT, TileCache(tmp_path), client=wms_client(childless),
                       zooms=(13,), reference=False)  # fmt: skip
    assert report.ok and "LOD" not in report.text()


class FileDem(DemSource):
    id = name = "test_check_files"
    native_resolution_m = 10.0

    def files_for_bounds(self, bounds, client=None):
        return [SourceFile("https://files.example/dem.tif", "dem.tif", (6.8, 45.85, 6.95, 46.0))]


def test_file_sources_are_downloaded_around_the_location(tmp_path):
    def handler(request):
        if request.url.host == "files.example":
            return httpx.Response(200, content=geotiff(1042.0, (6.8, 45.85, 6.95, 46.0)))
        return httpx.Response(404)  # Copernicus, the default reference: not there

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = check_dem(FileDem(), LON, LAT, TileCache(tmp_path), client=client)
    text = report.text()
    assert report.ok, text
    assert "test_check_files: 1 source file(s) for the z13 tile, 1 downloaded, 0 failed" in text
    assert "copernicus_glo30: 1 source file(s) for the z13 tile, 0 downloaded, 0 failed" in text
    assert "z13" in text and "z14" in text and "1042.0" in text  # file sources: z13 and finer
    assert "vs copernicus_glo30" not in text  # no reference data: no comparison


register_coverage("test_check_square", shapely.box(7.0, 46.0, 8.0, 47.0), 0.0)


def test_default_location():
    def session(lon, lat):
        segment = SimpleNamespace(lon=np.asarray(lon, float), lat=np.asarray(lat, float))
        return SimpleNamespace(tracks=[SimpleNamespace(segments=[segment])])

    worldwide = SimpleNamespace(coverage_area=None)
    regional = TileProvider("test_check_regional", "R", "imagery", coverage="test_check_square")
    track = session([6.5, 7.1, 7.5, 7.9], [46.5, 46.5, 46.5, 46.5])
    assert default_location(track, worldwide) == (6.5, 46.5)  # the first point
    assert default_location(SimpleNamespace(tracks=[]), worldwide) == (7.0, 46.0)
    assert default_location(track, regional) == (7.5, 46.5)  # the deepest inside
    lon, lat = default_location(session([2.0], [48.0]), regional)  # none inside
    assert 7.0 < lon < 8.0 and 46.0 < lat < 47.0
