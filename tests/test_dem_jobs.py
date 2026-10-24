import threading

import httpx
import shapely

from earthling.core.geo import lonlat_to_tile
from earthling.data.cache import TileCache
from earthling.data.dem import DemSource, SourceFile, download_sources, source_dir
from earthling.data.jobs import source_area


class FineSource:
    native_resolution_m = 2.0
    coverage_area = None


class CoarseSource:
    native_resolution_m = 30.0
    coverage_area = None


def tiles_around(lon, lat, zooms):
    out = []
    for z in zooms:
        tx, ty = lonlat_to_tile(lon, lat, z)
        out.append((z, int(tx), int(ty)))
    return out


def test_fine_sources_are_only_fetched_for_detailed_tiles():
    area = shapely.box(5.0, 45.0, 9.0, 47.0)
    coarse_tiles = tiles_around(7.0, 46.0, [7, 9, 11])  # ~1.2 km .. 53 m per sample
    fine_tiles = tiles_around(7.5, 46.5, [13])  # ~13 m per sample
    # 2 m: only below the z13 tile (spacing <= 32 m), not the huge low-zoom tiles
    fine = source_area(FineSource(), coarse_tiles + fine_tiles, area)
    assert not fine.is_empty and fine.area < 0.02
    assert fine.intersects(shapely.Point(7.5, 46.5))
    assert source_area(FineSource(), coarse_tiles, area).is_empty
    # 30 m sources are worth it down to ~480 m samples (z9 and finer)
    coarse = source_area(CoarseSource(), coarse_tiles, area)
    assert coarse.area > 0.2


def test_download_sources_failures_timeouts_and_cancel(tmp_path):
    """Server errors and timeouts fail one file (the others go on); a cancel in the middle of
    a file leaves no partial file behind."""

    class Source(DemSource):
        id = "test_download_sources"

    def handler(request):
        name = request.url.path.lstrip("/")
        if name == "broken.tif":
            return httpx.Response(500)
        if name == "slow.tif":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, content=b"\x01" * (1 << 20) + b"\x02" * 1000)  # 2 chunks

    def files(*names):
        return [SourceFile(f"https://dem.example/{n}", n, (0, 0, 1, 1)) for n in names]

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cache = TileCache(tmp_path)
    folder = source_dir(cache, Source())
    done = []
    progress = download_sources(Source(), files("broken.tif", "slow.tif", "ok.tif"), cache, client,
                                on_progress=lambda p: done.append(p.done))  # fmt: skip
    assert (progress.failed, progress.downloaded, progress.done) == (2, 1, 3)
    assert done == [1, 2, 2, 2, 3]  # per file, and per chunk while downloading
    assert "broken.tif" in progress.errors[0] and "timed out" in progress.errors[1]
    assert progress.bytes == (1 << 20) + 1000
    assert (folder / "ok.tif").stat().st_size == progress.bytes
    assert not (folder / "broken.tif").exists() and not (folder / "broken.tif.missing").exists()

    cancel = threading.Event()
    seen = []

    def on_progress(p):
        seen.append(p.bytes)
        cancel.set()  # after the first chunk

    progress = download_sources(Source(), files("big.tif", "next.tif"), cache, client,
                                on_progress=on_progress, cancel=cancel)  # fmt: skip
    assert seen == [1 << 20] and progress.downloaded == 0 and progress.done == 0
    assert not (folder / "big.tif").exists() and not (folder / "next.tif").exists()
