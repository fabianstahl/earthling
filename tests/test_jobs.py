"""Download jobs of a session (the GUI's download dialog and ``earthling fetch``)."""

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin

from earthling.core.geo import lonlat_to_tile
from earthling.data import net
from earthling.data.cache import TileCache
from earthling.data.dem import DemSource, SourceFile, read_heightmap, source_dir
from earthling.data.jobs import _flatten, download_jobs
from earthling.data.providers import TileProvider
from helpers import DEMO, mock_http

PROVIDER = TileProvider(id="test_jobs", name="Test tiles", kind="imagery",
                        url_template="https://tiles.example/{z}/{x}/{y}.png", ext="png",
                        max_requests_per_second=0)  # fmt: skip
AREA = shapely.box(7.0, 46.0, 7.3, 46.2)


def tiles_of(area, z):
    west, south, east, north = area.bounds
    x0, y0 = (int(v) for v in lonlat_to_tile(west, north, z))
    x1, y1 = (int(v) for v in lonlat_to_tile(east, south, z))
    return np.array([(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)])


def fake_session(tmp_path, tiles=None, labels=False):
    """What download_jobs needs of a Session: the plan, provider tiles, labels, cache, AOI."""
    tiles = tiles or {}
    return SimpleNamespace(
        plan=object(),
        cache=TileCache(tmp_path),
        labels=SimpleNamespace(bounds=AREA.bounds) if labels else None,
        aoi=SimpleNamespace(aoi=AREA),
        provider_tiles=lambda kind: tiles.get(kind, []),
    )


def test_jobs_planned_from_the_session(tmp_path):
    from earthling.core.config import Config, Project, ProjectSection, SourcesSection
    from earthling.session import Session

    config = Config(
        project=ProjectSection(cache_dir=tmp_path),
        sources=SourcesSection(imagery=("esri_world_imagery",),
                               topo=("opentopomap", "esri_world_imagery"),
                               dem=("swisstopo_alti3d", "aws_terrarium")),
    )  # fmt: skip
    session = Session(Project(DEMO, config))
    names = [j.name for j in download_jobs(session, {"imagery", "topo", "labels", "dem"})]
    assert names == [
        "Imagery (Esri World Imagery)",
        "Topo (OpenTopoMap)",  # the second topo provider is a fallback (fetched on demand)
        "Labels (OpenStreetMap)",
        "DEM sources (swisstopo swissALTI3D (Switzerland))",
        "DEM heightmap tiles (swisstopo swissALTI3D (Switzerland))",
        "DEM heightmap tiles (AWS Terrain Tiles (worldwide, coarse))",  # direct: no files
    ]
    assert [j.name for j in download_jobs(session, {"dem"})][0].startswith("DEM sources")
    assert download_jobs(session, set()) == []


def test_no_plan_no_jobs(tmp_path):
    session = fake_session(tmp_path)
    session.plan = None
    assert download_jobs(session, {"imagery", "labels", "dem"}) == []


def test_flatten_sorts_zooms_and_caps_them():
    tiles = {14: np.array([[5, 6]]), 10: np.array([[1, 2], [3, 4]]), 12: np.array([[7, 8]])}
    assert _flatten(tiles, None) == [(10, 1, 2), (10, 3, 4), (12, 7, 8), (14, 5, 6)]
    assert _flatten(tiles, 12) == [(10, 1, 2), (10, 3, 4), (12, 7, 8)]
    assert _flatten(tiles, 5) == []
    assert all(type(v) is int for v in _flatten(tiles, None)[0])


def test_tile_job_downloads_up_to_max_zoom_and_cancels(tmp_path, monkeypatch):
    requests = mock_http(monkeypatch, lambda r: httpx.Response(200, content=b"tile"))
    tiles = {10: tiles_of(AREA, 10), 12: tiles_of(AREA, 12)}
    session = fake_session(tmp_path, {"imagery": [(PROVIDER, tiles), (PROVIDER, {})]})
    [job] = download_jobs(session, {"imagery"}, max_zoom=10)  # (no job for empty tile sets)
    seen = []
    progress = job.run(seen.append)
    assert progress.downloaded == len(tiles[10]) and progress.failed == 0
    assert {r.url.path.split("/")[1] for r in requests} == {"10"}
    assert seen[-1].finished
    # a cancelled job requests nothing
    requests.clear()
    [job] = download_jobs(session, {"imagery"})
    job.cancel()
    assert job.run(None).downloaded == 0 and requests == []


def test_labels_job_reports_success_and_failure(tmp_path, monkeypatch):
    answers = [
        httpx.Response(200, json={"elements": []}),
        *(httpx.Response(504) for _ in range(net.MAX_RETRIES + 1)),  # (retried)
        httpx.Response(200, json={"elements": [], "remark": "runtime error: timeout"}),
    ]
    requests = mock_http(monkeypatch, lambda r: answers.pop(0))
    session = fake_session(tmp_path, labels=True)
    [job] = download_jobs(session, {"labels"})
    seen = []
    ok = job.run(seen.append)
    assert (ok.downloaded, ok.failed, ok.done) == (1, 0, 1) and ok.bytes > 0
    assert seen == [ok] and requests[0].method == "POST"
    failed = job.run(None)
    assert failed.failed == 1 and failed.done == 1 and "Overpass" in failed.errors[0]
    timeout = job.run(None)  # an Overpass timeout is not an empty area
    assert timeout.failed == 1 and "runtime error" in timeout.errors[0]
    # labels need the session's label data (no detail area: no job)
    assert download_jobs(fake_session(tmp_path), {"labels"}) == []


# --- DEM sources: download the files of the area, then bake heightmap tiles ------------------
def geotiff(value: float, bounds) -> bytes:
    """A small float32 GeoTIFF (EPSG:4326) of a constant height."""
    west, south, east, north = bounds
    n = 60
    transform = from_origin(west, north, (east - west) / n, (north - south) / n)
    profile = dict(driver="GTiff", width=n, height=n, count=1, dtype="float32",
                   crs="EPSG:4326", transform=transform)  # fmt: skip
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(np.full((n, n), value, dtype=np.float32), 1)
        return mem.read()


class FileSource(DemSource):
    id = "test_job_files"
    name = "Test files"
    native_resolution_m = 30.0

    def __init__(self):
        self.listed = 0

    def files_for_bounds(self, bounds, client=None):
        self.listed += 1
        return [
            SourceFile("https://dem.example/a.tif", "a.tif", (6.9, 45.9, 7.4, 46.3)),
            SourceFile("https://dem.example/sea.tif", "sea.tif", (7.1, 46.0, 7.2, 46.1)),
            SourceFile("https://dem.example/far.tif", "far.tif", (20.0, 10.0, 21.0, 11.0)),
        ]


def dem_server(path):
    if path.endswith("/a.tif"):
        return httpx.Response(200, content=geotiff(1234.0, (6.9, 45.9, 7.4, 46.3)))
    return httpx.Response(404)  # e.g. a sea tile that does not exist


def test_dem_jobs_download_the_area_files_then_bake(tmp_path, monkeypatch):
    requests = mock_http(monkeypatch, lambda r: dem_server(r.url.path))
    source = FileSource()
    session = fake_session(tmp_path, {"dem": [(source, {10: tiles_of(AREA, 10)})]})
    download, bake = download_jobs(session, {"dem"})
    assert source.listed == 0  # listing may need the network: only when the job runs
    progress = download.run(None)
    assert (progress.total, progress.downloaded, progress.missing) == (2, 1, 1)
    assert sorted(r.url.path for r in requests) == ["/a.tif", "/sea.tif"]  # not the far one
    folder = source_dir(session.cache, source)
    assert (folder / "a.tif").exists() and (folder / "sea.tif.missing").exists()
    baked = bake.run(None)
    assert baked.downloaded == len(tiles_of(AREA, 10)) and not baked.errors
    z, (x, y) = 10, tiles_of(AREA, 10)[0]
    heights = read_heightmap(session.cache, source.id, z, x, y)
    assert np.nanmedian(heights) == pytest.approx(1234.0, abs=0.2)
    # a second download skips what is there (and what is known to be missing)
    requests.clear()
    assert download.run(None).skipped == 2 and requests == []


def test_cancelling_a_dem_job_stops_both_steps(tmp_path, monkeypatch):
    requests = mock_http(monkeypatch, lambda r: dem_server(r.url.path))
    session = fake_session(tmp_path, {"dem": [(FileSource(), {10: tiles_of(AREA, 10)})]})
    download, bake = download_jobs(session, {"dem"})
    bake.cancel()  # (one event for the source: downloading and baking)
    assert download.run(None).done == 0 and requests == []
    assert bake.run(None).done == 0


def test_regional_source_outside_the_area_lists_nothing(tmp_path, monkeypatch):
    mock_http(monkeypatch, lambda r: pytest.fail(f"unexpected request {r.url}"))

    class Elsewhere(FileSource):
        coverage_area = SimpleNamespace(lonlat=shapely.box(2.0, 48.0, 2.5, 48.5))

        def files_for_bounds(self, bounds, client=None):
            raise AssertionError("no listing for an empty area")

    session = fake_session(tmp_path, {"dem": [(Elsewhere(), {10: tiles_of(AREA, 10)})]})
    download, _ = download_jobs(session, {"dem"})
    progress = download.run(None)
    assert progress.total == 0 and progress.finished


def test_direct_dem_source_is_baked_per_tile(tmp_path):
    class Direct(FileSource):
        direct = True

    session = fake_session(tmp_path, {"dem": [(Direct(), {10: tiles_of(AREA, 10)})]})
    assert [j.name for j in download_jobs(session, {"dem"})] == ["DEM heightmap tiles (Test files)"]


def test_index_of_remotely_listed_sources(tmp_path, monkeypatch):
    """Catalogue sources keep a local index of the downloaded files (for baking offline)."""
    from earthling.data.dem import download_sources, read_index

    class Listed(FileSource):
        listed_remotely = True

    mock_http(monkeypatch, lambda r: dem_server(r.url.path))
    cache = TileCache(tmp_path)
    source = Listed()
    files = source.files_for_bounds(AREA.bounds)[:1]
    download_sources(source, files, cache)
    folder = source_dir(cache, source)
    assert read_index(folder) == files
    assert json.loads((folder / "index.json").read_text())[0]["filename"] == "a.tif"
    assert source.local_files(folder, AREA.bounds) == files
    assert source.local_files(folder, (20.0, 10.0, 21.0, 11.0)) == []
