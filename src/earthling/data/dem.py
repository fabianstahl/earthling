"""Elevation data: source rasters -> baked Web Mercator heightmap tiles.

Heightmap tile format
---------------------
Each tile stores a grid of ``HEIGHTMAP_SAMPLES`` x ``HEIGHTMAP_SAMPLES`` elevation samples
placed at the *corners* of a 256-interval grid over the tile, plus one border sample on every
side: sample ``i`` (``-1 <= i <= 257``) lies at ``tile_min + i * tile_size / 256``. Samples 0 and
256 coincide with the neighbouring tiles' edge samples, so meshes are seamless; the border is
used for normals. Samples are stored in a 16-bit grayscale PNG as
``elevation_m = value * HEIGHT_SCALE + HEIGHT_OFFSET`` (0.2 m steps, -1000 .. 12107 m).
Value 0 marks "no data".
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling

from earthling.core.geo import (
    MERCATOR_HALF,
    mercator_to_lonlat,
    tile_bounds_lonlat,
    tile_bounds_mercator,
)
from earthling.data.cache import TileCache, write_atomic
from earthling.data.downloader import MAX_RETRIES, DownloadProgress, RateLimiter, http_client
from earthling.data.providers import License

log = logging.getLogger(__name__)

GRID_INTERVALS = 256
HEIGHTMAP_SAMPLES = GRID_INTERVALS + 3  # 259
HEIGHT_SCALE = 0.2
HEIGHT_OFFSET = -1000.0
NODATA_VALUE = 0

ProgressCallback = Callable[[DownloadProgress], None]


# --- encoding ----------------------------------------------------------------------------
def encode_heightmap(heights: np.ndarray) -> bytes:
    values = np.round((heights - HEIGHT_OFFSET) / HEIGHT_SCALE)
    values = np.clip(values, 1, 65535)
    values[~np.isfinite(heights)] = NODATA_VALUE
    img = Image.fromarray(values.astype(np.uint16))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def decode_heightmap(data: bytes) -> np.ndarray:
    """Returns float32 heights, NaN where no data."""
    values = np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)
    heights = values * HEIGHT_SCALE + HEIGHT_OFFSET
    heights[values == NODATA_VALUE] = np.nan
    return heights


def sample_positions_mercator(z: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """Mercator x/y (1-D arrays, west->east and north->south) of the heightmap samples."""
    min_x, min_y, max_x, max_y = tile_bounds_mercator(z, x, y)
    step = (max_x - min_x) / GRID_INTERVALS
    idx = np.arange(-1, GRID_INTERVALS + 2, dtype=np.float64)
    return min_x + idx * step, max_y - idx * step


def sample_spacing_m(z: int, lat: float) -> float:
    """Ground distance between heightmap samples at zoom ``z``."""
    return 2 * MERCATOR_HALF / (2**z) / GRID_INTERVALS * math.cos(math.radians(lat))


# --- source rasters ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFile:
    url: str
    filename: str
    bounds: tuple[float, float, float, float]  # west, south, east, north (lon/lat)


class DemSource:
    """Base class for DEM sources consisting of downloadable raster files."""

    id: str = ""
    name: str = ""
    native_resolution_m: float = 30.0
    license: License = License("unknown", "")
    coverage: str | None = None  # regional sources, see earthling.data.coverage
    feather_m: float = 300.0
    # direct sources serve heightmap tiles per request (e.g. a WMS) instead of source files
    direct: bool = False
    # the file list comes from a remote catalogue: downloads keep a local index for baking
    listed_remotely: bool = False
    concurrency: int = 1
    max_requests_per_second: float = 0.0

    @property
    def coverage_area(self):
        from earthling.data.coverage import get_coverage

        return get_coverage(self.coverage, self.feather_m)

    def files_for_bounds(
        self, bounds: tuple[float, float, float, float], client: httpx.Client | None = None
    ) -> list[SourceFile]:
        raise NotImplementedError

    def local_files(self, folder: Path, bounds: tuple[float, float, float, float]):
        """Downloaded files overlapping ``bounds`` (no network)."""
        if not self.listed_remotely:
            return [f for f in self.files_for_bounds(bounds) if (folder / f.filename).exists()]
        west, south, east, north = bounds
        return [
            f
            for f in read_index(folder)
            if f.bounds[0] < east and f.bounds[2] > west and f.bounds[1] < north
            and f.bounds[3] > south and (folder / f.filename).exists()
        ]  # fmt: skip

    def fetch_tile(self, z: int, x: int, y: int, client: httpx.Client) -> np.ndarray | None:
        """Heightmap samples of one tile (direct sources); None where there is no data."""
        raise NotImplementedError


class WmsDemSource(DemSource):
    """Elevation from a WMS serving raw float32 rasters (``image/x-bil;bits=32``) in
    EPSG:3857. One request per heightmap tile; the request box is widened by 1.5 sample steps
    so the pixel centres are exactly the tile's corner-aligned samples (incl. the border)."""

    direct = True
    concurrency = 4
    max_requests_per_second = 8.0
    wms_url = ""
    layer = ""
    nodata_below = -1000.0  # e.g. -99999 outside the data
    # Request ``oversample`` x the sample resolution and box-average: servers that resample
    # with nearest neighbour snap to their internal grid, which shifts the result by up to a
    # pixel; averaging finer pixels shrinks that error (see ``earthling check-source``).
    oversample: int = 1

    def files_for_bounds(self, bounds, client=None):
        return []

    def request_params(self, z: int, x: int, y: int) -> dict[str, str]:
        min_x, min_y, max_x, max_y = tile_bounds_mercator(z, x, y)
        pad = 1.5 * (max_x - min_x) / GRID_INTERVALS
        bbox = (min_x - pad, min_y - pad, max_x + pad, max_y + pad)
        size = str(HEIGHTMAP_SAMPLES * self.oversample)
        return {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": self.layer,
            "STYLES": "",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(f"{v:.4f}" for v in bbox),
            "WIDTH": size,
            "HEIGHT": size,
            "FORMAT": "image/x-bil;bits=32",
        }

    def fetch_tile(self, z, x, y, client):
        response = client.get(self.wms_url, params=self.request_params(z, x, y))
        if response.status_code in (204, 404):
            return None
        response.raise_for_status()
        n, k = HEIGHTMAP_SAMPLES, self.oversample
        content_type = response.headers.get("content-type", "")
        if "bil" not in content_type or len(response.content) != n * n * k * k * 4:
            raise ValueError(f"unexpected WMS answer ({content_type}): {response.text[:200]}")
        pixels = np.frombuffer(response.content, dtype="<f4").reshape(n * k, n * k)
        pixels = np.where(pixels > self.nodata_below, pixels, np.nan)
        # the centre of every k x k block is exactly one heightmap sample
        heights = pixels.reshape(n, k, n, k).mean(axis=(1, 3)).astype(np.float32)
        return None if np.isnan(heights).all() else heights


class IgnRgeAlti(WmsDemSource):
    id = "ign_rgealti"
    name = "IGN RGE ALTI (France)"
    native_resolution_m = 1.0
    wms_url = "https://data.geopf.fr/wms-r/wms"
    layer = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"
    coverage = "france"
    oversample = 2
    license = License(
        "Licence Ouverte / Open Licence 2.0 (Etalab)",
        "Elevation: © IGN – RGE ALTI®",
        "https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
        commercial_use=True,
    )


class CopernicusGlo30(DemSource):
    id = "copernicus_glo30"
    name = "Copernicus DEM GLO-30"
    native_resolution_m = 30.0
    license = License(
        "Copernicus DEM licence (free, attribution required)",
        "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
        "2014-2018 provided under COPERNICUS by the European Union and ESA",
        "https://spacedata.copernicus.eu/documents/20126/0/CSCDA_ESA_Mission-specific+Annex.pdf",
        commercial_use=True,
    )
    BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

    def files_for_bounds(self, bounds, client=None):
        west, south, east, north = bounds
        files = []
        for lat in range(math.floor(south), math.floor(north) + 1):
            for lon in range(math.floor(west), math.floor(east) + 1):
                ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
                ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                stem = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
                files.append(
                    SourceFile(
                        f"{self.BASE_URL}/{stem}/{stem}.tif",
                        f"{stem}.tif",
                        (lon, lat, lon + 1, lat + 1),
                    )
                )
        return files


class StacDemSource(DemSource):
    """Raster files listed by a STAC API collection (one item per map sheet and edition).

    ``asset_suffix`` selects the asset (e.g. the resolution) and ``sheet_pattern`` extracts
    (edition, sheet) from the item id so only the newest edition of every sheet is used."""

    listed_remotely = True
    items_url = ""
    asset_suffix = ""
    sheet_pattern = re.compile(r"_(\d{4})_([^_]+)$")
    page_size = 100

    def _sheet(self, name: str) -> tuple[str, str]:
        stem = name.removesuffix(self.asset_suffix)
        match = self.sheet_pattern.search(stem)
        return (match.group(1), match.group(2)) if match else ("", stem)

    def files_for_bounds(self, bounds, client=None):
        own = client is None
        client = client or http_client()
        newest: dict[str, tuple[str, SourceFile]] = {}
        try:
            url: str | None = self.items_url
            params: dict | None = {
                "bbox": ",".join(f"{v:.6f}" for v in bounds),
                "limit": self.page_size,
            }
            while url:
                response = client.get(url, params=params)
                response.raise_for_status()
                page = response.json()
                for item in page.get("features", []):
                    for name, asset in item.get("assets", {}).items():
                        if not name.endswith(self.asset_suffix):
                            continue
                        edition, sheet = self._sheet(name)
                        entry = SourceFile(asset["href"], name, tuple(item["bbox"][:4]))
                        if sheet not in newest or edition > newest[sheet][0]:
                            newest[sheet] = (edition, entry)
                url = next(
                    (link["href"] for link in page.get("links", []) if link.get("rel") == "next"),
                    None,
                )
                params = None  # the next link carries the query
        finally:
            if own:
                client.close()
        return [entry for _, entry in newest.values()]

    def local_files(self, folder, bounds):
        files = super().local_files(folder, bounds)
        newest: dict[str, tuple[str, SourceFile]] = {}
        for f in files:
            edition, sheet = self._sheet(f.filename)
            if sheet not in newest or edition > newest[sheet][0]:
                newest[sheet] = (edition, f)
        return [f for _, f in newest.values()]


class SwissAlti3d(StacDemSource):
    id = "swisstopo_alti3d"
    name = "swisstopo swissALTI3D (Switzerland)"
    native_resolution_m = 2.0
    items_url = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
    asset_suffix = "_2_2056_5728.tif"  # 2 m GeoTIFF (COG) in LV95 / LN02
    coverage = "switzerland"
    license = License(
        "swisstopo terms of use for free geodata (open use with source attribution)",
        "Elevation: © swisstopo – swissALTI3D",
        "https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices",
        commercial_use=True,
    )


DEM_SOURCES: dict[str, DemSource] = {
    s.id: s for s in (CopernicusGlo30(), IgnRgeAlti(), SwissAlti3d())
}


def get_dem_source(source_id: str) -> DemSource:
    try:
        return DEM_SOURCES[source_id]
    except KeyError:
        known = ", ".join(sorted(DEM_SOURCES))
        raise KeyError(f"unknown DEM source '{source_id}' (known: {known})") from None


# --- download ----------------------------------------------------------------------------
def source_dir(cache: TileCache, source: DemSource) -> Path:
    return cache.root / "_sources" / source.id


INDEX_NAME = "index.json"


def read_index(folder: Path) -> list[SourceFile]:
    path = folder / INDEX_NAME
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SourceFile(e["url"], e["filename"], tuple(e["bounds"])) for e in data]


def update_index(folder: Path, files: Iterable[SourceFile]) -> None:
    entries = {f.filename: f for f in read_index(folder)}
    entries.update({f.filename: f for f in files})
    data = [
        {"url": f.url, "filename": f.filename, "bounds": list(f.bounds)}
        for f in sorted(entries.values(), key=lambda f: f.filename)
    ]
    write_atomic(folder / INDEX_NAME, json.dumps(data, indent=0).encode("utf-8"))


def download_sources(
    source: DemSource,
    files: list[SourceFile],
    cache: TileCache,
    client: httpx.Client | None = None,
    on_progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> DownloadProgress:
    client = client or http_client()
    folder = source_dir(cache, source)
    folder.mkdir(parents=True, exist_ok=True)
    if source.listed_remotely:
        update_index(folder, files)
    progress = DownloadProgress(total=len(files))
    for f in files:
        if cancel is not None and cancel.is_set():
            break
        target = folder / f.filename
        missing = folder / (f.filename + ".missing")
        if target.exists() or missing.exists():
            progress.skipped += 1
        else:
            try:
                with client.stream("GET", f.url) as response:
                    if response.status_code in (403, 404):
                        missing.touch()  # e.g. ocean tiles do not exist
                        progress.missing += 1
                    else:
                        response.raise_for_status()
                        buf = io.BytesIO()
                        for chunk in response.iter_bytes(1 << 20):
                            if cancel is not None and cancel.is_set():
                                raise InterruptedError
                            buf.write(chunk)
                            progress.bytes += len(chunk)
                            if on_progress:
                                on_progress(progress)
                        write_atomic(target, buf.getvalue())
                        progress.downloaded += 1
            except InterruptedError:
                break
            except httpx.HTTPError as exc:
                progress.failed += 1
                progress.errors.append(f"{f.filename}: {exc}")
        progress.done += 1
        if on_progress:
            on_progress(progress)
    return progress


# --- sampling ----------------------------------------------------------------------------
class _Raster:
    """A source raster (optionally decimated via overviews) held in memory."""

    def __init__(self, path: Path, decimation: int) -> None:
        with rasterio.open(path) as ds:
            height = max(1, ds.height // decimation)
            width = max(1, ds.width // decimation)
            data = ds.read(
                1, out_shape=(height, width), resampling=Resampling.average, masked=True
            ).astype(np.float32)
            self.values = np.ma.filled(data, np.nan)
            scale_x = ds.width / width
            scale_y = ds.height / height
            self.transform = ds.transform @ rasterio.Affine.scale(scale_x, scale_y)
            self.crs = ds.crs
            self.bounds = ds.bounds
        self.inverse = ~self.transform
        self._to_src = None
        if self.crs is not None and self.crs.to_epsg() != 4326:
            self._to_src = Transformer.from_crs(4326, self.crs, always_xy=True)

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Bilinear sampling at lon/lat; NaN outside or where any neighbour is nodata."""
        if self._to_src is not None:
            sx, sy = self._to_src.transform(lon, lat)
        else:
            sx, sy = lon, lat
        col, row = self.inverse @ (np.asarray(sx), np.asarray(sy))
        h, w = self.values.shape
        # Everything within the raster footprint is sampled; within the outermost half pixel
        # the edge value is extended, so adjacent source files join without gaps.
        inside = (col >= 0) & (row >= 0) & (col <= w) & (row <= h)
        out = np.full(np.shape(col), np.nan, dtype=np.float32)
        if not inside.any():
            return out
        col = np.clip(col[inside] - 0.5, 0, w - 1)  # pixel centres
        row = np.clip(row[inside] - 0.5, 0, h - 1)
        c0 = np.minimum(np.floor(col).astype(np.int64), max(w - 2, 0))
        r0 = np.minimum(np.floor(row).astype(np.int64), max(h - 2, 0))
        c1 = np.minimum(c0 + 1, w - 1)
        r1 = np.minimum(r0 + 1, h - 1)
        fc = col - c0
        fr = row - r0
        v = self.values
        top = v[r0, c0] * (1 - fc) + v[r0, c1] * fc
        bottom = v[r1, c0] * (1 - fc) + v[r1, c1] * fc
        out[inside] = top * (1 - fr) + bottom * fr
        return out


class DemBaker:
    """Bakes heightmap tiles from the downloaded source files of a DEM source."""

    def __init__(
        self,
        source: DemSource,
        cache: TileCache,
        max_rasters: int = 6,
        client: httpx.Client | None = None,
    ) -> None:
        self.source = source
        self.cache = cache
        self._client = client
        self._limiter = RateLimiter(source.max_requests_per_second)
        self.folder = source_dir(cache, source)
        self._rasters: OrderedDict[tuple[str, int], _Raster] = OrderedDict()
        self._max_rasters = max_rasters

    def _raster(self, filename: str, decimation: int) -> _Raster:
        key = (filename, decimation)
        raster = self._rasters.get(key)
        if raster is None:
            raster = _Raster(self.folder / filename, decimation)
            self._rasters[key] = raster
            while len(self._rasters) > self._max_rasters:
                self._rasters.popitem(last=False)
        else:
            self._rasters.move_to_end(key)
        return raster

    def _decimation_for(self, z: int, lat: float) -> int:
        spacing = sample_spacing_m(z, lat)
        factor = 1
        while self.source.native_resolution_m * factor * 2 <= spacing and factor < 256:
            factor *= 2
        return factor

    def _fetch_direct(self, z: int, x: int, y: int) -> np.ndarray | None:
        if self._client is None:
            self._client = http_client()
        delay = 1.0
        last: Exception | None = None
        for _attempt in range(MAX_RETRIES + 1):
            self._limiter.wait()
            try:
                return self.source.fetch_tile(z, x, y, self._client)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status < 500 and status != 429:
                    raise
                last = exc
            except httpx.TransportError as exc:
                last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        assert last is not None
        raise last

    def bake_tile(self, z: int, x: int, y: int) -> np.ndarray | None:
        if self.source.direct:
            return self._fetch_direct(z, x, y)
        west, south, east, north = tile_bounds_lonlat(z, x, y)
        margin = (east - west) / GRID_INTERVALS * 2
        files = self.source.local_files(
            self.folder, (west - margin, south - margin, east + margin, north + margin)
        )
        if not files:
            return None
        mx, my = sample_positions_mercator(z, x, y)
        gx, gy = np.meshgrid(mx, my)
        lon, lat = mercator_to_lonlat(gx, gy)
        decimation = self._decimation_for(z, (south + north) / 2)
        heights = np.full(gx.shape, np.nan, dtype=np.float32)
        for f in files:
            todo = np.isnan(heights)
            if not todo.any():
                break
            values = self._raster(f.filename, decimation).sample(lon[todo], lat[todo])
            heights[todo] = values
        if np.isnan(heights).all():
            return None
        return heights

    def bake(
        self,
        tiles: Iterable[tuple[int, int, int]],
        on_progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
        overwrite: bool = False,
    ) -> DownloadProgress:
        tiles = list(tiles)
        progress = DownloadProgress(total=len(tiles))
        sid = self.source.id
        lock = threading.Lock()
        # Coarse-to-fine and spatially sorted, so raster cache hits are likely.
        tiles.sort(key=lambda t: (t[0], t[1] >> 3, t[2] >> 3, t[1], t[2]))

        def work(tile: tuple[int, int, int]) -> None:
            z, x, y = tile
            if cancel is not None and cancel.is_set():
                return
            outcome, size, error = "skipped", 0, ""
            if overwrite or not self.cache.is_known(sid, z, x, y, "png"):
                try:
                    heights = self.bake_tile(z, x, y)
                except Exception as exc:  # rasterio / network errors etc.
                    outcome, error = "failed", f"{z}/{x}/{y}: {exc}"
                    log.warning("baking %s/%s/%s failed: %s", z, x, y, exc)
                else:
                    if heights is None:
                        self.cache.mark_missing(sid, z, x, y)
                        outcome = "missing"
                    else:
                        data = encode_heightmap(heights)
                        self.cache.write(sid, z, x, y, "png", data)
                        outcome, size = "downloaded", len(data)
            with lock:
                setattr(progress, outcome, getattr(progress, outcome) + 1)
                progress.bytes += size
                if error:
                    progress.errors.append(error)
                progress.done += 1
                if on_progress:
                    on_progress(progress)

        workers = self.source.concurrency if self.source.direct else 1
        if workers > 1:
            with ThreadPoolExecutor(workers, thread_name_prefix=f"dem-{sid}") as pool:
                list(pool.map(work, tiles))
        else:
            for tile in tiles:
                if cancel is not None and cancel.is_set():
                    break
                work(tile)
        return progress


def read_heightmap(cache: TileCache, source_id: str, z: int, x: int, y: int) -> np.ndarray | None:
    data = cache.read(source_id, z, x, y, "png")
    return None if data is None else decode_heightmap(data)
