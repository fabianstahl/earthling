"""Water mask from OpenStreetMap: sea, lakes and river areas per LOD node.

The polygons come from the OpenFreeMap vector tiles (OpenMapTiles schema, layer ``water``:
the OSM coastline and water areas, up to zoom 14, a few metres accurate). A node's mask is
rasterised from the vector tile of its own zoom (or a crop of the zoom 14 ancestor) with 2x2
supersampling, so the shore is as sharp as the imagery. Channel 0 is the water coverage,
channel 1 a blurred copy ("open water": 0 at the shore, 1 a few hundred metres out), used to
fade the waves and the glint towards the beach.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from earthling.data import net
from earthling.data.cache import TileCache, write_atomic
from earthling.data.providers import License, TileProvider
from earthling.data.vector_tiles import decode_tile, ring_area

log = logging.getLogger(__name__)

TILEJSON_URL = "https://tiles.openfreemap.org/planet"
PROVIDER_ID = "openfreemap"
MAX_ZOOM = 14
MASK_SIZE = 512
SUPERSAMPLE = 2
OPEN_WATER_PX = 6.0  # blur radius of the "open water" channel (mask pixels)
# OpenMapTiles water classes that are water surfaces (swimming pools are not)
WATER_CLASSES = {"ocean", "lake", "river", "pond", "dock", "reservoir", "basin", "lagoon"}
ATTRIBUTION = "Water: OpenFreeMap, © OpenStreetMap contributors"
TILEJSON_MAX_AGE_S = 30 * 86400


def _provider(url_template: str) -> TileProvider:
    return TileProvider(
        id=PROVIDER_ID,
        name="OpenFreeMap vector tiles (water)",
        kind="vector",
        url_template=url_template,
        ext="pbf",
        max_zoom=MAX_ZOOM,
        license=License("ODbL (OpenStreetMap data)", ATTRIBUTION, "https://openfreemap.org"),
        max_requests_per_second=10.0,
        concurrency=4,
    )


class WaterData:
    """Downloads (on demand) and rasterises the water polygons of the vector tiles."""

    def __init__(self, cache: TileCache, classes: set[str] | None = None) -> None:
        """``classes``: the water classes to include (default: all water surfaces; e.g.
        {"ocean"} for the sea only)."""
        self.cache = cache
        self.classes = set(classes) if classes is not None else WATER_CLASSES
        self.allow_download = True
        self._provider: TileProvider | None = None
        self._downloader = None
        self._lock = threading.Lock()
        self._polygons: dict[tuple[int, int, int], list | None] = {}

    # --- source -----------------------------------------------------------------------
    def _tilejson_path(self):
        return self.cache.root / "_vector" / "openfreemap.json"

    def provider(self) -> TileProvider | None:
        """The provider with the current tile URL (from the TileJSON, cached for a month)."""
        with self._lock:
            if self._provider is not None:
                return self._provider
            path = self._tilejson_path()
            doc = None
            if path.exists():
                doc = json.loads(path.read_text(encoding="utf-8"))
                if time.time() - path.stat().st_mtime > TILEJSON_MAX_AGE_S and self.allow_download:
                    doc = self._download_tilejson() or doc
            elif self.allow_download:
                doc = self._download_tilejson()
            if not doc or not doc.get("tiles"):
                return None
            self._provider = _provider(doc["tiles"][0])
            return self._provider

    def _download_tilejson(self) -> dict | None:
        try:
            with net.client() as client:
                response = net.get(client, TILEJSON_URL, timeout=30.0)
                response.raise_for_status()
            doc = response.json()
        except Exception as exc:  # offline: the water mask is optional
            log.warning("cannot load %s: %s", TILEJSON_URL, exc)
            return None
        write_atomic(self._tilejson_path(), json.dumps(doc).encode("utf-8"))
        return doc

    def tile_bytes(self, z: int, x: int, y: int) -> bytes | None:
        provider = self.provider()
        if provider is None:
            return None
        data = self.cache.read(PROVIDER_ID, z, x, y, "pbf")
        if data is not None or not self.allow_download:
            return data
        if self.cache.is_missing(PROVIDER_ID, z, x, y):
            return None
        with self._lock:
            if self._downloader is None:
                from earthling.data.downloader import TileDownloader

                self._downloader = TileDownloader(provider, self.cache)
            downloader = self._downloader
        try:
            downloader.fetch_one(z, x, y)
        except Exception as exc:  # network problems never break rendering
            log.warning("water tile %s/%s/%s: %s", z, x, y, exc)
        return self.cache.read(PROVIDER_ID, z, x, y, "pbf")

    def polygons(self, z: int, x: int, y: int) -> list | None:
        """Water polygons of vector tile (z, x, y): [(extent, [rings])], None if unavailable.
        Rings are lists of (x, y) tile coordinates; the first ring of a polygon is its
        exterior, the following ones (negative area) its holes."""
        key = (z, x, y)
        if key in self._polygons:
            return self._polygons[key]
        data = self.tile_bytes(z, x, y)
        result = None
        if data is not None:
            try:
                layer = decode_tile(data, {"water"}).get("water")
            except Exception as exc:  # corrupt tile
                log.warning("cannot decode water tile %s: %s", key, exc)
                layer = None
            result = []
            if layer is not None:
                for f in layer.features:
                    if not f.is_polygon or f.tags.get("class") not in self.classes:
                        continue
                    if f.tags.get("intermittent") in (1, True):
                        continue
                    result.append((layer.extent, f.rings))
        with self._lock:
            if len(self._polygons) > 256:
                self._polygons.pop(next(iter(self._polygons)))
            self._polygons[key] = result
        return result

    # --- masks ------------------------------------------------------------------------
    def coverage(self, z: int, x: int, y: int, size: int = MASK_SIZE) -> np.ndarray | None:
        """Water coverage (size x size float32, rows north -> south) of node (z, x, y)."""
        zs = min(z, MAX_ZOOM)
        d = z - zs
        sx, sy = x >> d, y >> d
        polys = self.polygons(zs, sx, sy)
        if polys is None:
            return None
        n = 1 << d
        ix, iy = x - (sx << d), y - (sy << d)
        big = size * SUPERSAMPLE
        image = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(image)
        for extent, rings in polys:
            scale = n * big / extent
            for ring in rings:
                if len(ring) < 3:
                    continue
                pts = [((px * scale - ix * big), (py * scale - iy * big)) for px, py in ring]
                draw.polygon(pts, fill=255 if ring_area(ring) > 0 else 0)
        image = image.resize((size, size), Image.Resampling.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0

    def mask(self, z: int, x: int, y: int, size: int = MASK_SIZE) -> np.ndarray | None:
        """(size, size, 2) float32: water coverage and "open water" (blurred coverage)."""
        cover = self.coverage(z, x, y, size)
        if cover is None:
            return None
        image = Image.fromarray((cover * 255.0 + 0.5).astype(np.uint8))  # (no blur of "F")
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(OPEN_WATER_PX)), np.float32)
        open_water = np.clip((blurred / 255.0 - 0.5) * 2.0, 0.0, 1.0)
        return np.dstack([cover, open_water * cover]).astype(np.float32)

    def is_water(self, lon, lat, zoom: int = MAX_ZOOM) -> np.ndarray:
        """Water coverage (0..1) at the given points (vectorised), from zoom ``zoom`` masks."""
        from earthling.core.geo import lonlat_to_tile

        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        tx, ty = lonlat_to_tile(lon, lat, zoom)
        ix, iy = np.floor(tx).astype(np.int64), np.floor(ty).astype(np.int64)
        out = np.zeros(lon.shape)
        for x, y in {(int(a), int(b)) for a, b in zip(ix, iy, strict=True)}:
            sel = (ix == x) & (iy == y)
            cover = self.coverage(zoom, x, y, 256)
            if cover is None:
                continue
            px = np.clip(((tx[sel] - x) * 256).astype(int), 0, 255)
            py = np.clip(((ty[sel] - y) * 256).astype(int), 0, 255)
            out[sel] = cover[py, px]
        return out

    def raster(self, bounds: tuple[float, float, float, float], zoom: int,
               size: int = 256) -> tuple[np.ndarray, tuple[int, int]]:  # fmt: skip
        """Water coverage of all zoom-``zoom`` tiles covering lon/lat ``bounds`` (west, south,
        east, north), stitched: (array, (first tile x, first tile y)). Rows north -> south."""
        from earthling.core.geo import lonlat_to_tile

        west, south, east, north = bounds
        x0, y0 = (int(np.floor(v)) for v in lonlat_to_tile(west, north, zoom))
        x1, y1 = (int(np.floor(v)) for v in lonlat_to_tile(east, south, zoom))
        out = np.zeros(((y1 - y0 + 1) * size, (x1 - x0 + 1) * size), dtype=np.float32)
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                cover = self.coverage(zoom, tx, ty, size)
                if cover is not None:
                    r, c = (ty - y0) * size, (tx - x0) * size
                    out[r : r + size, c : c + size] = cover
        return out, (x0, y0)
