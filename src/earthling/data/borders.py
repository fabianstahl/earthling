"""Country and region borders (Natural Earth 10 m) as distance-field textures per LOD node.

A node's border texture stores, for every texel, the distance to the nearest border line in
*node uv units* (0..1 across the node), clamped to :data:`MAX_DISTANCE_UV`. Channel 0 holds
country borders, channel 1 regional (admin-1) borders. The shader turns distances into
pixel-exact, anti-aliased lines and glows at any zoom level.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import shapely
import shapely.errors
from shapely.geometry import box
from shapely.strtree import STRtree

from earthling.core.geo import lonlat_to_mercator, tile_bounds_lonlat, tile_bounds_mercator
from earthling.data.cache import TileCache, write_atomic
from earthling.data.downloader import http_client

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
DATASETS = {
    "countries": "ne_10m_admin_0_boundary_lines_land",
    "regions": "ne_10m_admin_1_states_provinces_lines",
}
ATTRIBUTION = "Borders: Natural Earth (public domain)"
FIELD_SIZE = 256
MAX_DISTANCE_UV = 0.15  # glow reach: 15 % of a node width
MIN_REGION_ZOOM = 7  # regional borders are too dense to matter on coarser nodes


def dataset_path(cache: TileCache, name: str) -> Path:
    return cache.root / "_vector" / "naturalearth" / f"{DATASETS[name]}.geojson"


def clamped_distance_field(segments: np.ndarray, size: int, max_dist: float) -> np.ndarray:
    """Distance of every texel centre of a (size x size) uv grid to the nearest segment,
    clamped to ``max_dist``. Segments are bucketed per 32x32-texel cell, so only nearby
    segments are tested (distances beyond ``max_dist`` are irrelevant)."""
    cell = 32
    cells = max(1, size // cell)
    step = 1.0 / cells
    out = np.full((size, size), max_dist, dtype=np.float64)
    if len(segments) == 0:
        return out
    lo = np.minimum(segments[:, :2], segments[:, 2:]) - max_dist
    hi = np.maximum(segments[:, :2], segments[:, 2:]) + max_dist
    c = (np.arange(cell) + 0.5) / size
    for cy in range(cells):
        v0 = cy * step
        rows = (lo[:, 1] <= v0 + step) & (hi[:, 1] >= v0)
        if not rows.any():
            continue
        row_segments = segments[rows]
        row_lo, row_hi = lo[rows], hi[rows]
        for cx in range(cells):
            u0 = cx * step
            near = (row_lo[:, 0] <= u0 + step) & (row_hi[:, 0] >= u0)
            if not near.any():
                continue
            gu, gv = np.meshgrid(u0 + c, v0 + c)
            pts = np.column_stack([gu.ravel(), gv.ravel()])
            d = point_segment_distances(pts, row_segments[near]).reshape(cell, cell)
            ys, xs = cy * cell, cx * cell
            out[ys : ys + cell, xs : xs + cell] = np.minimum(d, max_dist)
    return out


def point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Minimum distance of every point (N, 2) to a set of segments (M, 4) -> (N,)."""
    best = np.full(len(points), np.inf)
    if len(segments) == 0:
        return best
    chunk = max(1, 4_000_000 // max(len(points), 1))
    px, py = points[:, 0:1], points[:, 1:2]
    for start in range(0, len(segments), chunk):
        s = segments[start : start + chunk]
        ax, ay, bx, by = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        dx, dy = bx - ax, by - ay
        length2 = np.maximum(dx * dx + dy * dy, 1e-18)
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
        cx, cy = ax + t * dx, ay + t * dy
        d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2).min(axis=1)
        best = np.minimum(best, d)
    return best


class BorderData:
    """Loads border lines (downloading them once) and builds node distance fields."""

    def __init__(self, cache: TileCache, datasets: tuple[str, ...] = ("countries", "regions")):
        self.cache = cache
        self.datasets = datasets
        self._trees: dict[str, tuple[STRtree, np.ndarray] | None] = {}
        self._lock = threading.Lock()
        self._loader: threading.Thread | None = None
        self._ready = threading.Event()
        self.allow_download = True

    # --- loading ----------------------------------------------------------------------
    def ensure_downloaded(self, name: str) -> Path | None:
        path = dataset_path(self.cache, name)
        if path.exists():
            return path
        if not self.allow_download:
            return None
        url = f"{BASE_URL}/{DATASETS[name]}.geojson"
        log.info("downloading %s", url)
        try:
            with http_client() as client:
                response = client.get(url, timeout=120.0)
                response.raise_for_status()
            write_atomic(path, response.content)
        except Exception as exc:  # offline etc.: borders are optional
            log.warning("cannot download %s: %s", url, exc)
            return None
        return path

    @staticmethod
    def _clean(path: Path) -> None:
        """Drop features without geometry (the GEOS JSON reader rejects them)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        data["features"] = [f for f in data.get("features", []) if f.get("geometry")]
        write_atomic(path, json.dumps(data).encode("utf-8"))

    def _load(self, name: str):
        path = self.ensure_downloaded(name)
        if path is None:
            return None
        # C-level GeoJSON parsing; flatten (Multi)LineStrings without Python loops
        try:
            collection = shapely.from_geojson(path.read_text(encoding="utf-8"))
        except shapely.errors.GEOSException:
            self._clean(path)  # once; the cleaned file is kept in the cache
            collection = shapely.from_geojson(path.read_text(encoding="utf-8"))
        parts = shapely.get_parts(shapely.get_parts(collection))
        lines = parts[shapely.get_type_id(parts) == 1]  # LineString
        return STRtree(lines), lines

    def start_loading(self) -> None:
        """Load all datasets in a background thread (idempotent)."""
        with self._lock:
            if self._loader is not None:
                return
            self._loader = threading.Thread(target=self._load_all, daemon=True, name="borders")
            self._loader.start()

    def _load_all(self) -> None:
        for name in self.datasets:
            try:
                tree = self._load(name)
            except Exception as exc:  # corrupt download etc.
                log.warning("cannot load border dataset %s: %s", name, exc)
                tree = None
            with self._lock:
                self._trees[name] = tree
        self._ready.set()

    @property
    def ready(self) -> bool:
        """True once all datasets are loaded (or known to be unavailable)."""
        if not self._ready.is_set():
            self.start_loading()
        return self._ready.is_set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        self.start_loading()
        return self._ready.wait(timeout)

    def _tree(self, name: str):
        self.wait_ready()
        with self._lock:
            return self._trees.get(name)

    def lines_in(self, name: str, bounds: tuple[float, float, float, float]) -> np.ndarray:
        tree = self._tree(name)
        if tree is None:
            return np.empty(0, dtype=object)
        strtree, geoms = tree
        return geoms[strtree.query(box(*bounds))]

    # --- distance fields --------------------------------------------------------------
    def distance_field(self, z: int, x: int, y: int, size: int = FIELD_SIZE) -> np.ndarray | None:
        """(size, size, 2) float32 distances in node uv units, or None without data."""
        min_x, min_y, max_x, max_y = tile_bounds_mercator(z, x, y)
        width = max_x - min_x
        west, south, east, north = tile_bounds_lonlat(z, x, y)
        mw, mh = (east - west) * MAX_DISTANCE_UV * 1.5, (north - south) * MAX_DISTANCE_UV * 1.5
        bounds = (west - mw, south - mh, east + mw, north + mh)
        # half a texel in degrees: finer detail is invisible in this node's texture
        tolerance = (east - west) / size * 0.5
        out = np.full((size, size, 2), MAX_DISTANCE_UV, dtype=np.float32)
        any_data = False
        for channel, name in enumerate(("countries", "regions")):
            if name not in self.datasets or (name == "regions" and z < MIN_REGION_ZOOM):
                continue
            lines = self.lines_in(name, bounds)
            if not len(lines):
                continue
            # vectorised (C level): clip, simplify, flatten to coordinate arrays
            lines = shapely.clip_by_rect(lines, *bounds)
            lines = shapely.simplify(lines, tolerance)
            coords, index = shapely.get_coordinates(lines, return_index=True)
            if len(coords) < 2:
                continue
            mx, my = lonlat_to_mercator(coords[:, 0], coords[:, 1])
            uv = np.column_stack([(mx - min_x) / width, (max_y - my) / width])
            same = index[1:] == index[:-1]  # consecutive points of the same line
            seg = np.hstack([uv[:-1][same], uv[1:][same]])
            if not len(seg):
                continue
            # rows run north -> south, like all node textures
            out[..., channel] = clamped_distance_field(seg, size, MAX_DISTANCE_UV)
            any_data = True
        return out if any_data else None
