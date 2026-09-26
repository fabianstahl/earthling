"""Coverage areas of regional data providers and feathered blend weights at their edges.

A coverage is a (multi)polygon in lon/lat. Blend weights are computed from the distance to the
coverage edge in metres, in world space, so neighbouring tiles of any LOD level get identical
weights along shared edges: weight 0 at the edge, 1 at ``feather_m`` inside (smoothstep).

Coverage files are GeoJSON files in ``earthling/data/coverage`` (see
``tools/make_coverage.py``), referenced by providers via their file stem.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import shapely

from earthling.core.geo import MERCATOR_HALF, lonlat_to_mercator, mercator_to_lonlat

COVERAGE_DIR = Path(__file__).parent / "coverage"
_REGISTERED: dict[str, Coverage] = {}
COARSE_SAMPLES = 33  # distance evaluations per axis and tile; bilinear in between


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _upsample(coarse: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Bilinear resampling of a grid whose first/last samples coincide with the output's."""
    cy, cx = coarse.shape
    xs = np.linspace(0, cx - 1, nx)
    ys = np.linspace(0, cy - 1, ny)
    rows = np.stack([np.interp(xs, np.arange(cx), row) for row in coarse])
    return np.stack([np.interp(ys, np.arange(cy), col) for col in rows.T], axis=1)


class Coverage:
    def __init__(self, geometry: shapely.Geometry, feather_m: float = 300.0, name: str = ""):
        self.name = name
        self.lonlat = shapely.make_valid(geometry)
        self.feather_m = feather_m
        self.merc = shapely.transform(
            self.lonlat, lambda c: np.column_stack(lonlat_to_mercator(c[:, 0], c[:, 1]))
        )
        # mercator units per metre are largest at the pole-most latitude: a conservative
        # inner area for "fully covered" tests
        max_lat = min(max(abs(self.lonlat.bounds[1]), abs(self.lonlat.bounds[3])), 85.0)
        self.inner = self.merc.buffer(-feather_m / math.cos(math.radians(max_lat)))
        self.boundary = self.merc.boundary
        for geom in (self.merc, self.inner, self.boundary):
            shapely.prepare(geom)

    # --- classification ---------------------------------------------------------------
    def classify(self, bounds_merc: tuple[float, float, float, float]) -> str:
        """'none', 'partial' or 'full' (weight 1 everywhere) for a mercator box."""
        box = shapely.box(*bounds_merc)
        if not self.merc.intersects(box):
            return "none"
        if self.inner.contains(box):
            return "full"
        return "partial"

    def classify_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(touches, full) boolean arrays for an array of shapely boxes (mercator)."""
        return shapely.intersects(self.merc, boxes), shapely.contains(self.inner, boxes)

    def contains_lonlat(self, lon: float, lat: float) -> bool:
        return bool(shapely.contains_xy(self.lonlat, lon, lat))

    # --- weights ----------------------------------------------------------------------
    def weights(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Blend weights on the grid of mercator ``xs`` (columns) x ``ys`` (rows)."""
        nx, ny = len(xs), len(ys)
        cx, cy = min(nx, COARSE_SAMPLES), min(ny, COARSE_SAMPLES)
        gx, gy = np.meshgrid(np.linspace(xs[0], xs[-1], cx), np.linspace(ys[0], ys[-1], cy))
        points = shapely.points(gx.ravel(), gy.ravel())
        distance = shapely.distance(self.boundary, points).reshape(gx.shape)
        inside = shapely.contains_xy(self.merc, gx.ravel(), gy.ravel()).reshape(gx.shape)
        _, lat = mercator_to_lonlat(gx, gy)
        metres = np.where(inside, distance, -distance) * np.cos(np.radians(lat))
        if (cx, cy) != (nx, ny):
            metres = _upsample(metres, ny, nx)
        return _smoothstep(metres / self.feather_m).astype(np.float32)

    def pixel_weights(self, bounds_merc, width: int, height: int) -> np.ndarray:
        """Weights at the pixel centres of an image covering ``bounds_merc``."""
        min_x, min_y, max_x, max_y = bounds_merc
        px = (max_x - min_x) / width
        py = (max_y - min_y) / height
        xs = min_x + (np.arange(width) + 0.5) * px
        ys = max_y - (np.arange(height) + 0.5) * py
        return self.weights(xs, ys)


def register_coverage(name: str, geometry: shapely.Geometry, feather_m: float = 300.0) -> None:
    """Make a coverage available by name (tests, user-defined regions)."""
    _REGISTERED[name] = Coverage(geometry, feather_m, name)
    get_coverage.cache_clear()


@cache
def get_coverage(name: str | None, feather_m: float = 300.0) -> Coverage | None:
    """Coverage by name (registered, or a GeoJSON file in COVERAGE_DIR, or a path)."""
    if name is None:
        return None
    if name in _REGISTERED:
        return _REGISTERED[name]
    path = Path(name)
    if not path.suffix:
        path = COVERAGE_DIR / f"{name}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"coverage '{name}' not found ({path})")
    geometry = shapely.from_geojson(path.read_text(encoding="utf-8"))
    if isinstance(geometry, shapely.GeometryCollection) and not isinstance(
        geometry, shapely.MultiPolygon
    ):
        geometry = shapely.union_all(list(geometry.geoms))
    return Coverage(geometry, feather_m, name)


def tile_boxes_mercator(z: int, xy: np.ndarray) -> np.ndarray:
    size = 2 * MERCATOR_HALF / (1 << z)
    x0 = -MERCATOR_HALF + xy[:, 0] * size
    y1 = MERCATOR_HALF - xy[:, 1] * size
    return shapely.box(x0, y1 - size, x0 + size, y1)


def needed_tiles(coverages: list[Coverage | None], tiles: dict[int, np.ndarray], used_at=None):
    """For providers in priority order: the tiles each one is needed for, i.e. tiles its
    coverage touches that are not fully covered by a higher-priority provider.
    ``used_at(i, z)`` (optional): False where provider i is not used at zoom z at all (e.g. a
    very fine DEM on coarse tiles); it then neither takes nor covers those tiles.
    Returns one {zoom: (n, 2) array} dict per provider."""
    out: list[dict[int, np.ndarray]] = [{} for _ in coverages]
    for z, xy in tiles.items():
        xy = np.asarray(xy).reshape(-1, 2)
        remaining = np.ones(len(xy), dtype=bool)
        boxes = tile_boxes_mercator(z, xy) if any(c is not None for c in coverages) else None
        for i, cov in enumerate(coverages):
            if used_at is not None and not used_at(i, z):
                continue
            if cov is None:
                touches, full = np.ones(len(xy), bool), np.ones(len(xy), bool)
            else:
                touches, full = cov.classify_boxes(boxes)
            take = remaining & touches
            if take.any():
                out[i][z] = xy[take]
            remaining &= ~full
            if not remaining.any():
                break
    return out
