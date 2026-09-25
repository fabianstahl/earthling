"""Area of interest around the tracks and the tile download plan.

The AOI is the union of all tracks buffered by ``border_km``. Buffers are computed in a local
azimuthal-equidistant projection centred on the tracks, so distances are accurate even for
long hikes. Each resolution zone is a buffer of its own distance; a zone's maximum zoom
applies to all tiles intersecting it (the closest zone wins).

The tile plan is a pyramid: for zoom ``z`` it contains every tile intersecting the region
whose allowed maximum zoom is ``>= z``. It is computed top-down by subdividing only tiles that
intersect the relevant region, which keeps it fast for corridor-shaped areas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform

from earthling.core.config import AreaSection
from earthling.core.geo import lonlat_to_tile, tile_to_lonlat
from earthling.core.gpx import Track

MIN_PLAN_ZOOM = 5

# Rough average on-disk sizes used for estimates (bytes per tile).
AVG_TILE_BYTES = {"imagery": 22_000, "dem": 150_000}


@dataclass
class AreaOfInterest:
    """Polygons in lon/lat (EPSG:4326)."""

    aoi: shapely.Geometry
    zones: list[shapely.Geometry]  # cumulative buffers, same order as config zones
    tracks: shapely.Geometry

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(self.aoi.bounds)  # type: ignore[return-value]


def _local_transformers(lon: float, lat: float):
    local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    wgs = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs, local, always_xy=True)
    inv = Transformer.from_crs(local, wgs, always_xy=True)
    return fwd, inv


def compute_aoi(tracks: list[Track], area: AreaSection) -> AreaOfInterest | None:
    lines = []
    for track in tracks:
        for seg in track.segments:
            if len(seg) >= 2:
                lines.append(LineString(np.column_stack([seg.lon, seg.lat])))
    if not lines:
        return None
    geom = MultiLineString(lines)
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    fwd, inv = _local_transformers((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
    local = transform(fwd.transform, geom).simplify(10.0)

    def buffered(km: float) -> shapely.Geometry:
        poly = local.buffer(max(km, 0.001) * 1000.0, quad_segs=16)
        return transform(inv.transform, poly)

    zones = [buffered(z.within_km) for z in area.zones]
    return AreaOfInterest(aoi=buffered(area.border_km), zones=zones, tracks=geom)


def _tile_boxes(z: int, xy: np.ndarray) -> np.ndarray:
    west, north = tile_to_lonlat(xy[:, 0], xy[:, 1], z)
    east, south = tile_to_lonlat(xy[:, 0] + 1, xy[:, 1] + 1, z)
    return shapely.box(west, south, east, north)


@dataclass
class TilePlan:
    """Tiles per kind ('imagery', 'dem') and zoom: arrays of shape (n, 2) with (x, y)."""

    levels: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)

    def count(self, kind: str, z: int | None = None) -> int:
        levels = self.levels.get(kind, {})
        if z is not None:
            return int(len(levels.get(z, ())))
        return int(sum(len(v) for v in levels.values()))

    def zooms(self, kind: str) -> list[int]:
        return sorted(self.levels.get(kind, {}))

    def estimated_bytes(self, kind: str) -> int:
        return self.count(kind) * AVG_TILE_BYTES[kind]

    def tiles(self, kind: str):
        """Iterate (z, x, y) coarse to fine."""
        for z in self.zooms(kind):
            for x, y in self.levels[kind][z]:
                yield z, int(x), int(y)


def _pyramid(regions: list[tuple[shapely.Geometry, int]], min_zoom: int) -> dict[int, np.ndarray]:
    """``regions`` are (polygon, max_zoom) pairs; the union forms the plan."""
    max_zoom = max(z for _, z in regions)
    levels: dict[int, np.ndarray] = {}
    all_geom = shapely.union_all([g for g, _ in regions])
    min_lon, min_lat, max_lon, max_lat = all_geom.bounds
    x0, y0 = lonlat_to_tile(min_lon, max_lat, min_zoom)
    x1, y1 = lonlat_to_tile(max_lon, min_lat, min_zoom)
    xs, ys = np.meshgrid(np.arange(int(x0), int(x1) + 1), np.arange(int(y0), int(y1) + 1))
    candidates = np.column_stack([xs.ravel(), ys.ravel()])
    for z in range(min_zoom, max_zoom + 1):
        region = shapely.union_all([g for g, zmax in regions if zmax >= z])
        shapely.prepare(region)
        hits = shapely.intersects(region, _tile_boxes(z, candidates))
        tiles = candidates[hits]
        levels[z] = tiles
        if z < max_zoom and len(tiles):
            children = np.repeat(tiles * 2, 4, axis=0)
            children += np.tile(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), (len(tiles), 1))
            candidates = children
    return {z: t for z, t in levels.items() if len(t)}


def plan_tiles(aoi: AreaOfInterest, area: AreaSection, min_zoom: int = MIN_PLAN_ZOOM) -> TilePlan:
    plan = TilePlan()
    zones = list(zip(aoi.zones, area.zones, strict=True))
    # The AOI itself uses the outermost zone's resolution.
    outer = area.zones[-1]
    imagery = [(g, z.imagery_zoom) for g, z in zones] + [(aoi.aoi, outer.imagery_zoom)]
    dem = [(g, z.dem_zoom) for g, z in zones] + [(aoi.aoi, outer.dem_zoom)]
    plan.levels["imagery"] = _pyramid(imagery, min(min_zoom, area.max_imagery_zoom))
    plan.levels["dem"] = _pyramid(dem, min(min_zoom, area.max_dem_zoom))
    return plan


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def plan_report(plan: TilePlan) -> str:
    lines = [f"{'zoom':>4}  {'imagery':>10}  {'dem':>10}"]
    zooms = sorted(set(plan.zooms("imagery")) | set(plan.zooms("dem")))
    for z in zooms:
        lines.append(f"{z:>4}  {plan.count('imagery', z):>10}  {plan.count('dem', z):>10}")
    lines.append(f"{'sum':>4}  {plan.count('imagery'):>10}  {plan.count('dem'):>10}")
    lines.append(
        f"estimated size: imagery {format_bytes(plan.estimated_bytes('imagery'))}, "
        f"dem {format_bytes(plan.estimated_bytes('dem'))}"
    )
    return "\n".join(lines)
