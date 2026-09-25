"""Quadtree level-of-detail selection for terrain (pure logic, no GL).

Nodes are XYZ tiles ``(z, x, y)``. A node is refined into its four children when its imagery
texels would appear larger than ``pixel_threshold`` pixels on screen, provided finer data
exists (the :class:`NodeSet` knows which tiles the tile plan contains). Children outside the
plan are still drawn (using ancestor data) but never refined further.

The selection only refines a node when all four children are *ready* (GPU resources loaded);
otherwise the node itself is drawn and the children are requested, so there are never holes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from pyglm import glm

from earthling.core.geo import MERCATOR_HALF, LocalFrame, geodetic_to_ecef, tile_to_lonlat

TileKey = tuple[int, int, int]
TEXELS_PER_NODE = 1024  # imagery texture edge length per node


def children(key: TileKey) -> list[TileKey]:
    z, x, y = key
    return [(z + 1, 2 * x + dx, 2 * y + dy) for dy in (0, 1) for dx in (0, 1)]


def parent(key: TileKey) -> TileKey:
    z, x, y = key
    return (z - 1, x >> 1, y >> 1)


class NodeSet:
    """Which nodes exist (i.e. may be refined into) at every zoom level."""

    def __init__(self, levels: dict[int, Iterable[tuple[int, int]]]) -> None:
        self.levels: dict[int, set[tuple[int, int]]] = {
            z: {(int(x), int(y)) for x, y in tiles} for z, tiles in levels.items()
        }
        self.min_zoom = min(self.levels) if self.levels else 0
        self.max_zoom = max(self.levels) if self.levels else 0

    def __contains__(self, key: TileKey) -> bool:
        z, x, y = key
        return (x, y) in self.levels.get(z, ())

    def roots(self) -> list[TileKey]:
        return sorted((self.min_zoom, x, y) for x, y in self.levels.get(self.min_zoom, ()))

    @classmethod
    def from_plan(cls, dem_levels, imagery_levels, texel_zoom_offset: int = 2) -> NodeSet:
        """Nodes = DEM tiles plus the ancestors (``texel_zoom_offset`` up) of imagery tiles."""
        levels: dict[int, set[tuple[int, int]]] = {}
        for z, tiles in dem_levels.items():
            levels.setdefault(z, set()).update((int(x), int(y)) for x, y in tiles)
        for zi, tiles in imagery_levels.items():
            z = zi - texel_zoom_offset
            if z < 0:
                continue
            levels.setdefault(z, set()).update(
                (int(x) >> texel_zoom_offset, int(y) >> texel_zoom_offset) for x, y in tiles
            )
        # make the set closed under parents so refinement can always reach every node
        for z in sorted(levels, reverse=True):
            if z - 1 >= min(levels):
                levels.setdefault(z - 1, set()).update((x >> 1, y >> 1) for x, y in levels[z])
        return cls(levels)


@dataclass
class NodeBounds:
    center: np.ndarray  # ENU, float64
    radius: float
    tile_width_m: float


def node_bounds(frame: LocalFrame, key: TileKey, min_h: float, max_h: float) -> NodeBounds:
    z, x, y = key
    u = np.array([0.0, 0.5, 1.0])
    tx, ty = np.meshgrid(x + u, y + u)
    lon, lat = tile_to_lonlat(tx.ravel(), ty.ravel(), z)
    pts = np.concatenate(
        [
            frame.ecef_to_enu(geodetic_to_ecef(lat, lon, min_h)),
            frame.ecef_to_enu(geodetic_to_ecef(lat, lon, max_h)),
        ]
    )
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    lat_c = float(lat[4])
    width = 2 * MERCATOR_HALF / 2**z * math.cos(math.radians(lat_c))
    return NodeBounds(center, radius, width)


def frustum_planes(view_proj: glm.mat4) -> np.ndarray:
    """Six normalised planes (a, b, c, d) of a camera-relative view-projection matrix."""
    m = np.array(view_proj.to_list(), dtype=np.float64).T  # row-major
    rows = [m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]]
    planes = np.array(rows)
    planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    return planes


def sphere_visible(planes: np.ndarray, center_rel: np.ndarray, radius: float) -> bool:
    d = planes[:, :3] @ center_rel + planes[:, 3]
    return bool(np.all(d >= -radius))


# Geomorphing: a node drawn in place of its parent blends from the parent's surface to its
# own heights while the parent's screen error grows from the split threshold (ratio 1) to
# MORPH_END. The child itself splits at ratio 2, so it is fully morphed by then.
MORPH_END = 1.6


@dataclass
class Selection:
    draw: list[TileKey] = field(default_factory=list)
    request: list[TileKey] = field(default_factory=list)  # most important first
    morph: dict[TileKey, float] = field(default_factory=dict)  # 0 = parent surface, 1 = own


@dataclass
class LodParams:
    pixel_threshold: float = 1.0  # max screen pixels per imagery texel
    max_nodes: int = 600


def select_nodes(
    nodes: NodeSet,
    bounds_of: Callable[[TileKey], NodeBounds],
    is_ready: Callable[[TileKey], bool],
    camera_pos: np.ndarray,
    planes: np.ndarray,
    pixels_per_radian: float,
    params: LodParams | None = None,
) -> Selection:
    """``pixels_per_radian`` = viewport_height / (2 * tan(fov_y / 2))."""
    params = params or LodParams()
    sel = Selection()
    requests: list[tuple[float, TileKey]] = []

    def screen_texel(key: TileKey, b: NodeBounds) -> float:
        dist = max(1.0, float(np.linalg.norm(b.center - camera_pos)) - b.radius)
        return b.tile_width_m / TEXELS_PER_NODE * pixels_per_radian / dist

    def visit(key: TileKey, parent_ratio: float | None = None) -> None:
        b = bounds_of(key)
        if not sphere_visible(planes, b.center - camera_pos, b.radius):
            return
        texel = screen_texel(key, b)
        kids = children(key)
        refinable = key in nodes and any(k in nodes for k in kids)
        if refinable and texel > params.pixel_threshold and len(sel.draw) < params.max_nodes:
            missing = [k for k in kids if not is_ready(k)]
            if not missing:
                ratio = texel / max(params.pixel_threshold, 1e-9)
                for k in kids:
                    visit(k, ratio)
                return
            for k in missing:
                requests.append((-texel, k))
        sel.draw.append(key)
        if parent_ratio is not None:
            t = min(max((parent_ratio - 1.0) / (MORPH_END - 1.0), 0.0), 1.0)
            sel.morph[key] = t * t * (3.0 - 2.0 * t)

    for root in nodes.roots():
        if is_ready(root):
            visit(root)
        else:
            requests.append((-math.inf, root))
    requests.sort()
    sel.request = [k for _, k in requests]
    return sel
