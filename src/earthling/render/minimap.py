"""Mini-map overlay: the whole journey on a small map in a corner of the picture.

The route of all walked tracks (the drawn part in the track colours, the rest faint) and a dot at
the head, on land and water from the OpenStreetMap water polygons (the same vector tiles as the
water surface) and the country borders. Web Mercator, north up, the route's bounds with a margin.
The geometry is built once per path (normalised 0..1 coordinates) and only scaled per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from earthling.core.geo import lonlat_to_mercator

MARGIN = 0.1  # of the route's extent, on every side


@dataclass
class MinimapGeometry:
    aspect: float  # height / width of the map
    routes: list[tuple[int, np.ndarray, np.ndarray]]  # (track index, xy (n, 2), global distance)
    water: np.ndarray  # triangles (n, 3, 2), normalised
    borders: np.ndarray  # segments (n, 2, 2), normalised


def _to_unit(mx, my, bounds):
    x0, y0, x1, y1 = bounds
    return np.column_stack([(mx - x0) / (x1 - x0), (y1 - my) / (y1 - y0)])  # y down


def build(path, frame, water=None, borders=None, max_aspect: float = 1.4) -> MinimapGeometry:
    """Geometry of the mini-map for a ProgressPath (``water``: WaterData, ``borders``:
    BorderData, both optional)."""
    routes_ll = []
    for g in path.tracks:
        lat, lon, _ = frame.enu_to_geodetic(np.asarray(g.enu, dtype=np.float64))
        step = max(1, len(lon) // 1500)  # (a few hundred points per track are plenty)
        routes_ll.append((g, np.asarray(lon)[::step], np.asarray(lat)[::step],
                          (g.dist + g.offset_m)[::step]))  # fmt: skip
    lon_all = np.concatenate([r[1] for r in routes_ll])
    lat_all = np.concatenate([r[2] for r in routes_ll])
    mx, my = lonlat_to_mercator(lon_all, lat_all)
    x0, x1, y0, y1 = mx.min(), mx.max(), my.min(), my.max()
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    # margin, and an aspect ratio between landscape and a little higher than wide
    w, h = w * (1 + 2 * MARGIN), h * (1 + 2 * MARGIN)
    if h / w > max_aspect:
        w = h / max_aspect
    if h / w < 0.5:
        h = w * 0.5
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bounds = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    routes = []
    for g, lon, lat, dist in routes_ll:
        rx, ry = lonlat_to_mercator(lon, lat)
        routes.append((g.index, _to_unit(rx, ry, bounds), dist))
    tri = _water_triangles(water, bounds) if water is not None else np.zeros((0, 3, 2))
    seg = _border_segments(borders, bounds) if borders is not None else np.zeros((0, 2, 2))
    return MinimapGeometry(h / w, routes, tri, seg)


def _bounds_lonlat(bounds):
    from earthling.core.geo import mercator_to_lonlat

    x0, y0, x1, y1 = bounds
    west, south = mercator_to_lonlat(x0, y0)
    east, north = mercator_to_lonlat(x1, y1)
    return float(west), float(south), float(east), float(north)


def _water_triangles(water, bounds) -> np.ndarray:
    """Water polygons of the vector tiles covering the map, clipped, simplified, triangulated."""
    import shapely
    from shapely.geometry import Polygon, box

    from earthling.core.geo import MERCATOR_HALF, lonlat_to_tile

    x0, y0, x1, y1 = bounds
    # a zoom with 2 .. 4 tiles across the map
    z = int(np.clip(math.floor(math.log2(2 * MERCATOR_HALF / (x1 - x0) * 3.0)), 0, 14))
    west, south, east, north = _bounds_lonlat(bounds)
    tx0, ty0 = (int(math.floor(v)) for v in lonlat_to_tile(west, north, z))
    tx1, ty1 = (int(math.floor(v)) for v in lonlat_to_tile(east, south, z))
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64:
        return np.zeros((0, 3, 2))
    tile_m = 2 * MERCATOR_HALF / (1 << z)
    tolerance = (x1 - x0) / 500.0  # about a pixel of the map
    view = box(x0, y0, x1, y1)
    out = []
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            polys = water.polygons(z, tx % (1 << z), ty)
            if not polys:
                continue
            left = -MERCATOR_HALF + tx * tile_m
            top = MERCATOR_HALF - ty * tile_m
            tile_box = box(left, top - tile_m, left + tile_m, top).intersection(view)
            if tile_box.is_empty:
                continue
            for extent, rings in polys:
                current, holes = None, []
                shapes = []
                for ring in rings:
                    pts = [(left + px / extent * tile_m, top - py / extent * tile_m)
                           for px, py in ring]  # fmt: skip
                    if len(pts) < 4:
                        continue
                    from earthling.data.vector_tiles import ring_area

                    if ring_area(ring) > 0:  # a new exterior
                        if current is not None:
                            shapes.append(Polygon(current, holes))
                        current, holes = pts, []
                    elif current is not None:
                        holes.append(pts)
                if current is not None:
                    shapes.append(Polygon(current, holes))
                for shape in shapes:
                    shape = shapely.make_valid(shape).intersection(tile_box)
                    shape = shapely.simplify(shape, tolerance)
                    if shape.is_empty or shape.area < tolerance * tolerance:
                        continue
                    tris = shapely.constrained_delaunay_triangles(shape)
                    for t in shapely.get_parts(tris):
                        c = np.asarray(t.exterior.coords)[:3]
                        out.append(_to_unit(c[:, 0], c[:, 1], bounds))
    return np.array(out) if out else np.zeros((0, 3, 2))


def _border_segments(borders, bounds) -> np.ndarray:
    import shapely

    west, south, east, north = _bounds_lonlat(bounds)
    lines = borders.lines_in("countries", (west, south, east, north))
    if not len(lines):
        return np.zeros((0, 2, 2))
    lines = shapely.simplify(shapely.clip_by_rect(lines, west, south, east, north),
                             (east - west) / 600.0)  # fmt: skip
    coords, index = shapely.get_coordinates(lines, return_index=True)
    if len(coords) < 2:
        return np.zeros((0, 2, 2))
    mx, my = lonlat_to_mercator(coords[:, 0], coords[:, 1])
    xy = _to_unit(mx, my, bounds)
    same = index[1:] == index[:-1]
    return np.stack([xy[:-1][same], xy[1:][same]], axis=1)
