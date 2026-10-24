"""Lines between places for journeys: a straight line, a flight (an arc in the air) or the road
route between two places (OSRM, OpenStreetMap data, online), written as GPX tracks."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from earthling.core.geo import great_circle
from earthling.core.gpx import haversine_m, write_gpx
from earthling.data import net

KINDS = [("straight", "Straight line"), ("flight", "Flight (an arc in the air)"),
         ("road", "Road route (online)")]  # fmt: skip
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"


def arc_heights(points: list[tuple[float, float]]) -> np.ndarray:
    """Flight altitude along a line: a visible arc (5 % of the distance, at least 10 km),
    rising and sinking smoothly."""
    lat = np.array([p[0] for p in points])
    lon = np.array([p[1] for p in points])
    along = np.concatenate([[0.0], np.cumsum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    total = float(along[-1]) or 1.0
    top = max(10_000.0, 0.05 * total)
    return top * np.sin(math.pi * along / total)


def road_route(a: tuple[float, float], b: tuple[float, float],
               cache_dir: Path) -> list[tuple[float, float]]:  # fmt: skip
    """The likely road route from ``a`` to ``b`` (lat, lon), cached in ``cache_dir``; it starts
    and ends exactly at the two places (the route itself snaps to the nearest roads)."""
    coords = f"{a[1]:.5f},{a[0]:.5f};{b[1]:.5f},{b[0]:.5f}"
    cache = Path(cache_dir) / (hashlib.md5(coords.encode()).hexdigest()[:12] + ".json")
    if not cache.exists():
        with net.client() as client:
            params = {"overview": "full", "geometries": "geojson"}
            response = net.get(client, OSRM_URL + coords, params=params)
            response.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(response.text, encoding="utf-8")
    route = json.loads(cache.read_text(encoding="utf-8"))["routes"][0]
    return [tuple(a), *((lat, lon) for lon, lat in route["geometry"]["coordinates"]), tuple(b)]


def make_line(path: Path, name: str, kind: str, a: tuple[float, float], b: tuple[float, float],
              start: datetime, end: datetime, cache_dir: Path) -> Path:  # fmt: skip
    """Write the line from ``a`` to ``b`` (lat, lon) as a GPX track with times from ``start``
    to ``end`` (aware datetimes, spread by distance)."""
    if kind == "road":
        points = road_route(a, b, Path(cache_dir) / "_routes")
        return write_gpx(path, name, points, None, start, end)
    points = great_circle(a, b, step_km=2.0)
    heights = arc_heights(points) if kind == "flight" else None
    return write_gpx(path, name, points, heights, start, end)
