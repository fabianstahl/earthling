"""Named features for 3D labels: peaks, passes, places and huts from OpenStreetMap.

Fetched once per area of interest through the Overpass API and cached as JSON. ``enrich`` adds
what OSM rarely has: an elevation for every feature (DEM), a prominence estimate for peaks
and the distance to the tracks, which the label filters and the decluttering use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

from earthling.data.cache import TileCache, write_atomic
from earthling.data.downloader import http_client

log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ATTRIBUTION = "Labels: © OpenStreetMap contributors (ODbL)"
KINDS = ("peak", "pass", "place", "hut")
PLACE_RANK = {"city": 4, "town": 3, "village": 2, "hamlet": 1}
_ELE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*(m|ft)?\s*$")


@dataclass
class LabelFeature:
    osm_id: str
    kind: str  # "peak" | "pass" | "place" | "hut"
    name: str
    lon: float
    lat: float
    ele: float | None = None  # metres (OSM tag, else DEM after enrich)
    rank: int = 0  # places: 1 hamlet .. 4 city
    prominence: float | None = None  # peaks: estimate in metres
    track_distance_m: float = math.inf
    ground: float | None = None  # DEM height at the point (label anchor)

    @property
    def anchor_height(self) -> float | None:
        return self.ground if self.ground is not None else self.ele


def overpass_query(bounds: tuple[float, float, float, float]) -> str:
    west, south, east, north = bounds
    bbox = f"({south:.5f},{west:.5f},{north:.5f},{east:.5f})"
    return (
        "[out:json][timeout:120];("
        f'node["natural"="peak"]["name"]{bbox};'
        f'node["natural"="volcano"]["name"]{bbox};'
        f'node["mountain_pass"="yes"]["name"]{bbox};'
        f'node["natural"="saddle"]["name"]{bbox};'
        f'node["place"~"^(city|town|village|hamlet)$"]["name"]{bbox};'
        f'node["tourism"~"^(alpine_hut|wilderness_hut)$"]["name"]{bbox};'
        f'way["tourism"~"^(alpine_hut|wilderness_hut)$"]["name"]{bbox};'
        ");out center tags;"
    )


def parse_elevation(text: str | None) -> float | None:
    if not text:
        return None
    match = _ELE.match(text.replace(" ", " "))
    if match is None:
        return None
    value = float(match.group(1).replace(",", "."))
    return value * 0.3048 if match.group(2) == "ft" else value


def parse_overpass(data: dict) -> list[LabelFeature]:
    features = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        if "lat" in element:
            lon, lat = element["lon"], element["lat"]
        elif "center" in element:
            lon, lat = element["center"]["lon"], element["center"]["lat"]
        else:
            continue
        if tags.get("natural") in ("peak", "volcano"):
            kind = "peak"
        elif tags.get("mountain_pass") == "yes" or tags.get("natural") == "saddle":
            kind = "pass"
        elif tags.get("place") in PLACE_RANK:
            kind = "place"
        elif tags.get("tourism") in ("alpine_hut", "wilderness_hut"):
            kind = "hut"
        else:
            continue
        features.append(
            LabelFeature(
                osm_id=f"{element.get('type', 'node')}/{element.get('id')}",
                kind=kind,
                name=name,
                lon=float(lon),
                lat=float(lat),
                ele=parse_elevation(tags.get("ele")),
                rank=PLACE_RANK.get(tags.get("place", ""), 0),
            )
        )
    return features


def cache_path(cache: TileCache, bounds) -> Path:
    key = hashlib.sha1(",".join(f"{v:.3f}" for v in bounds).encode()).hexdigest()[:12]
    return cache.root / "_vector" / "osm" / f"labels_{key}.json"


def download_labels(cache: TileCache, bounds, client=None) -> Path:
    """Fetch the features of ``bounds`` (lon/lat) from Overpass into the cache."""
    path = cache_path(cache, bounds)
    own = client is None
    client = client or http_client()
    try:
        response = client.post(OVERPASS_URL, data={"data": overpass_query(bounds)}, timeout=180.0)
        response.raise_for_status()
        data = response.json()
    finally:
        if own:
            client.close()
    write_atomic(path, json.dumps(data).encode("utf-8"))
    return path


# --- enrichment ------------------------------------------------------------------------------
def estimate_prominence(peaks: list[LabelFeature], heights_at, samples: int = 48) -> None:
    """Prominence estimate for peaks with known elevation: the drop to the lowest point on the
    straight line to the nearest higher peak (an upper bound of the true key col drop, but a
    good importance measure). The highest peak gets its drop to the lowest sampled point."""
    known = [p for p in peaks if p.ele is not None]
    if not known:
        return
    lon = np.array([p.lon for p in known])
    lat = np.array([p.lat for p in known])
    ele = np.array([p.ele for p in known], dtype=np.float64)
    coslat = math.cos(math.radians(float(lat.mean())))
    dx = (lon[:, None] - lon[None, :]) * coslat
    dy = lat[:, None] - lat[None, :]
    dist = np.hypot(dx, dy)
    higher = ele[None, :] > ele[:, None]
    dist = np.where(higher, dist, np.inf)
    target = np.argmin(dist, axis=1)
    has_higher = np.isfinite(dist[np.arange(len(known)), target])
    t = np.linspace(0.0, 1.0, samples)[None, :]
    line_lon = lon[:, None] + (lon[target] - lon)[:, None] * t
    line_lat = lat[:, None] + (lat[target] - lat)[:, None] * t
    heights = np.asarray(heights_at(line_lon.ravel(), line_lat.ravel())).reshape(line_lon.shape)
    finite = heights[np.isfinite(heights)]
    lowest = float(finite.min()) if finite.size else float(ele.min())
    col = np.where(np.isfinite(heights), heights, np.inf).min(axis=1)
    for i, peak in enumerate(known):
        if has_higher[i] and np.isfinite(col[i]):
            peak.prominence = max(0.0, float(ele[i] - min(col[i], ele[i])))
        else:
            peak.prominence = max(0.0, float(ele[i] - lowest))


def enrich(features: list[LabelFeature], heights_at=None, frame=None, track_lines=None) -> None:
    """Fill in DEM elevations, peak prominence and distances to the tracks.

    ``heights_at(lon, lat)``: vectorised DEM heights (NaN where unknown). ``track_lines``:
    lists of (lon, lat) arrays; ``frame`` (LocalFrame) turns them into metres."""
    if not features:
        return
    lon = np.array([f.lon for f in features])
    lat = np.array([f.lat for f in features])
    if heights_at is not None:
        dem = np.asarray(heights_at(lon, lat), dtype=np.float64)
        for f, h in zip(features, dem, strict=True):
            if f.ele is None and np.isfinite(h):
                f.ele = float(h)
        estimate_prominence([f for f in features if f.kind == "peak"], heights_at)
    if frame is not None and track_lines:
        lines = []
        for line_lon, line_lat in track_lines:
            if len(line_lon) >= 2:
                enu = frame.geodetic_to_enu(np.asarray(line_lat), np.asarray(line_lon), 0.0)
                lines.append(enu[:, :2])
        if lines:
            tracks = shapely.MultiLineString(lines)
            enu = frame.geodetic_to_enu(lat, lon, 0.0)
            d = shapely.distance(tracks, shapely.points(enu[:, 0], enu[:, 1]))
            for f, dist in zip(features, d, strict=True):
                f.track_distance_m = float(dist)


class LabelData:
    """The label features of an area: cached file, optional download in the background."""

    def __init__(self, cache: TileCache, bounds: tuple[float, float, float, float]) -> None:
        self.cache = cache
        # snapped outwards to 0.01 deg: small changes of the area reuse the cached download
        west, south, east, north = (float(v) for v in bounds)
        self.bounds = (
            math.floor(west * 100) / 100,
            math.floor(south * 100) / 100,
            math.ceil(east * 100) / 100,
            math.ceil(north * 100) / 100,
        )
        self.allow_download = True
        self.features: list[LabelFeature] | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._loader: threading.Thread | None = None
        self.version = 0  # increases when features change (layers rebuild their caches)

    @property
    def path(self) -> Path:
        return cache_path(self.cache, self.bounds)

    @property
    def ready(self) -> bool:
        return self.features is not None

    def load(self, enricher=None) -> list[LabelFeature] | None:
        """Synchronous: read the cache (downloading first if allowed); ``enricher(features)``
        runs before the features become visible (e.g. DEM heights, see LabelLayer)."""
        path = self.path
        if not path.exists():
            if not self.allow_download:
                return None
            try:
                download_labels(self.cache, self.bounds)
            except Exception as exc:  # offline / Overpass busy: labels are optional
                self.error = f"{type(exc).__name__}: {exc}"
                log.warning("cannot download labels: %s", self.error)
                return None
        try:
            features = parse_overpass(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return None
        if enricher is not None:
            try:
                enricher(features)
            except Exception:  # never lose the labels over missing DEM data
                log.exception("enriching labels failed")
        with self._lock:
            self.features = features
            self.version += 1
        return features

    def start(self, enricher=None) -> None:
        """Load in a background thread (no-op when loaded or loading)."""
        with self._lock:
            if self.features is not None or (self._loader and self._loader.is_alive()):
                return
            self._loader = threading.Thread(
                target=self.load, args=(enricher,), daemon=True, name="labels"
            )
            self._loader.start()

    def wait(self, timeout: float | None = None) -> bool:
        loader = self._loader
        if loader is not None:
            loader.join(timeout)
        return self.ready
