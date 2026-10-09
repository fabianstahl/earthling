"""Helpers that generate the synthetic GPX tracks of the showcase scenes.

Plausible tracks along real paths: places from Nominatim, the routes from the OpenStreetMap
routers (routing.openstreetmap.de: foot, bike, car), ship and canoe routes over the water mask,
heights from the AWS Terrain Tiles (Terrarium), timestamps from simple speed models (Tobler's
hiking function, a cyclist's power balance, a runner's pace with fatigue) and a heart rate that
follows the effort. Every network answer is cached in ``tools/cache`` (not committed).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import heapq
import io
import json
import math
import time
from pathlib import Path

import httpx
import numpy as np

TOOLS = Path(__file__).resolve().parent
SHOWCASE = TOOLS.parent
CACHE = TOOLS / "cache"
UA = {"User-Agent": "earthling-showcase/0.1 (synthetic demo tracks; f.stahl@sinabis.com)"}
R_EARTH = 6371008.8
UTC = dt.UTC


# --- places ---------------------------------------------------------------------------------
def geocode(query: str) -> tuple[float, float]:
    """(lat, lon) of the first Nominatim result (cached)."""
    path = CACHE / "geocode.json"
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if query not in known:
        time.sleep(1.1)  # Nominatim usage policy: at most one request per second
        r = httpx.get("https://nominatim.openstreetmap.org/search",
                      params={"q": query, "format": "json", "limit": 1}, headers=UA, timeout=60)  # fmt: skip
        r.raise_for_status()
        hits = r.json()
        if not hits:
            raise KeyError(f"no geocoding result for {query!r}")
        known[query] = [float(hits[0]["lat"]), float(hits[0]["lon"])]
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
    lat, lon = known[query]
    return lat, lon


def place(p) -> tuple[float, float]:
    """A (lat, lon) tuple as is, or a Nominatim query string."""
    return geocode(p) if isinstance(p, str) else (float(p[0]), float(p[1]))


# --- geometry ---------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cumulative(pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    steps = haversine(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1])
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample(pts, spacing: float) -> np.ndarray:
    """Points every ``spacing`` metres along the polyline (lat, lon)."""
    p = np.asarray(pts, dtype=np.float64)
    s = cumulative(p)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    q = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])])


def great_circle(a, b, step_m: float = 10000.0) -> np.ndarray:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    v1 = np.array([math.cos(la1) * math.cos(lo1), math.cos(la1) * math.sin(lo1), math.sin(la1)])
    v2 = np.array([math.cos(la2) * math.cos(lo2), math.cos(la2) * math.sin(lo2), math.sin(la2)])
    omega = math.acos(max(-1.0, min(1.0, float(v1 @ v2))))
    n = max(2, int(omega * R_EARTH / step_m) + 1)
    out = []
    for i in range(n + 1):
        f = i / n
        w1, w2 = ((1 - f, f) if omega < 1e-9
                  else (math.sin((1 - f) * omega) / math.sin(omega), math.sin(f * omega) / math.sin(omega)))  # fmt: skip
        v = w1 * v1 + w2 * v2
        out.append((math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))),
                    math.degrees(math.atan2(v[1], v[0]))))  # fmt: skip
    return np.array(out)


def smooth_line(pts, window: int) -> np.ndarray:
    """Moving average (ends kept) - takes the corners out of grid routes."""
    p = np.asarray(pts, dtype=np.float64)
    if window <= 1 or len(p) < 3:
        return p
    half = window // 2
    pad = np.pad(p, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    out = np.column_stack([np.convolve(pad[:, k], kernel, mode="valid") for k in range(2)])
    out[0], out[-1] = p[0], p[-1]
    return out


# --- routing ----------------------------------------------------------------------------------
PROFILES = {"foot": "routed-foot", "bike": "routed-bike", "car": "routed-car"}


def route(profile: str, stops, snap_ends: bool = True) -> np.ndarray:
    """(lat, lon) points of the route through ``stops`` (OSRM on OpenStreetMap data)."""
    stops = [place(s) for s in stops]
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in stops)
    key = hashlib.md5(f"{profile}|{coords}".encode()).hexdigest()[:16]
    path = CACHE / "routes" / f"{key}.json"
    if not path.exists():
        url = f"https://routing.openstreetmap.de/{PROFILES[profile]}/route/v1/driving/{coords}"
        for attempt in range(5):
            r = httpx.get(url, params={"overview": "full", "geometries": "geojson"}, headers=UA,
                          timeout=180)  # fmt: skip
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
        time.sleep(1.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    pts = [(lat, lon) for lon, lat in doc["routes"][0]["geometry"]["coordinates"]]
    if snap_ends:  # start and end exactly at the stops (the router snaps to the network)
        pts = [stops[0], *pts, stops[-1]]
    return np.array(pts)


def overpass(query: str) -> dict:
    key = hashlib.md5(query.encode()).hexdigest()[:16]
    path = CACHE / "overpass" / f"{key}.json"
    if not path.exists():
        for attempt in range(5):
            r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": query},
                           headers=UA, timeout=300)  # fmt: skip
            if r.status_code == 200:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def trail(name_regex: str, bbox, start, end) -> np.ndarray:
    """A named trail from OpenStreetMap (all ways whose name matches), merged into one line and
    cut from the point nearest ``start`` to the point nearest ``end``. ``bbox`` = (south, west,
    north, east). For trails the routers do not connect."""
    import shapely
    import shapely.ops
    from shapely.geometry import LineString, Point

    s, w, n, e = bbox
    doc = overpass(f'[out:json][timeout:120];way["name"~"{name_regex}"]({s},{w},{n},{e});out geom;')
    lines = [LineString([(p["lon"], p["lat"]) for p in el["geometry"]])
             for el in doc["elements"] if el.get("geometry")]  # fmt: skip
    merged = shapely.line_merge(shapely.MultiLineString(lines))
    parts = list(getattr(merged, "geoms", [merged]))
    a, b = place(start), place(end)
    pa, pb = Point(a[1], a[0]), Point(b[1], b[0])
    best = min(parts, key=lambda g: g.distance(pa) + g.distance(pb))
    d0, d1 = best.project(pa), best.project(pb)
    seg = shapely.ops.substring(best, min(d0, d1), max(d0, d1))
    pts = np.array([(lat, lon) for lon, lat in seg.coords])
    if d0 > d1:
        pts = pts[::-1]
    return np.vstack([a, pts, b])


# --- heights (AWS Terrain Tiles, Terrarium encoding) ------------------------------------------
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_tiles: dict[tuple[int, int, int], np.ndarray] = {}


def _terrarium(z: int, x: int, y: int) -> np.ndarray:
    key = (z, x, y)
    if key not in _tiles:
        path = Path.home() / "earthling-cache" / "_terrarium_raw" / str(z) / str(x) / f"{y}.png"
        if not path.exists():
            r = httpx.get(TERRARIUM.format(z=z, x=x, y=y), headers=UA, timeout=60)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
        from PIL import Image

        rgb = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"), np.float64)
        _tiles[key] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    return _tiles[key]


def heights(pts, z: int = 12) -> np.ndarray:
    """Terrain heights (m) at the (lat, lon) points, bilinear on zoom-``z`` tiles."""
    p = np.asarray(pts, dtype=np.float64)
    lat, lon = p[:, 0], p[:, 1]
    n = 2**z
    tx = (lon + 180.0) / 360.0 * n
    ty = (1.0 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    out = np.zeros(len(p))
    px, py = tx * 256 - 0.5, ty * 256 - 0.5
    for i in range(len(p)):
        x0, y0 = int(math.floor(px[i])), int(math.floor(py[i]))
        fx, fy = px[i] - x0, py[i] - y0
        vals = []
        for dy in (0, 1):
            for dx in (0, 1):
                gx, gy = x0 + dx, y0 + dy
                tile = _terrarium(z, gx // 256 % n, gy // 256)
                vals.append(tile[gy % 256, gx % 256])
        out[i] = ((vals[0] * (1 - fx) + vals[1] * fx) * (1 - fy)
                  + (vals[2] * (1 - fx) + vals[3] * fx) * fy)  # fmt: skip
    return out


def smooth(values, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(v) < 3:
        return v
    half = window // 2
    pad = np.pad(v, (half, half), mode="edge")
    return np.convolve(pad, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")


# --- speed models -------------------------------------------------------------------------------
def tobler_times(pts, ele, start: dt.datetime, factor: float = 1.0, breaks=(),
                 seed: int = 0) -> np.ndarray:  # fmt: skip
    """Walking times (UTC epoch seconds) with Tobler's hiking function. ``breaks``: (fraction
    of the distance, minutes) pauses. ``factor`` scales the speed (fit hikers > 1)."""
    s = cumulative(pts)
    ds = np.diff(s)
    slope = np.diff(smooth(ele, 9)) / np.maximum(ds, 0.5)
    rng = np.random.default_rng(seed)
    wobble = 1.0 + 0.08 * smooth(rng.normal(size=len(ds)), 25)
    v = 6.0 * np.exp(-3.5 * np.abs(slope + 0.05)) / 3.6 * factor * wobble  # m/s
    t = np.concatenate([[0.0], np.cumsum(ds / np.maximum(v, 0.15))])
    for frac, minutes in breaks:
        t[s >= frac * s[-1]] += minutes * 60.0
    return start.timestamp() + t


def bike_times(pts, ele, start: dt.datetime, power_w: float = 230.0, mass: float = 82.0,
               v_max_kmh: float = 55.0, seed: int = 0) -> np.ndarray:  # fmt: skip
    """Cycling times from the power balance (rolling, gravity, air)."""
    s = cumulative(pts)
    ds = np.diff(s)
    slope = np.diff(smooth(ele, 15)) / np.maximum(ds, 0.5)
    rng = np.random.default_rng(seed)
    power = power_w * (1.0 + 0.07 * smooth(rng.normal(size=len(ds)), 31))
    g, crr, cda, rho = 9.81, 0.004, 0.32, 1.1
    v = np.empty(len(ds))
    for i, (gr, p) in enumerate(zip(slope, power, strict=True)):
        lo, hi = 0.5, v_max_kmh / 3.6
        for _ in range(40):  # bisection: P(v) = p
            mid = (lo + hi) / 2
            need = (mass * g * (math.sin(math.atan(gr)) + crr) + 0.5 * rho * cda * mid * mid) * mid
            lo, hi = (mid, hi) if need < p else (lo, mid)
        v[i] = (lo + hi) / 2
    v = smooth(v, 7)
    t = np.concatenate([[0.0], np.cumsum(ds / np.maximum(v, 0.5))])
    return start.timestamp() + t


def run_times(pts, ele, start: dt.datetime, finish_s: float, fade: float = 0.06,
              seed: int = 0) -> np.ndarray:  # fmt: skip
    """Running times: even pace with a little hill effect, slowing by ``fade`` towards the
    end, scaled to finish in ``finish_s`` seconds."""
    s = cumulative(pts)
    ds = np.diff(s)
    slope = np.diff(smooth(ele, 9)) / np.maximum(ds, 0.5)
    rng = np.random.default_rng(seed)
    x = s[1:] / s[-1]
    slope = np.clip(smooth(slope, 15), -0.04, 0.06)  # (bridges and DEM noise in a flat city)
    pace = (1.0 + fade * x**2) * (1.0 + 3.0 * slope) * (1.0 + 0.03 * smooth(rng.normal(size=len(ds)), 41))
    t = np.concatenate([[0.0], np.cumsum(ds * pace)])
    return start.timestamp() + t * finish_s / t[-1]


def heart_rate(times, pts, ele, rest: float = 62.0, base: float = 128.0, gain: float = 2600.0,
               tau_s: float = 45.0, hr_max: float = 186.0, seed: int = 0,
               speed_gain: float = 6.0) -> np.ndarray:  # fmt: skip
    """A plausible heart rate: effort (speed plus climbing rate), lagging by ``tau_s`` and
    drifting up slowly."""
    t = np.asarray(times, dtype=np.float64)
    s = cumulative(pts)
    dt_ = np.maximum(np.diff(t), 0.5)
    speed = np.diff(s) / dt_
    climb = np.clip(np.diff(np.asarray(ele, dtype=np.float64)) / dt_, -0.5, 2.0)
    effort = np.concatenate([[0.0], speed * speed_gain + climb * gain / 60.0])
    moving = np.concatenate([[0.0], (speed > 0.3).astype(float)])
    rng = np.random.default_rng(seed)
    target = np.where(moving > 0, base + effort + 8.0 * (t - t[0]) / 3600.0, rest + 20.0)
    hr = np.empty(len(t))
    hr[0] = rest + 25.0
    for i in range(1, len(t)):
        a = 1.0 - math.exp(-(t[i] - t[i - 1]) / tau_s)
        hr[i] = hr[i - 1] + (target[i] - hr[i - 1]) * a
    hr += 1.5 * smooth(rng.normal(size=len(t)), 9)
    return np.clip(np.round(hr), rest, hr_max)


def schedule_times(pts, stops_at, stop_times) -> np.ndarray:
    """Times along a line through scheduled stops: ``stops_at`` are point indices, ``stop_times``
    matching (arrival, departure) UTC datetimes; in between the speed is constant."""
    s = cumulative(pts)
    t = np.zeros(len(s))
    for k in range(len(stops_at) - 1):
        i0, i1 = stops_at[k], stops_at[k + 1]
        dep = stop_times[k][1].timestamp()
        arr = stop_times[k + 1][0].timestamp()
        seg = s[i0 : i1 + 1]
        f = (seg - seg[0]) / max(seg[-1] - seg[0], 1e-6)
        t[i0 : i1 + 1] = dep + f * (arr - dep)
    return t


# --- routing over water (ships, canoes) --------------------------------------------------------
def water_route(a, b, land_cost: float | None = None, zoom: int = 12, margin_deg: float = 0.25,
                shore_m: float = 300.0) -> np.ndarray:  # fmt: skip
    """Shortest path from a to b over water (A* on the OpenStreetMap water mask). Ships keep
    off the shore (``shore_m``); with ``land_cost`` land can be crossed at that cost per metre
    (portages: a canoe is carried over the narrowest land). Returns (lat, lon) points."""
    from earthling.core.geo import lonlat_to_tile, tile_to_lonlat
    from earthling.data.cache import TileCache
    from earthling.data.water import WaterData

    a, b = place(a), place(b)
    key = hashlib.md5(f"{a}|{b}|{land_cost}|{zoom}|{margin_deg}|{shore_m}|v2".encode()).hexdigest()[:16]
    path = CACHE / "water_routes" / f"{key}.json"
    if path.exists():
        return np.array(json.loads(path.read_text()))
    # ships stay on the sea (a port geocoded inland must not snap into a lake)
    water = WaterData(TileCache(Path.home() / "earthling-cache"),
                      {"ocean"} if land_cost is None else None)  # fmt: skip
    west, east = min(a[1], b[1]) - margin_deg, max(a[1], b[1]) + margin_deg
    south, north = min(a[0], b[0]) - margin_deg, max(a[0], b[0]) + margin_deg
    size = 128
    grid, (x0, y0) = water.raster((west, south, east, north), zoom, size)
    h, w = grid.shape
    cell_m = 2 * math.pi * R_EARTH * math.cos(math.radians((a[0] + b[0]) / 2)) / (2**zoom * size)
    wet = grid > 0.5
    # distance to the shore (in cells, up to `reach`) by repeated erosion
    reach = max(1, int(shore_m / cell_m))
    shore = np.full(grid.shape, reach, dtype=np.float32)
    current = wet.copy()
    for k in range(reach):
        eroded = current.copy()
        eroded[1:, :] &= current[:-1, :]
        eroded[:-1, :] &= current[1:, :]
        eroded[:, 1:] &= current[:, :-1]
        eroded[:, :-1] &= current[:, 1:]
        shore[current & ~eroded] = k
        current = eroded
    cost = np.where(wet, 1.0 + 2.0 * (1.0 - shore / reach) ** 2, np.inf)
    if land_cost is not None:
        cost = np.where(wet, cost, land_cost)

    def cell(p):
        tx, ty = lonlat_to_tile(p[1], p[0], zoom)
        return int(np.clip((ty - y0) * size, 0, h - 1)), int(np.clip((tx - x0) * size, 0, w - 1))

    start, goal = cell(a), cell(b)
    if land_cost is None:  # a port given as a town: the nearest water (the quay)
        start, goal = nearest_wet(wet, start), nearest_wet(wet, goal)
    cost[start] = cost[goal] = 1.0  # the piers / put-ins themselves
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]  # fmt: skip
    best = np.full(grid.shape, np.inf)
    came = np.full(grid.shape + (2,), -1, dtype=np.int32)
    best[start] = 0.0
    heap = [(0.0, start)]
    gy, gx = goal
    while heap:
        f, (y, x) = heapq.heappop(heap)
        if (y, x) == goal:
            break
        g0 = best[y, x]
        for dy, dx, step in steps:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            c = cost[ny, nx]
            if not np.isfinite(c):
                continue
            g1 = g0 + step * c
            if g1 < best[ny, nx]:
                best[ny, nx] = g1
                came[ny, nx] = (y, x)
                heapq.heappush(heap, (g1 + math.hypot(ny - gy, nx - gx), (ny, nx)))
    if not np.isfinite(best[goal]):
        raise RuntimeError(f"no water route from {a} to {b}")
    cells = [goal]
    while cells[-1] != start:
        y, x = came[cells[-1]]
        cells.append((int(y), int(x)))
    cells.reverse()
    tx = np.array([x0 + (c[1] + 0.5) / size for c in cells])
    ty = np.array([y0 + (c[0] + 0.5) / size for c in cells])
    lon, lat = tile_to_lonlat(tx, ty, zoom)
    pts = np.column_stack([lat, lon])
    pts = smooth_line(pts, 9)
    if land_cost is not None:
        pts = np.vstack([a, pts[1:-1], b])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pts.tolist()))
    return pts


# --- writing ------------------------------------------------------------------------------------
def write_gpx(path: Path, name: str, pts, ele, times, hr=None) -> Path:
    """A GPX 1.1 track (heart rate as Garmin TrackPointExtension)."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="earthling showcase generator" '
             'xmlns="http://www.topografix.com/GPX/1/1" '
             'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">',
             f"  <trk><name>{name}</name><trkseg>"]  # fmt: skip
    for i, (lat, lon) in enumerate(np.asarray(pts)):
        stamp = dt.datetime.fromtimestamp(float(times[i]), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        ext = ""
        if hr is not None:
            ext = (f"<extensions><gpxtpx:TrackPointExtension><gpxtpx:hr>{int(hr[i])}</gpxtpx:hr>"
                   "</gpxtpx:TrackPointExtension></extensions>")  # fmt: skip
        lines.append(f'    <trkpt lat="{lat:.6f}" lon="{lon:.6f}"><ele>{float(ele[i]):.1f}</ele>'
                     f"<time>{stamp}</time>{ext}</trkpt>")  # fmt: skip
    lines.append("  </trkseg></trk>\n</gpx>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    km = cumulative(pts)[-1] / 1000.0
    hours = (float(times[-1]) - float(times[0])) / 3600.0
    print(f"wrote {path.relative_to(SHOWCASE)}: {len(pts)} points, {km:.1f} km, {hours:.2f} h")
    return path


def thin_by_time(pts, ele, times, every_s: float, hr=None):
    """Keep about one point every ``every_s`` seconds (and the last one)."""
    t = np.asarray(times, dtype=np.float64)
    keep = [0]
    for i in range(1, len(t) - 1):
        if t[i] - t[keep[-1]] >= every_s:
            keep.append(i)
    keep.append(len(t) - 1)
    keep = np.array(keep)
    out = [np.asarray(pts)[keep], np.asarray(ele)[keep], t[keep]]
    return (*out, np.asarray(hr)[keep]) if hr is not None else tuple(out)


def local(y, mo, d, h, mi, utc_offset_h: float) -> dt.datetime:
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone(dt.timedelta(hours=utc_offset_h)))


def zigzag(a, b, width_m: float, period_m: float, spacing: float = 10.0) -> np.ndarray:
    """Switchbacks from a to b: a smooth path swinging ``width_m`` to each side."""
    line = resample([place(a), place(b)], spacing)
    s = cumulative(line)
    lat0 = math.radians(line[0, 0])
    d = np.array([line[-1, 0] - line[0, 0], (line[-1, 1] - line[0, 1]) * math.cos(lat0)])
    d /= np.linalg.norm(d)
    normal = np.array([-d[1], d[0]])  # (dlat, dlon * cos)
    swing = np.sin(2 * math.pi * s / period_m) * np.sin(math.pi * s / s[-1]) ** 0.5 * width_m
    out = line.copy()
    out[:, 0] += np.degrees(swing * normal[0] / R_EARTH)
    out[:, 1] += np.degrees(swing * normal[1] / R_EARTH / math.cos(lat0))
    return out


def nearest_wet(wet: np.ndarray, cell: tuple[int, int], reach: int = 200) -> tuple[int, int]:
    """The water cell nearest to ``cell`` (itself if it is water)."""
    y, x = cell
    if wet[y, x]:
        return cell
    h, w = wet.shape
    y0, y1, x0, x1 = max(0, y - reach), min(h, y + reach + 1), max(0, x - reach), min(w, x + reach + 1)
    ys, xs = np.nonzero(wet[y0:y1, x0:x1])
    if not len(ys):
        raise RuntimeError(f"no water near {cell}")
    k = int(np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2))
    return int(ys[k] + y0), int(xs[k] + x0)
