"""The journey of a project for scene scripts: tracks, local frame, terrain heights, times.

:func:`load_journey` builds the tracks without a GPU and makes the journey *current*: the kit's
helpers that need the frame or the time zone (``pose``, ``local``, ``poi`` ...) use it.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from earthling.core.config import Project
from earthling.render.progress import ProgressPath, journey_path
from earthling.render.track_geometry import BuiltTrack, TrackGeometryOptions, build_tracks
from earthling.session import Session

_CURRENT: Journey | None = None


@dataclass
class Journey:
    """A project's tracks as scene scripts see them: the local frame, the journey axis (walked
    tracks in time order, ``path``), every built track (``gpu``) and the terrain heights."""

    session: Session
    path: ProgressPath  # the walked tracks as one journey
    gpu: list[BuiltTrack]  # all built tracks (walked and planned)
    data: object  # TerrainData (heights)
    tz: dt.tzinfo = dt.UTC  # local time of the project (sun, clock times)
    select: Callable | None = None  # (journey, day) -> index on the path; see day_index
    places: dict = field(default_factory=dict)  # name -> (lat, lon), for place()
    level_horizon: bool = True  # pose(): keep the local vertical upright far from the origin

    @property
    def frame(self):
        return self.session.frame

    # --- coordinates ---------------------------------------------------------------------
    def enu(self, lat: float, lon: float, h: float = 0.0) -> np.ndarray:
        return np.asarray(self.frame.geodetic_to_enu(lat, lon, h), dtype=np.float64)

    def lonlat(self, p) -> tuple[float, float]:
        lat, lon, _ = self.frame.enu_to_geodetic(np.asarray(p, dtype=np.float64))
        return float(lon), float(lat)

    def ground_z(self, p) -> float:
        """ENU z of the terrain below the ENU point ``p`` (NaN without DEM data)."""
        lon, lat = self.lonlat(p)
        h = self.data.height_at(lon, lat)
        if h is None:
            return float("nan")
        return float(self.enu(lat, lon, h)[2])

    def max_ground_z(self, a, b, samples: int = 40, radius: float = 0.0) -> float:
        """Highest terrain along the segment a-b (for clearance checks)."""
        best = -1e9
        for f in np.linspace(0.0, 1.0, samples):
            p = np.asarray(a) + (np.asarray(b) - np.asarray(a)) * f
            for dx, dy in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)):
                z = self.ground_z(p + np.array([dx, dy, 0.0]))
                if math.isfinite(z):
                    best = max(best, z)
        return best

    # --- tracks (the journey: walked tracks, bus rides, ferries and flights in time order) --
    def day_index(self, day) -> int:
        """Index on the journey path of track ``day``: by default its number (1 = the first in
        time) or its file stem; projects pass their own convention as ``select``."""
        if self.select is not None:
            return self.select(self, day)
        if isinstance(day, int):
            return day - 1
        for i, g in enumerate(self.path.tracks):
            if g.track.source.stem == day:
                return i
        raise KeyError(day)

    def day(self, day):
        return self.path.tracks[self.day_index(day)]

    def day_name(self, day) -> str:
        return self.day(day).track.name

    def day_times(self, day) -> tuple[float, float]:
        t = self.day(day).time
        t = t[np.isfinite(t)]
        return float(t[0]), float(t[-1])

    def start(self, day) -> np.ndarray:
        g = self.day(day)
        return g.enu[0].copy()

    def end(self, day) -> np.ndarray:
        g = self.day(day)
        return g.enu[-1].copy()

    def place(self, key: str, h: float | None = None) -> np.ndarray:
        """ENU of a named place (``places``) on the terrain."""
        lat, lon = self.places[key]
        if h is None:
            h = self.data.height_at(lon, lat) or 0.0
        return self.enu(lat, lon, h)

    def head_at_time(self, epoch: float) -> float:
        """progress.head (mode "time") for a UTC epoch (clamped)."""
        return min(max(self.time_progress(epoch), 0.0), 1.0)

    def time_progress(self, seconds: float) -> float:
        """progress.head (mode "time") for a UTC epoch."""
        t = self.path.time[np.isfinite(self.path.time)]
        return float((seconds - t[0]) / (t[-1] - t[0]))

    def distance_progress(self, distance: float) -> float:
        return float(distance / self.path.total_m)

    def smoothed(self, distance: float, window: float) -> tuple[np.ndarray, np.ndarray]:
        """Mean position and direction of the walked path around ``distance``."""
        ds = np.linspace(distance - window, distance + window, 41)
        ds = np.clip(ds, 0.0, self.path.total_m)
        pts = np.array([self.path.position_at(d) for d in ds])
        weights = np.hanning(len(ds) + 2)[1:-1]
        centre = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        direction = pts[-1] - pts[0]
        direction[2] = 0.0
        n = float(np.linalg.norm(direction))
        direction = direction / n if n > 1e-3 else np.array([0.0, 1.0, 0.0])
        return centre, direction


def load_journey(project, origin: tuple[float, float] | None = None, *,
                 tz: dt.tzinfo | None = None, select: Callable | None = None,
                 places: dict | None = None, level_horizon: bool = True,
                 cls: type[Journey] = Journey) -> Journey:  # fmt: skip
    """The tracks of the project folder ``project`` on the terrain, with the local frame centred
    on ``origin`` (lat, lon) when given (new_scene gives the scene the same origin). ``tz``:
    the local time (default: the project's time zone). The journey becomes the current one."""
    global _CURRENT
    session = Session(Project.load(Path(project)))
    session.set_origin(origin)
    data = session.terrain_data()
    options = TrackGeometryOptions(elevation_source="dem", height_offset_m=3.0, exaggeration=1.0,
                                   smoothing=5)  # fmt: skip
    built = build_tracks(session.all_tracks, session.frame, options, data.heights_at)
    path = journey_path(built)
    _CURRENT = cls(session, path, built, data, tz or ZoneInfo(session.config.project.timezone),
                   select, dict(places or {}), level_horizon)  # fmt: skip
    return _CURRENT


def current() -> Journey | None:
    """The journey of the last :func:`load_journey` (or :func:`set_current`)."""
    return _CURRENT


def set_current(journey: Journey | None) -> None:
    """Make ``journey`` the current one (e.g. one built from the application's session)."""
    global _CURRENT
    _CURRENT = journey


def local(epoch: float) -> dt.datetime:
    """Naive local datetime (the current journey's time zone) of a UTC epoch, for
    sun.datetime."""
    tz = _CURRENT.tz if _CURRENT is not None else dt.UTC
    return dt.datetime.fromtimestamp(epoch, tz).replace(tzinfo=None)


def epoch_at(hike: Journey, day: int, clock: str) -> float:
    """UTC epoch of a local clock time ("HH:MM") on a hike day."""
    date = local(hike.day_times(day)[0]).date()
    h, m = (int(x) for x in clock.split(":"))
    return dt.datetime(date.year, date.month, date.day, h, m, tzinfo=hike.tz).timestamp()


def stops(hike: Journey, day: int, min_minutes: float = 4.0, radius_m: float = 40.0):
    """Breaks of a day: [(start epoch, end epoch, ENU position)] where we stayed within
    ``radius_m`` for at least ``min_minutes``."""
    g = hike.day(day)
    ok = np.isfinite(g.time)
    t, p = g.time[ok], g.enu[ok]
    out, i = [], 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and np.linalg.norm((p[j + 1] - p[i])[:2]) < radius_m:
            j += 1
        if t[j] - t[i] >= min_minutes * 60.0:
            out.append((float(t[i]), float(t[j]), p[i : j + 1].mean(axis=0)))
            i = j + 1
        else:
            i += 1
    return out


def stop_near(hike: Journey, day: int, epoch: float, **kw):
    """The break closest in time to ``epoch``."""
    best = None
    for start, end, pos in stops(hike, day, **kw):
        gap = 0.0 if start <= epoch <= end else min(abs(epoch - start), abs(epoch - end))
        if best is None or gap < best[0]:
            best = (gap, start, end, pos)
    return best[1:] if best else None


def distance_near(hike: Journey, lon: float, lat: float, days) -> float:
    """Path distance of the walked point closest to lon/lat within the given days."""
    target = hike.enu(lat, lon, 0.0)[:2]
    best = (math.inf, 0.0)
    for day in days:
        d0, d1 = hike.path.track_range(hike.day_index(day))  # day: track number or stem
        for d in np.arange(d0, d1, 10.0):
            gap = float(np.linalg.norm(hike.path.position_at(d)[:2] - target))
            if gap < best[0]:
                best = (gap, float(d))
    return best[1]


def nearest_on(points: np.ndarray, max_gap: float = 3000.0):
    """companion() for follow_shots: the nearest of ``points`` (ENU), or the centre itself when
    nothing is within ``max_gap`` metres."""
    pts = np.asarray(points, dtype=np.float64)

    def companion(centre):
        d = np.linalg.norm(pts[:, :2] - centre[:2], axis=1)
        i = int(np.argmin(d))
        return pts[i] if d[i] <= max_gap else np.asarray(centre)

    return companion


def track_point_near(hike: Journey, day, key_or_enu) -> tuple[np.ndarray, float]:
    """(ENU point, path distance) of walked track ``day`` closest to a place / ENU point."""
    g = hike.day(day)
    p = hike.place(key_or_enu) if isinstance(key_or_enu, str) else np.asarray(key_or_enu)
    i = int(np.argmin(np.linalg.norm((g.enu - p)[:, :2], axis=1)))
    return g.enu[i].copy(), float(g.offset_m + g.dist[i])


def track_highest(hike: Journey, day) -> tuple[np.ndarray, float]:
    """(ENU point, path distance) of the highest point of walked track ``day``."""
    g = hike.day(day)
    i = int(np.nanargmax(g.ele))
    return g.enu[i].copy(), float(g.offset_m + g.dist[i])


def first_reach(hike: Journey, day, point, radius_m: float = 60.0) -> float:
    """Path distance where walked track ``day`` first comes within ``radius_m`` of ``point``."""
    g = hike.day(day)
    gap = np.linalg.norm((g.enu - np.asarray(point))[:, :2], axis=1)
    i = int(np.argmax(gap < radius_m)) if (gap < radius_m).any() else len(gap) - 1
    return float(g.offset_m + g.dist[i])


def track_time_at(path, distance: float) -> float:
    """Recorded time at a path distance, from the track the distance lies on (path.time_at
    interpolates across the gap between two tracks: a time in the night between two days)."""
    g = path.track_at(distance)
    ok = np.isfinite(g.time)
    local_d = min(max(distance - g.offset_m, 0.0), g.length_m)
    return float(np.interp(local_d, g.dist[ok], g.time[ok]))
