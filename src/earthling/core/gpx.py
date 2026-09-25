"""Loading GPX files and computing track statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_008.8
# Elevation changes smaller than this are treated as GPS noise when summing ascent/descent.
ELEVATION_HYSTERESIS_M = 5.0


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in meters (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def ascent_descent(elevation: np.ndarray, threshold: float = ELEVATION_HYSTERESIS_M):
    """Total ascent and descent using a hysteresis filter against GPS noise."""
    ele = elevation[~np.isnan(elevation)]
    if ele.size < 2:
        return 0.0, 0.0
    ascent = descent = 0.0
    ref = ele[0]
    for value in ele[1:]:
        delta = value - ref
        if delta >= threshold:
            ascent += delta
            ref = value
        elif delta <= -threshold:
            descent -= delta
            ref = value
    return float(ascent), float(descent)


@dataclass
class Segment:
    """A continuous run of points. Arrays have equal length."""

    lat: np.ndarray
    lon: np.ndarray
    ele: np.ndarray  # NaN where missing
    time: np.ndarray  # datetime64[ms], NaT where missing

    def __len__(self) -> int:
        return int(self.lat.size)

    def step_distances(self) -> np.ndarray:
        if len(self) < 2:
            return np.zeros(0)
        return haversine_m(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])


@dataclass
class TrackStats:
    distance_m: float
    ascent_m: float
    descent_m: float
    min_ele_m: float | None
    max_ele_m: float | None
    start_time: datetime | None
    end_time: datetime | None
    point_count: int

    @property
    def duration_s(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class Track:
    name: str
    source: Path
    segments: list[Segment] = field(default_factory=list)
    visible: bool = True

    @property
    def point_count(self) -> int:
        return sum(len(s) for s in self.segments)

    def all_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat = np.concatenate([s.lat for s in self.segments]) if self.segments else np.zeros(0)
        lon = np.concatenate([s.lon for s in self.segments]) if self.segments else np.zeros(0)
        ele = np.concatenate([s.ele for s in self.segments]) if self.segments else np.zeros(0)
        return lat, lon, ele

    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)"""
        lat, lon, _ = self.all_points()
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    @cached_property
    def stats(self) -> TrackStats:
        distance = sum(float(s.step_distances().sum()) for s in self.segments)
        ascent = descent = 0.0
        for s in self.segments:
            a, d = ascent_descent(s.ele)
            ascent += a
            descent += d
        _, _, ele = self.all_points()
        valid_ele = ele[~np.isnan(ele)]
        times = np.concatenate([s.time for s in self.segments]) if self.segments else np.zeros(0)
        valid_t = times[~np.isnat(times)] if times.size else times
        start = end = None
        if valid_t.size:
            start = valid_t.min().astype("datetime64[ms]").astype(datetime)
            end = valid_t.max().astype("datetime64[ms]").astype(datetime)
        return TrackStats(
            distance_m=distance,
            ascent_m=ascent,
            descent_m=descent,
            min_ele_m=float(valid_ele.min()) if valid_ele.size else None,
            max_ele_m=float(valid_ele.max()) if valid_ele.size else None,
            start_time=start,
            end_time=end,
            point_count=self.point_count,
        )


def _segment_from_points(points: list[gpxpy.gpx.GPXTrackPoint]) -> Segment:
    lat = np.array([p.latitude for p in points], dtype=np.float64)
    lon = np.array([p.longitude for p in points], dtype=np.float64)
    ele = np.array(
        [p.elevation if p.elevation is not None else np.nan for p in points], dtype=np.float64
    )
    times = []
    for p in points:
        t = p.time
        if t is None:
            times.append(np.datetime64("NaT", "ms"))
        else:
            if t.tzinfo is not None:
                t = t.astimezone(UTC).replace(tzinfo=None)
            times.append(np.datetime64(t, "ms"))
    return Segment(lat, lon, ele, np.array(times, dtype="datetime64[ms]"))


def load_gpx_file(path: Path) -> list[Track]:
    """Load all tracks (and routes) of one GPX file. Times are kept in UTC (naive)."""
    with path.open(encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    tracks: list[Track] = []
    for index, trk in enumerate(gpx.tracks):
        segments = [_segment_from_points(s.points) for s in trk.segments if len(s.points) >= 2]
        if segments:
            name = trk.name or (path.stem if len(gpx.tracks) == 1 else f"{path.stem} #{index + 1}")
            tracks.append(Track(name, path, segments))
    for index, rte in enumerate(gpx.routes):
        if len(rte.points) >= 2:
            name = rte.name or f"{path.stem} route #{index + 1}"
            tracks.append(Track(name, path, [_segment_from_points(rte.points)]))
    return tracks


def load_gpx_folder(folder: Path) -> tuple[list[Track], list[str]]:
    """Load every ``*.gpx`` in ``folder`` (sorted by name). Returns tracks and error messages."""
    tracks: list[Track] = []
    errors: list[str] = []
    if not folder.is_dir():
        return tracks, [f"GPX folder does not exist: {folder}"]
    for path in sorted(folder.glob("*.gpx"), key=lambda p: p.name.lower()):
        try:
            tracks.extend(load_gpx_file(path))
        except Exception as exc:  # gpxpy raises various exception types
            log.warning("failed to load %s: %s", path, exc)
            errors.append(f"{path.name}: {exc}")
    tracks.sort(key=lambda t: (t.stats.start_time or datetime.max, t.name))
    return tracks, errors
