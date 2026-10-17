"""Values and texts of the stats overlay: walked distance, ascent, elevation, speed, heart
rate and clock along the journey (:class:`HikeStats`), and their formatting in English or
German. No drawing (see :mod:`earthling.render.hud`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from earthling.core.gpx import cumulative_ascent_descent
from earthling.render.track_geometry import motion_values

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]  # fmt: skip
# on-screen words per language (stats.language)
WORDS = {
    "en": {"day": "Day {n}", "days": "{n} days", "distance": "distance", "ascent": "ascent",
           "elevation": "elevation", "time": "{t:%H:%M}", "months": MONTHS, "speed": "speed",
           "pace": "pace", "gradient": "gradient", "heart_rate": "heart rate", "gap": "gap"},
    "de": {"day": "Tag {n}", "days": "{n} Tage", "distance": "Strecke", "ascent": "Aufstieg",
           "elevation": "Höhe", "time": "{t:%H:%M} Uhr", "speed": "Tempo", "pace": "Pace",
           "gradient": "Steigung", "heart_rate": "Puls", "gap": "Abstand",
           "months": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                      "September", "Oktober", "November", "Dezember"]},
}  # fmt: skip


def word(lang: str, key: str):
    return WORDS.get(lang, WORDS["en"])[key]


# --- values ----------------------------------------------------------------------------------
@dataclass
class StatsAt:
    day: int
    time: datetime | None  # local time at the head (None without GPX times)
    distance_m: float
    day_distance_m: float
    elevation_m: float | None
    ascent_m: float
    day_ascent_m: float
    descent_m: float
    day_descent_m: float
    speed_kmh: float | None = None  # at the head (None without times)
    gradient: float | None = None  # % (None without elevations)
    heart_rate: float | None = None  # bpm (None without heart rate data)
    track: int = 0  # index of the track at the head


def _fill_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    if ok.all() or not ok.any():
        return values
    idx = np.arange(len(values))
    return np.interp(idx, idx[ok], values[ok])


class HikeStats:
    """Per-point running totals of the whole hike (``ProgressPath`` order)."""

    def __init__(self, path, timezone: str = "UTC") -> None:
        self.path = path
        self.tz = ZoneInfo(timezone)
        self.dist = path.dist  # progress axis (drawn track)
        self.ground = path.ground  # metres walked
        n = len(self.dist)
        self.has_elevation = bool(np.isfinite(path.ele).any())
        self.ele = _fill_nan(path.ele)
        self.ascent = np.zeros(n)
        self.descent = np.zeros(n)
        up = down = 0.0
        for index in np.unique(path.track_index):
            sel = np.flatnonzero(path.track_index == index)
            a, d = cumulative_ascent_descent(self.ele[sel])  # no jumps between tracks
            self.ascent[sel] = a + up
            self.descent[sel] = d + down
            up, down = self.ascent[sel[-1]], self.descent[sel[-1]]
        # elevation, gradient, speed and heart rate per point
        values = getattr(path, "values", None)
        if values is not None and len(values) == n:
            self.values = values  # the same values the tracks are coloured by
        else:
            self.values = motion_values(self.ground, self.ele, path.time, getattr(path, "hr", None))
        times = path.time
        self.has_time = bool(np.isfinite(times).sum() >= 2)
        if self.has_time:
            self.time = _fill_nan(times)
            dates = self._local_dates(self.time)
            first = dates[0]
            self.day = np.array([(d - first).days + 1 for d in dates])
        else:
            self.time = times
            self.day = path.track_index.astype(int) + 1
        # index of the first point of each point's day
        _, first_index = np.unique(self.day, return_index=True)
        start_of = dict(zip(self.day[first_index], first_index, strict=True))
        self.day_start = np.array([start_of[d] for d in self.day]) if n else np.zeros(0, int)
        # ... and of each point's track
        self.track = np.asarray(path.track_index, dtype=int)
        _, first_index = np.unique(self.track, return_index=True)
        start_of = dict(zip(self.track[first_index], first_index, strict=True))
        self.track_start = (np.array([start_of[t] for t in self.track]) if n
                            else np.zeros(0, int))  # fmt: skip

    def _local_dates(self, seconds: np.ndarray) -> list[date]:
        # the UTC offset only changes a few times a year: compute it per hour
        cache: dict[int, int] = {}
        out = []
        for s in seconds:
            hour = int(s // 3600)
            offset = cache.get(hour)
            if offset is None:
                utc = datetime.fromtimestamp(hour * 3600, tz=UTC)
                offset = int(utc.astimezone(self.tz).utcoffset().total_seconds())
                cache[hour] = offset
            out.append(datetime.fromtimestamp(s + offset, tz=UTC).date())
        return out

    def at(self, distance: float, by_track: bool = False) -> StatsAt | None:
        """The values at ``distance``; the day_* totals count from the start of the day, or
        with ``by_track`` from the start of the track."""
        n = len(self.dist)
        if n == 0:
            return None
        d = min(max(distance, float(self.dist[0])), float(self.dist[-1]))
        i = int(np.clip(np.searchsorted(self.dist, d, side="right") - 1, 0, n - 1))
        j = min(i + 1, n - 1)
        span = self.dist[j] - self.dist[i]
        f = (d - self.dist[i]) / span if span > 0 else 0.0

        def lerp(a: np.ndarray) -> float:
            return float(a[i] + (a[j] - a[i]) * f)

        def value(k: int) -> float | None:
            a, b = self.values[i, k], self.values[j, k]
            if not (np.isfinite(a) and np.isfinite(b)):
                return None
            return float(a + (b - a) * f)

        s = int(self.track_start[i] if by_track else self.day_start[i])
        time = None
        if self.has_time:
            time = datetime.fromtimestamp(lerp(self.time), tz=UTC).astimezone(self.tz)
        return StatsAt(
            day=int(self.day[i]),
            time=time,
            distance_m=lerp(self.ground),
            day_distance_m=lerp(self.ground) - float(self.ground[s]),
            elevation_m=lerp(self.ele) if self.has_elevation else None,
            ascent_m=lerp(self.ascent),
            day_ascent_m=lerp(self.ascent) - float(self.ascent[s]),
            descent_m=lerp(self.descent),
            day_descent_m=lerp(self.descent) - float(self.descent[s]),
            speed_kmh=value(2),
            gradient=value(1) if self.has_elevation else None,
            heart_rate=value(3),
            track=int(self.track[i]),
        )

    def part(self, v: StatsAt, scope: str):
        """Selection of the points of the whole hike, of v's day or of v's track."""
        if scope == "day":
            return self.day == v.day
        if scope == "track":
            return self.track == v.track
        return slice(None)

    def profile(self, day: int | None = None, sel=None):
        """(walked metres, elevation, progress distance) arrays of the hike, of one day or of
        the points ``sel``."""
        if sel is None:
            sel = slice(None) if day is None else self.day == day
        return self.ground[sel], self.ele[sel], self.dist[sel]


def format_date(t: datetime, style: str, lang: str = "en") -> str:
    if style == "iso":
        return t.strftime("%Y-%m-%d")
    months = word(lang, "months")
    dot = "." if lang == "de" else ""  # "14. Juli 2026"
    if style == "short":
        return f"{t.day}{dot} {months[t.month - 1][:3]}{dot}"
    return f"{t.day}{dot} {months[t.month - 1]} {t.year}"


def format_number(value: float, decimals: int = 0, lang: str = "en") -> str:
    s = f"{value:,.{decimals}f}"
    if lang == "de":  # 12.345,7
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    return s.replace(",", " ")  # thin space as thousands separator


def _widest(text: str, lang: str = "en") -> str:
    """``text`` as wide as it can get while its numbers and the month change: digits as "8"
    (the font's digits are tabular), the month name as the longest one."""
    out = re.sub(r"\d", "8", text)
    months = word(lang, "months")
    for m in months:
        if m in out:
            return out.replace(m, max(months, key=len))
    return out


def format_gap(seconds: float) -> str:
    """ "+1:23" (the other one is behind), "-0:45", "+1:02:10"."""
    sign = "+" if seconds >= 0 else "−"
    s = int(round(abs(seconds)))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{sign}{h}:{m:02d}:{sec:02d}" if h else f"{sign}{m}:{sec:02d}"


def format_speed(kmh: float, unit: str, lang: str = "en") -> tuple[str, str]:
    """(value text, caption) of a speed: km/h, pace (min/km) or knots."""
    if unit == "pace":
        if kmh < 1.0:
            return "–:–– /km", word(lang, "pace")
        m, sec = divmod(int(round(3600.0 / kmh)), 60)
        return f"{m}:{sec:02d} /km", word(lang, "pace")
    if unit == "knots":
        return f"{format_number(kmh / 1.852, 1, lang)} kn", word(lang, "speed")
    return f"{format_number(kmh, 1, lang)} km/h", word(lang, "speed")
