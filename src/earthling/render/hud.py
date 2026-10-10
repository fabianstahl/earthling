"""Stats overlay (HUD): day, date, distance, ascent and elevation at the hike progress, an
elevation profile with the current position, and an optional attribution line.

Values are interpolated along the progress path at the head of the drawn track, so they always
match the hiker marker. Drawn after the labels, in output pixels (sizes are given for 1080p
and scale with the output height), with the labels' SDF font.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import moderngl
import numpy as np

from earthling.core.gpx import cumulative_ascent_descent
from earthling.render.labels import BASE_PX, SPREAD, FontAtlas
from earthling.render.scale import pixel_scale
from earthling.render.shader_library import ShaderLibrary

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
        from earthling.render.tracks import motion_values

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

    def at(self, distance: float) -> StatsAt | None:
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

        s = int(self.day_start[i])
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
        )

    def profile(self, day: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(walked metres, elevation, progress distance) of the hike or of one day."""
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


def _segment(batch, pts, thickness, color, alpha) -> None:
    _polyline(batch, pts, thickness, color, alpha)


def _polyline(batch, pts, thickness, color, alpha) -> None:
    """A line of quads through the points (pixels)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return
    d = np.diff(pts, axis=0)
    length = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    n = np.column_stack([-d[:, 1], d[:, 0]]) / length[:, None] * thickness / 2.0
    a0, a1 = pts[:-1] + n, pts[1:] + n
    b0, b1 = pts[:-1] - n, pts[1:] - n
    batch.triangles(np.stack([a0, a1, b1, a0, b1, b0], axis=1).reshape(-1, 2), color, alpha)


def _disc(batch, centre, radius, color, alpha, n: int = 16) -> None:
    a = np.linspace(0.0, 2.0 * math.pi, n + 1)
    ring = np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])
    tris = np.stack([np.repeat([centre], n, axis=0), ring[:-1], ring[1:]], axis=1)
    batch.triangles(tris.reshape(-1, 2), color, alpha)


def _widest(text: str, lang: str = "en") -> str:
    """``text`` as wide as it can get while its numbers and the month change: digits as "8"
    (the font's digits are tabular), the month name as the longest one."""
    import re

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


# --- drawing ---------------------------------------------------------------------------------
class _Batch:
    """Vertices: x, y (px, y down), u, v, r, g, b, a (display colour), scale, mode."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []

    def quad(self, x0, y0, x1, y1, color, alpha) -> None:
        self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, alpha)

    def polygon(self, points, color, alpha) -> None:
        """Convex polygon as a triangle fan."""
        p = np.asarray(points, dtype=np.float64)
        tris = np.stack([np.repeat(p[:1], len(p) - 2, axis=0), p[1:-1], p[2:]], axis=1).reshape(
            -1, 2
        )
        self.triangles(tris, color, alpha)

    def triangles(self, xy: np.ndarray, color, alpha) -> None:
        out = np.zeros((len(xy), 10))
        out[:, :2] = xy
        out[:, 4:7] = color
        out[:, 7] = alpha
        out[:, 8], out[:, 9] = 1.0, 1.0
        self.parts.append(out)

    def glyphs(self, rects: np.ndarray, uvs: np.ndarray, color, alpha, scale) -> None:
        n = len(rects)
        if not n:
            return
        x0, y0, x1, y1 = rects.T
        u0, v0, u1, v1 = uvs.T
        corners = ((x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1),
                   (x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1))  # fmt: skip
        out = np.zeros((n, 6, 10))
        for k, (xs, ys, us, vs) in enumerate(corners):
            out[:, k, 0], out[:, k, 1], out[:, k, 2], out[:, k, 3] = xs, ys, us, vs
        out[:, :, 4:7] = color
        out[:, :, 7] = alpha
        out[:, :, 8], out[:, :, 9] = scale, 0.0
        self.parts.append(out.reshape(-1, 10))

    def array(self) -> np.ndarray:
        return (
            np.concatenate(self.parts).astype(np.float32)
            if self.parts
            else np.zeros((0, 10), np.float32)
        )


class HudLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, atlas_provider) -> None:
        """``atlas_provider()`` returns the shared FontAtlas (created on first use)."""
        self.ctx = ctx
        self.shaders = shaders
        self.atlas_provider = atlas_provider
        self.attribution = ""  # e.g. "Imagery: … · Elevation: …"
        self._stats: HikeStats | None = None
        self._stats_key: tuple | None = None
        self._buffer: moderngl.Buffer | None = None
        self._vaos: dict[int, moderngl.VertexArray] = {}
        self.last_stats: StatsAt | None = None  # values shown in the last frame (tests)
        self.frame = None  # LocalFrame of the scene (time zone at the head position)
        self.gap_s: float | None = None  # clock mode: the second track's gap (seconds)
        # the mini-map: set by the renderer every frame
        self.map_path = None  # the whole journey (ProgressPath)
        self.map_heads: list[float] = []  # global distances of the heads
        self.map_colors: dict[int, tuple] = {}  # track index -> line colour (sRGB)
        self.map_sources = (None, None)  # (WaterData, BorderData)
        self._map_geometry = None
        self._map_key = None

    def stats_for(self, path, timezone: str) -> HikeStats | None:
        if path is None or len(path.dist) == 0:
            return None
        key = (id(path), timezone)
        if key != self._stats_key:
            self._stats = HikeStats(path, timezone)
            self._stats_key = key
        return self._stats

    # --- text helpers ---------------------------------------------------------------------
    def _text(self, batch, text, x, baseline, size, color, alpha, align="left") -> float:
        atlas: FontAtlas = self.atlas_provider()
        atlas.ensure(text)
        scale = size / BASE_PX
        width = atlas.width(text) * scale
        if align == "right":
            x -= width
        elif align == "center":
            x -= width / 2.0
        rects, uvs = atlas.quads(text)
        if len(rects):
            q = rects * scale
            q[:, [0, 2]] += x
            q[:, [1, 3]] += baseline
            batch.glyphs(q, uvs, color, alpha, scale)
        return width

    def _width(self, text: str, size: float) -> float:
        atlas: FontAtlas = self.atlas_provider()
        atlas.ensure(text)
        return atlas.width(text) * size / BASE_PX

    # --- frame ----------------------------------------------------------------------------
    def render(self, fbo, width: int, height: int, store, path, head_m: float, timezone: str):
        self.last_stats = None
        if store is None:
            return
        # adding glyphs repacks the atlas, which invalidates the uvs of text already in the
        # batch: build again until no new glyphs were needed
        for _ in range(4):
            generation = self.atlas_provider().generation
            batch = self._build(width, height, store, path, head_m, timezone)
            if self.atlas_provider().generation == generation:
                break
        vertices = batch.array()
        if len(vertices):
            self._submit(fbo, vertices, width, height, store, pixel_scale(width, height))

    def _build(self, width: int, height: int, store, path, head_m: float, timezone: str):
        batch = _Batch()
        k = pixel_scale(width, height)
        if store["minimap.visible"] and store["minimap.opacity"] > 0.0:
            self._draw_minimap(batch, width, height, k, store)
        attribution_h = 0.0
        if store["stats.attribution"] and self.attribution:
            attribution_h = self._draw_attribution(batch, width, height, k, store)
        opacity = store["stats.opacity"]
        if store["stats.visible"] and opacity > 0.0:
            stats = self.stats_for(path, timezone)
            if stats is not None:
                head = head_m if math.isfinite(head_m) else float(stats.dist[-1])
                values = stats.at(head)
                if values is not None and values.time is not None:
                    values.time = self._local_time(values.time, store, path, head)
                self.last_stats = values
                if values is not None:
                    self._draw_panel(batch, stats, values, head_m, width, height, k, store,
                                     attribution_h)  # fmt: skip
        return batch

    def _local_time(self, t: datetime, store, path, head: float) -> datetime:
        """The head's time in the project time zone, or (``stats.timezone`` = "position") in
        the time zone where the head is (journeys across time zones)."""
        if store["stats.timezone"] != "position" or self.frame is None:
            return t
        position = path.position_at(head)
        if position is None:
            return t
        from earthling.core.timezones import timezone_at

        lat, lon, _ = self.frame.enu_to_geodetic(np.asarray(position, dtype=np.float64))
        return t.astimezone(timezone_at(float(lon), float(lat)))

    # --- mini-map --------------------------------------------------------------------------
    def _minimap_geometry(self):
        path = self.map_path
        if path is None or not path.tracks or self.frame is None:
            return None
        key = (id(path), id(self.frame), path.total_m)
        if key != self._map_key:
            from earthling.render.minimap import build

            water, borders = self.map_sources
            try:
                self._map_geometry = build(path, self.frame, water, borders)
            except Exception:  # (network or data problems: a map without the background)
                self._map_geometry = build(path, self.frame)
            self._map_key = key
        return self._map_geometry

    def _draw_minimap(self, batch, width, height, k, store) -> None:
        geo = self._minimap_geometry()
        if geo is None:
            return
        s = k * store["minimap.scale"]
        alpha = store["minimap.opacity"]
        w = store["minimap.size"] * s
        h = w * geo.aspect
        margin = 28.0 * k
        corner = store["minimap.position"]
        x0 = margin if corner.endswith("left") else width - margin - w
        y0 = margin if corner.startswith("top") else height - margin - h

        def at(xy):
            return np.column_stack([x0 + xy[..., 0].ravel() * w, y0 + xy[..., 1].ravel() * h])

        frame_px = 1.5 * s
        batch.quad(x0 - frame_px, y0 - frame_px, x0 + w + frame_px, y0 + h + frame_px,
                   (0.95, 0.95, 0.95), 0.8 * alpha)  # fmt: skip
        batch.quad(x0, y0, x0 + w, y0 + h, store["minimap.land_color"], alpha)
        if len(geo.water):
            batch.triangles(at(geo.water), store["minimap.water_color"], alpha)
        for a, b in geo.borders:
            _segment(batch, at(np.array([a, b])), 1.0 * s, (0.75, 0.78, 0.82), 0.7 * alpha)
        heads = self.map_heads
        for index, xy, dist in geo.routes:
            color = self.map_colors.get(index)
            if color is None:
                continue
            pts = at(xy)
            head = max((hd for hd in heads if dist[0] - 1.0 <= hd), default=-1.0)
            done = dist <= head
            _polyline(batch, pts, 2.0 * s, (0.9, 0.9, 0.9), 0.3 * alpha)  # the whole route
            if done.sum() >= 1:
                n = int(done.sum())
                line = pts[:n]
                if n < len(pts):  # up to the head exactly
                    f = (head - dist[n - 1]) / max(dist[n] - dist[n - 1], 1e-6)
                    line = np.vstack([line, pts[n - 1] + (pts[n] - pts[n - 1]) * f])
                _polyline(batch, line, 5.0 * s, (0.05, 0.05, 0.05), 0.6 * alpha)  # outline
                _polyline(batch, line, 3.2 * s, color, alpha)
        for hd in heads:  # the dots at the heads
            for _index, xy, dist in geo.routes:
                if dist[0] - 1.0 <= hd <= dist[-1] + 1.0:
                    p = at(np.array([[np.interp(hd, dist, xy[:, 0]),
                                      np.interp(hd, dist, xy[:, 1])]]))[0]  # fmt: skip
                    _disc(batch, p, 5.5 * s, (0.05, 0.05, 0.05), alpha)
                    _disc(batch, p, 4.0 * s, (1.0, 1.0, 1.0), alpha)
                    break

    def _draw_attribution(self, batch, width, height, k, store) -> float:
        size = 11.0 * k
        margin = 8.0 * k
        self._text(batch, self.attribution, width - margin, height - margin - 3.0 * k, size,
                   store["stats.text_color"], 0.85, align="right")  # fmt: skip
        return size * 1.6

    def _draw_panel(self, batch, stats, v: StatsAt, head_m, width, height, k, store, below):
        s = k * store["stats.scale"]
        alpha = store["stats.opacity"]
        color = store["stats.text_color"]
        accent = store["stats.accent_color"]
        pad, gap = 16.0 * s, 26.0 * s
        day_scope = store["stats.scope"] == "day"
        # --- content sizes
        lang = store["stats.language"]
        day_word = "days" if store["stats.day_style"] == "count" else "day"
        title = word(lang, day_word).format(n=v.day) if store["stats.show_day"] else ""
        subtitle = ""
        if store["stats.show_date"] and v.time is not None:
            subtitle = format_date(v.time, store["stats.date_format"], lang)
            if store["stats.show_time"]:
                subtitle += "  " + word(lang, "time").format(t=v.time)
                if store["stats.show_utc_offset"]:
                    from earthling.core.timezones import utc_offset_text

                    subtitle += f" ({utc_offset_text(v.time)})"
        title_size, sub_size = 28.0 * s, 19.0 * s
        text = store["stats.text"].strip()
        # (text, caption, template): the template is the widest the value can get, so the
        # panel keeps its size while the numbers change
        columns = []
        if store["stats.show_distance"]:
            d = v.day_distance_m if day_scope else v.distance_m
            columns.append((f"{format_number(d / 1000.0, 1, lang)} km", word(lang, "distance"),
                            f"{format_number(8888.8, 1, lang)} km"))  # fmt: skip
        if store["stats.show_ascent"]:
            a = v.day_ascent_m if day_scope else v.ascent_m
            columns.append((f"{format_number(a, 0, lang)} m", word(lang, "ascent"),
                            f"{format_number(88888, 0, lang)} m"))  # fmt: skip
        if store["stats.show_elevation"] and v.elevation_m is not None:
            columns.append((f"{format_number(v.elevation_m, 0, lang)} m", word(lang, "elevation"),
                            f"{format_number(8888, 0, lang)} m"))  # fmt: skip
        if store["stats.show_speed"] and v.speed_kmh is not None:
            unit = store["stats.speed_unit"]
            text_, caption_ = format_speed(v.speed_kmh, unit, lang)
            widest = {"pace": "88:88 /km", "knots": f"{format_number(88.8, 1, lang)} kn"}.get(
                unit, f"{format_number(888.8, 1, lang)} km/h"
            )
            columns.append((text_, caption_, widest))
        if store["stats.show_gradient"] and v.gradient is not None:
            columns.append((f"{format_number(v.gradient, 1, lang)} %", word(lang, "gradient"),
                            f"−{format_number(88.8, 1, lang)} %"))  # fmt: skip
        if store["stats.show_heart_rate"] and v.heart_rate is not None:
            columns.append((f"{v.heart_rate:.0f} bpm", word(lang, "heart_rate"), "888 bpm"))
        if store["stats.show_gap"] and self.gap_s is not None:
            columns.append((format_gap(self.gap_s), word(lang, "gap"), "+8:88:88"))
        value_size, caption_size = 26.0 * s, 13.0 * s
        widths = [max(self._width(t, value_size), self._width(w, value_size),
                      self._width(c, caption_size)) for t, c, w in columns]  # fmt: skip
        columns = [(t, c) for t, c, _ in columns]
        values_w = sum(widths) + gap * max(len(columns) - 1, 0)
        title_w = self._width(_widest(title), title_size) + (
            self._width(_widest(subtitle, lang), sub_size) + 14.0 * s if subtitle else 0.0
        )
        show_profile = store["stats.profile"] and stats.has_elevation
        profile_w = store["stats.profile_width"] * s if show_profile else 0.0
        profile_h = store["stats.profile_height"] * s if show_profile else 0.0
        inner_w = max(title_w, values_w, profile_w, 1.0)
        if text:
            inner_w = max(inner_w, self._width(_widest(text), sub_size))
        rows = []
        if title or subtitle:
            rows.append(("title", title_size * 1.25))
        if text:
            rows.append(("text", sub_size * 1.3))
        if columns:
            rows.append(("values", value_size * 1.15 + caption_size * 1.4))
        if show_profile:
            rows.append(("profile", profile_h + 6.0 * s))
        if not rows:
            return
        inner_h = sum(h for _, h in rows) + 10.0 * s * (len(rows) - 1)
        panel_w, panel_h = inner_w + 2 * pad, inner_h + 2 * pad
        margin = 28.0 * k
        corner = store["stats.position"]
        x0 = margin if corner.endswith("left") else width - margin - panel_w
        y0 = margin if corner.startswith("top") else height - margin - below - panel_h
        background = store["stats.background"]
        if background > 0:
            batch.quad(x0, y0, x0 + panel_w, y0 + panel_h, (0.0, 0.0, 0.0), background * alpha)
        x, y = x0 + pad, y0 + pad
        for kind, row_h in rows:
            if kind == "title":
                base = y + title_size
                w = self._text(batch, title, x, base, title_size, color, alpha) if title else 0.0
                if subtitle:
                    self._text(batch, subtitle, x + w + (14.0 * s if title else 0.0), base,
                               sub_size, color, 0.8 * alpha)  # fmt: skip
            elif kind == "text":
                self._text(batch, text, x, y + sub_size, sub_size, accent, alpha)
            elif kind == "values":
                cx = x
                for (text, caption), w in zip(columns, widths, strict=True):
                    self._text(batch, text, cx, y + value_size, value_size, color, alpha)
                    caption_base = y + value_size * 1.15 + caption_size * 1.1
                    self._text(batch, caption.upper(), cx, caption_base, caption_size, color,
                               0.6 * alpha)  # fmt: skip
                    cx += w + gap
            else:
                self._draw_profile(batch, stats, v, head_m, x, y + 3.0 * s, inner_w, profile_h,
                                   accent, color, alpha, s, day_scope, store)  # fmt: skip
            y += row_h + 10.0 * s

    def _draw_profile(self, batch, stats, v, head_m, x, y, w, h, accent, color, alpha, s, day,
                      store=None):  # fmt: skip
        ground, ele, prog = stats.profile(v.day if day else None)
        if len(ground) < 2 or ground[-1] <= ground[0]:
            return
        n = max(8, int(w / 2))
        gx = np.linspace(ground[0], ground[-1], n)
        ge = np.interp(gx, ground, ele)
        lo, hi = float(ge.min()), float(ge.max())
        span = max(hi - lo, 50.0)
        lo -= span * 0.08
        px = x + (gx - gx[0]) / (gx[-1] - gx[0]) * w
        py = y + h - (ge - lo) / (span * 1.12) * h
        bottom = y + h
        walked = float(np.interp(head_m, prog, ground)) if math.isfinite(head_m) else ground[-1]
        # filled area: walked part in the accent colour, the rest faint
        base = np.full(n - 1, bottom)
        top0, top1 = np.c_[px[:-1], py[:-1]], np.c_[px[1:], py[1:]]
        bot0, bot1 = np.c_[px[:-1], base], np.c_[px[1:], base]
        quads = np.stack([top0, top1, bot1, top0, bot1, bot0], axis=1)
        done = gx[1:] <= walked
        color_by = store["tracks.color_by"] if store is not None else "none"
        if done.any() and color_by != "none":
            # the walked part in the colour ramp of the tracks (e.g. steep sections red)
            from earthling.render.colormaps import colormap
            from earthling.render.tracks import VALUE_KINDS

            column = stats.values[:, VALUE_KINDS.index(color_by)]
            if day:
                column = column[stats.day == v.day]
            values = np.interp(gx[1:], ground, np.nan_to_num(column))
            lo, hi = store["tracks.color_min"], store["tracks.color_max"]
            rgb = colormap(store["tracks.color_ramp"], (values - lo) / max(hi - lo, 1e-6))
            for q, c in zip(quads[done], rgb[done], strict=True):
                batch.triangles(q, tuple(float(x) for x in c), 0.75 * alpha)
        elif done.any():
            batch.triangles(quads[done].reshape(-1, 2), accent, 0.45 * alpha)
        if (~done).any():
            batch.triangles(quads[~done].reshape(-1, 2), color, 0.12 * alpha)
        # outline of the profile
        thickness = 1.6 * s
        dx, dy = np.diff(px), np.diff(py)
        length = np.maximum(np.hypot(dx, dy), 1e-6)
        nx, ny = -dy / length * thickness / 2, dx / length * thickness / 2
        a0 = np.c_[px[:-1] + nx, py[:-1] + ny]
        a1 = np.c_[px[1:] + nx, py[1:] + ny]
        b0 = np.c_[px[:-1] - nx, py[:-1] - ny]
        b1 = np.c_[px[1:] - nx, py[1:] - ny]
        line = np.stack([a0, a1, b1, a0, b1, b0], axis=1)
        if done.any():
            batch.triangles(line[done].reshape(-1, 2), accent, alpha)
        if (~done).any():
            batch.triangles(line[~done].reshape(-1, 2), color, 0.5 * alpha)
        # current position: vertical line and a diamond on the profile
        mx = x + (walked - gx[0]) / (gx[-1] - gx[0]) * w
        my = float(np.interp(mx, px, py))
        batch.quad(mx - 0.8 * s, my, mx + 0.8 * s, bottom, color, 0.7 * alpha)
        r = 5.5 * s
        batch.polygon([(mx, my - r - 1.5 * s), (mx + r + 1.5 * s, my), (mx, my + r + 1.5 * s),
                       (mx - r - 1.5 * s, my)], (0.0, 0.0, 0.0), 0.6 * alpha)  # fmt: skip
        batch.polygon([(mx, my - r), (mx + r, my), (mx, my + r), (mx - r, my)], (1.0, 1.0, 1.0),
                      alpha)  # fmt: skip

    def _submit(self, fbo, vertices: np.ndarray, width: int, height: int, store, k) -> None:
        atlas: FontAtlas = self.atlas_provider()
        program = self.shaders.get("hud")
        data = vertices.tobytes()
        if self._buffer is None or self._buffer.size < len(data):
            if self._buffer is not None:
                self._buffer.release()
                for vao in self._vaos.values():
                    vao.release()
                self._vaos.clear()
            self._buffer = self.ctx.buffer(reserve=max(len(data), 1 << 16) * 2)
        self._buffer.write(data)
        vao = self._vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(
                program, [(self._buffer, "2f 2f 4f 2f", "in_pos", "in_uv", "in_color", "in_params")]
            )
            self._vaos[program.glo] = vao
        if atlas.texture is None:  # (shapes only, no text yet: the atlas still needs a texture)
            atlas.ensure(" ")
        atlas.texture.use(0)
        for name, value in (
            ("u_atlas", 0),
            ("u_viewport", (float(width), float(height))),
            ("u_spread", float(SPREAD)),
            ("u_outline_px", float(1.6 * k * store["stats.scale"])),
            ("u_outline_alpha", float(store["stats.outline"])),
        ):
            if name in program:
                program[name] = value
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        vao.render(moderngl.TRIANGLES, vertices=len(vertices))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        if self._buffer is not None:
            self._buffer.release()
        for vao in self._vaos.values():
            vao.release()
