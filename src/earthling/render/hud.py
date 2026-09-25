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
from earthling.render.shader_library import ShaderLibrary

REFERENCE_HEIGHT = 1080.0
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]  # fmt: skip


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
        )

    def profile(self, day: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(walked metres, elevation, progress distance) of the hike or of one day."""
        sel = slice(None) if day is None else self.day == day
        return self.ground[sel], self.ele[sel], self.dist[sel]


def format_date(t: datetime, style: str) -> str:
    if style == "iso":
        return t.strftime("%Y-%m-%d")
    if style == "short":
        return f"{t.day} {MONTHS[t.month - 1][:3]}"
    return f"{t.day} {MONTHS[t.month - 1]} {t.year}"


def format_number(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ")  # thin space as thousands separator


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
        batch = _Batch()
        k = height / REFERENCE_HEIGHT
        attribution_h = 0.0
        if store["stats.attribution"] and self.attribution:
            attribution_h = self._draw_attribution(batch, width, height, k, store)
        opacity = store["stats.opacity"]
        if store["stats.visible"] and opacity > 0.0:
            stats = self.stats_for(path, timezone)
            if stats is not None:
                values = stats.at(head_m if math.isfinite(head_m) else float(stats.dist[-1]))
                self.last_stats = values
                if values is not None:
                    self._draw_panel(batch, stats, values, head_m, width, height, k, store,
                                     attribution_h)  # fmt: skip
        vertices = batch.array()
        if len(vertices):
            self._submit(fbo, vertices, width, height, store, k)

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
        title = f"Day {v.day}" if store["stats.show_day"] else ""
        subtitle = ""
        if store["stats.show_date"] and v.time is not None:
            subtitle = format_date(v.time, store["stats.date_format"])
            if store["stats.show_time"]:
                subtitle += f"  {v.time:%H:%M}"
        title_size, sub_size = 28.0 * s, 19.0 * s
        columns = []
        if store["stats.show_distance"]:
            d = v.day_distance_m if day_scope else v.distance_m
            columns.append((f"{format_number(d / 1000.0, 1)} km", "distance"))
        if store["stats.show_ascent"]:
            a = v.day_ascent_m if day_scope else v.ascent_m
            columns.append((f"{format_number(a)} m", "ascent"))
        if store["stats.show_elevation"] and v.elevation_m is not None:
            columns.append((f"{format_number(v.elevation_m)} m", "elevation"))
        value_size, caption_size = 26.0 * s, 13.0 * s
        widths = [max(self._width(t, value_size), self._width(c, caption_size))
                  for t, c in columns]  # fmt: skip
        values_w = sum(widths) + gap * max(len(columns) - 1, 0)
        title_w = self._width(title, title_size) + (
            self._width(subtitle, sub_size) + 14.0 * s if subtitle else 0.0
        )
        show_profile = store["stats.profile"] and stats.has_elevation
        profile_w = store["stats.profile_width"] * s if show_profile else 0.0
        profile_h = store["stats.profile_height"] * s if show_profile else 0.0
        inner_w = max(title_w, values_w, profile_w, 1.0)
        rows = []
        if title or subtitle:
            rows.append(("title", title_size * 1.25))
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
                                   accent, color, alpha, s, day_scope)  # fmt: skip
            y += row_h + 10.0 * s

    def _draw_profile(self, batch, stats, v, head_m, x, y, w, h, accent, color, alpha, s, day):
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
        if done.any():
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
        assert atlas.texture is not None
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
