"""GPS tracks as screen-space extruded ribbons with tube shading.

Geometry pipeline (CPU, float64):

1. resample every segment to a minimum point spacing and smooth it,
2. take heights from the DEM (plus a height offset) or from the GPX file,
3. convert to ENU (curvature correct) and expand every point into two vertices (left/right).

The vertex shader extrudes the ribbon perpendicular to the projected line, either with a
constant screen width or a width in metres, with miter joins. Every vertex also carries the
cumulative distance along its track, which drives the animated progress in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import moderngl
import numpy as np

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track, haversine_m
from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary

PRIMITIVE_RESTART = 0xFFFFFFFF

# Distinct, saturated colors for tracks / days (sRGB).
PALETTE = [
    (1.00, 0.35, 0.20),
    (0.20, 0.70, 1.00),
    (1.00, 0.80, 0.15),
    (0.55, 0.90, 0.30),
    (0.90, 0.40, 0.90),
    (0.20, 0.90, 0.80),
    (1.00, 0.55, 0.65),
    (0.70, 0.60, 1.00),
]


def track_color(index: int) -> tuple[float, float, float]:
    return PALETTE[index % len(PALETTE)]


def resample(lat: np.ndarray, lon: np.ndarray, ele: np.ndarray, spacing_m: float):
    """Drop points closer than ``spacing_m`` to the previously kept one (keeps the ends)."""
    idx = resample_indices(lat, lon, spacing_m)
    return lat[idx], lon[idx], ele[idx]


def resample_indices(lat: np.ndarray, lon: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(lat) < 3 or spacing_m <= 0:
        return np.arange(len(lat))
    steps = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    keep = [0]
    last = 0.0
    for i in range(1, len(lat) - 1):
        if cumulative[i] - last >= spacing_m:
            keep.append(i)
            last = cumulative[i]
    keep.append(len(lat) - 1)
    return np.array(keep)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the ends (NaN-free input)."""
    if window <= 1 or len(values) < 3:
        return values
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")


@dataclass
class TrackGeometryOptions:
    elevation_source: str = "dem"  # "dem" or "gpx"
    height_offset_m: float = 3.0
    exaggeration: float = 1.0
    spacing_m: float = 5.0
    smoothing: int = 5  # moving-average window in points


@dataclass
class TrackSegmentGeometry:
    enu: np.ndarray  # (n, 3) float64
    dist: np.ndarray  # (n,) metres along the drawn (exaggerated) track
    time: np.ndarray  # (n,) seconds since the epoch, NaN where unknown
    ele: np.ndarray | None = None  # (n,) true elevation (no exaggeration / offset)
    ground: np.ndarray | None = None  # (n,) metres actually walked (3D, true heights)


def build_track_positions(
    track: Track, frame: LocalFrame, options: TrackGeometryOptions, heights_at=None
) -> list[TrackSegmentGeometry]:
    """Per segment: ENU positions, cumulative distance along the track and timestamps.

    ``heights_at(lon, lat) -> heights`` (vectorised, NaN where unknown) provides DEM heights.
    """
    result = []
    distance_offset = 0.0
    ground_offset = 0.0
    for seg in track.segments:
        idx = resample_indices(seg.lat, seg.lon, options.spacing_m)
        lat, lon, ele = seg.lat[idx], seg.lon[idx], seg.ele[idx]
        times = seg.time[idx]
        seconds = np.where(
            np.isnat(times), np.nan, times.astype("datetime64[ms]").astype(np.float64) / 1000.0
        )
        if len(lat) < 2:
            continue
        gpx_h = np.nan_to_num(ele, nan=np.nanmean(ele) if np.isfinite(ele).any() else 0.0)
        if options.elevation_source == "dem" and heights_at is not None:
            dem_h = heights_at(lon, lat)
            h = np.where(np.isfinite(dem_h), dem_h, gpx_h)
        else:
            h = gpx_h
        lat = smooth(lat, options.smoothing)
        lon = smooth(lon, options.smoothing)
        if options.elevation_source != "dem":
            h = smooth(h, options.smoothing)
        true_h = h
        h = h * options.exaggeration + options.height_offset_m
        enu = frame.geodetic_to_enu(lat, lon, h)
        steps = np.linalg.norm(np.diff(enu, axis=0), axis=1)
        dist = distance_offset + np.concatenate([[0.0], np.cumsum(steps)])
        distance_offset = float(dist[-1])
        if options.exaggeration != 1.0:
            true_steps = np.linalg.norm(
                np.diff(frame.geodetic_to_enu(lat, lon, true_h), axis=0), axis=1
            )
        else:
            true_steps = steps
        ground = ground_offset + np.concatenate([[0.0], np.cumsum(true_steps)])
        ground_offset = float(ground[-1])
        result.append(TrackSegmentGeometry(enu, dist, seconds, np.asarray(true_h, float), ground))
    return result


def ribbon_vertices(segments: list[TrackSegmentGeometry], origin: np.ndarray):
    """Vertex array (pos, prev, next, side, dist) and strip indices with primitive restart."""
    verts, indices = [], []
    base = 0
    for geometry in segments:
        enu, dist = geometry.enu, geometry.dist
        n = len(enu)
        rel = enu - origin
        prev = np.vstack([2 * rel[0] - rel[1], rel[:-1]])  # mirror the ends
        nxt = np.vstack([rel[1:], 2 * rel[-1] - rel[-2]])
        block = np.empty((n, 2, 11), dtype=np.float64)
        for k, side in enumerate((-1.0, 1.0)):
            block[:, k, 0:3] = rel
            block[:, k, 3:6] = prev
            block[:, k, 6:9] = nxt
            block[:, k, 9] = side
            block[:, k, 10] = dist
        verts.append(block.reshape(-1, 11))
        indices.append(np.arange(base, base + 2 * n, dtype=np.uint32))
        indices.append(np.array([PRIMITIVE_RESTART], dtype=np.uint32))
        base += 2 * n
    if not verts:
        return np.zeros((0, 11), dtype="f4"), np.zeros(0, dtype=np.uint32)
    return np.vstack(verts).astype("f4"), np.concatenate(indices)


@dataclass
class _GpuTrack:
    track: Track
    index: int
    origin: np.ndarray
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    length_m: float
    vaos: dict[int, moderngl.VertexArray]
    offset_m: float = 0.0  # start of this track on the global (all tracks) distance axis
    enu: np.ndarray | None = None  # all points (float64), for head positions
    dist: np.ndarray | None = None
    time: np.ndarray | None = None
    ele: np.ndarray | None = None  # true elevations
    ground: np.ndarray | None = None  # walked distance within the track


class ProgressPath:
    """The whole hike as one path: global distance (m) <-> position <-> time."""

    def __init__(self, tracks: list[_GpuTrack]) -> None:
        self.tracks = tracks
        self.total_m = tracks[-1].offset_m + tracks[-1].length_m if tracks else 0.0
        dists, times, eles, grounds, owners = [], [], [], [], []
        ground_offset = 0.0
        for g in tracks:
            dists.append(g.dist + g.offset_m)
            times.append(g.time)
            n = len(g.dist)
            eles.append(g.ele if g.ele is not None else np.full(n, np.nan))
            ground = g.ground if g.ground is not None else g.dist
            grounds.append(ground + ground_offset)
            ground_offset += float(ground[-1]) if n else 0.0
            owners.append(np.full(n, g.index))
        self.dist = np.concatenate(dists) if dists else np.zeros(0)
        self.time = np.concatenate(times) if times else np.zeros(0)
        self.ele = np.concatenate(eles) if eles else np.zeros(0)  # true elevation per point
        self.ground = np.concatenate(grounds) if grounds else np.zeros(0)  # walked metres
        self.track_index = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
        valid = np.isfinite(self.time)
        # time mode needs monotonic timestamps; otherwise fall back to distance
        self.has_time = bool(valid.sum() >= 2) and bool(np.all(np.diff(self.time[valid]) >= 0))
        self._valid = valid

    def distance_for(self, progress: float, mode: str = "distance") -> float:
        progress = min(max(progress, 0.0), 1.0)
        if mode == "time" and self.has_time:
            t = self.time[self._valid]
            d = self.dist[self._valid]
            target = t[0] + progress * (t[-1] - t[0])
            return float(np.interp(target, t, d))
        return progress * self.total_m

    def position_at(self, distance: float) -> np.ndarray | None:
        for g in self.tracks:
            if g.enu is None or g.dist is None or len(g.dist) == 0:
                continue
            if distance <= g.offset_m + g.length_m or g is self.tracks[-1]:
                local = min(max(distance - g.offset_m, 0.0), g.length_m)
                return np.array([np.interp(local, g.dist, g.enu[:, k]) for k in range(3)])
        return None

    def distance_for_time(self, seconds: float) -> float:
        """Distance of the recorded point at ``seconds`` (UTC epoch), clamped to the hike."""
        valid = self._valid
        if not valid.any():
            return 0.0
        return float(np.interp(seconds, self.time[valid], self.dist[valid]))

    def progress_for_distance(self, distance: float, mode: str = "distance") -> float:
        """Inverse of :meth:`distance_for` (the progress value that puts the head there)."""
        if self.total_m <= 0:
            return 0.0
        if mode == "time" and self.has_time:
            t = self.time[self._valid]
            seconds = float(np.interp(distance, self.dist[self._valid], t))
            return (seconds - t[0]) / max(t[-1] - t[0], 1e-9)
        return min(max(distance / self.total_m, 0.0), 1.0)

    def track_range(self, index: int) -> tuple[float, float]:
        """(start, end) distance of walked track ``index`` on the global axis."""
        g = self.tracks[index]
        return g.offset_m, g.offset_m + g.length_m

    def time_at(self, distance: float) -> float | None:
        valid = self._valid
        if not valid.any():
            return None
        return float(np.interp(distance, self.dist[valid], self.time[valid]))


class TrackLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self._gpu: list[_GpuTrack] = []
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.visible = True
        self.tracks: list[Track] = []
        self.frame: LocalFrame | None = None
        self.heights_at = None  # vectorised DEM height function
        self._built_key: tuple | None = None
        self.uniforms: dict[str, object] = {}  # lighting/atmosphere uniforms
        self.path: ProgressPath | None = None
        self.head_m = float("inf")  # global distance of the visible end (progress)
        self.tail_m = 0.0  # global distance of the visible start

    def set_tracks(self, tracks: list[Track], frame: LocalFrame) -> None:
        self.tracks = tracks
        self.frame = frame
        self._built_key = None
        self.release()
        self.bounds = self._raw_bounds()

    def _raw_bounds(self):
        if not self.tracks or self.frame is None:
            return None
        pts = []
        for t in self.tracks:
            lat, lon, ele = t.all_points()
            pts.append(self.frame.geodetic_to_enu(lat, lon, np.nan_to_num(ele)))
        allp = np.vstack(pts)
        return allp.min(axis=0), allp.max(axis=0)

    def walked(self) -> list[Track]:
        return [t for t in self.tracks if t.role == "walked"]

    def ensure_built(self, options: TrackGeometryOptions, dem_version: object = None) -> None:
        key = (
            options.elevation_source,
            options.height_offset_m,
            options.exaggeration,
            options.spacing_m,
            options.smoothing,
            dem_version,
        )
        if key == self._built_key or self.frame is None:
            return
        self.release()
        offset = 0.0
        group_index: dict[str, int] = {}
        for track in self.tracks:
            index = group_index.get(track.group, 0)
            group_index[track.group] = index + 1
            spacing = track.spacing_m or (options.spacing_m if track.role == "walked" else 25.0)
            track_options = replace(options, spacing_m=spacing)
            segments = build_track_positions(track, self.frame, track_options, self.heights_at)
            if not segments:
                continue
            origin = segments[0].enu[0].copy()
            verts, indices = ribbon_vertices(segments, origin)
            gpu = _GpuTrack(
                track,
                index,
                origin,
                self.ctx.buffer(verts.tobytes()),
                self.ctx.buffer(indices.tobytes()),
                float(segments[-1].dist[-1]),
                {},
                offset_m=offset if track.role == "walked" else 0.0,
                enu=np.vstack([s.enu for s in segments]),
                dist=np.concatenate([s.dist for s in segments]),
                time=np.concatenate([s.time for s in segments]),
                ele=np.concatenate([s.ele for s in segments]),
                ground=np.concatenate([s.ground for s in segments]),
            )
            if track.role == "walked":  # progress runs along the walked tracks only
                offset += gpu.length_m
            self._gpu.append(gpu)
        self.path = ProgressPath([g for g in self._gpu if g.track.role == "walked"])
        self._built_key = key

    def render(
        self, camera: Camera, view_proj, width: int, height: int, store=None, glow_pass=False
    ) -> None:
        if not self._gpu or not self.visible:
            return
        if glow_pass:
            program = self.shaders.get("track", defines={"GLOW_PASS": 1})
        else:
            program = self.shaders.get("track")
        program["u_view_proj"].write(view_proj)
        _set(program, "u_log_depth_coef", camera.log_depth_coef)
        _set(program, "u_viewport", (float(width), float(height)))
        focal = height / (2.0 * np.tan(np.radians(camera.fov_y) / 2.0))
        _set(program, "u_focal_px", float(focal))
        if store is not None:
            from earthling.core.properties import bind_uniforms

            bind_uniforms(program, store)
        for name, value in self.uniforms.items():
            _set(program, name, value)
        self.ctx.enable(moderngl.BLEND)
        if glow_pass:
            # emission composited "over" in draw order (premultiplied): a track overlapping
            # itself does not add up its glow, and a track on top of another one replaces its
            # glow instead of mixing both colours into a third one
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        else:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # depth test only: tracks on top of each other blend (a faint line must not hide the
        # one below), and clouds / rain keep the terrain depth
        fbo = self.ctx.fbo
        fbo.depth_mask = False
        casing = store is not None and store["tracks.casing"] > 0.0
        parts = (0, 1) if casing and not glow_pass else (1,)
        for part in parts:
            _set(program, "u_part", part)
            self._draw_tracks(program, camera, store)
        fbo.depth_mask = True
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _draw_tracks(self, program, camera: Camera, store) -> None:
        single = store is not None and store["tracks.color_mode"] == "single"
        base = {
            "opacity": store["tracks.opacity"] if store is not None else 1.0,
            "width": store["tracks.width"] if store is not None else 5.0,
            "glow": store["tracks.glow"] if store is not None else 1.2,
        }
        # planned routes first: the walked tracks are drawn on top of them
        for g in sorted(self._gpu, key=lambda g: g.track.role == "walked"):
            if not g.track.visible:
                continue
            style = self._group_style(store, g, single)
            if style is None:
                continue
            vao = g.vaos.get(program.glo)
            if vao is None:
                vao = self.ctx.vertex_array(
                    program,
                    [
                        (
                            g.vbo,
                            "3f 3f 3f 1f 1f",
                            "in_pos",
                            "in_prev",
                            "in_next",
                            "in_side",
                            "in_dist",
                        )
                    ],
                    g.ibo,
                    index_element_size=4,
                    skip_errors=True,  # attributes a shader variant does not use
                )
                g.vaos[program.glo] = vao
            _set(program, "u_offset", camera.relative(g.origin))
            color, opacity, width, glow, dash = style
            _set(program, "u_color", tuple(c**2.2 for c in color))
            _set(program, "u_track_opacity", base["opacity"] * opacity)
            _set(program, "u_track_width", base["width"] * width)
            _set(program, "u_track_glow", base["glow"] * glow)
            _set(program, "u_dash_px", dash)
            _set(program, "u_track_length", g.length_m)
            _set(program, "u_track_offset", g.offset_m)
            walked = g.track.role == "walked"  # planned routes are always drawn in full
            _set(program, "u_head_m", min(self.head_m, 1e12) if walked else 1e12)
            _set(program, "u_tail_m", self.tail_m if walked else 0.0)
            vao.render(moderngl.TRIANGLE_STRIP)

    @staticmethod
    def _group_style(store, g: _GpuTrack, single: bool):
        """(colour, opacity, width factor, glow factor, dash px) or None when hidden."""
        default_color = store["tracks.color"] if single and store is not None else None
        if default_color is None:
            default_color = track_color(g.index)
        p = f"trackgroup.{g.track.group}."
        if store is None or p + "visible" not in store.registry:
            return default_color, 1.0, 1.0, 1.0, 0.0
        if not store[p + "visible"] or store[p + "opacity"] <= 0.0:
            return None
        mode = store[p + "color_mode"]
        if mode == "single":
            color = store[p + "color"]
        elif mode == "per_track":
            pid = p + f"track{g.index}.color"
            color = store[pid] if pid in store.registry else track_color(g.index)
        else:
            color = default_color
        opacity = store[p + "opacity"]
        highlight = store[p + "highlight"]
        if highlight != "none":
            if g.track.name == highlight:
                color = store[p + "highlight_color"]
            else:
                opacity *= store[p + "dim"]
        return color, opacity, store[p + "width"], store[p + "glow"], store[p + "dash"]

    def release(self) -> None:
        for g in self._gpu:
            for vao in g.vaos.values():
                vao.release()
            g.vbo.release()
            g.ibo.release()
        self._gpu = []


def _set(program: moderngl.Program, name: str, value) -> None:
    if name in program:
        program[name] = value
