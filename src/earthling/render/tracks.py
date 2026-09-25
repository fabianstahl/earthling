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

from dataclasses import dataclass

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
    if len(lat) < 3 or spacing_m <= 0:
        return lat, lon, ele
    steps = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    keep = [0]
    last = 0.0
    for i in range(1, len(lat) - 1):
        if cumulative[i] - last >= spacing_m:
            keep.append(i)
            last = cumulative[i]
    keep.append(len(lat) - 1)
    idx = np.array(keep)
    return lat[idx], lon[idx], ele[idx]


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


def build_track_positions(
    track: Track, frame: LocalFrame, options: TrackGeometryOptions, heights_at=None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per segment: ENU positions (n, 3) and cumulative distance along the track (n,).

    ``heights_at(lon, lat) -> heights`` (vectorised, NaN where unknown) provides DEM heights.
    """
    result = []
    distance_offset = 0.0
    for seg in track.segments:
        lat, lon, ele = resample(seg.lat, seg.lon, seg.ele, options.spacing_m)
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
        h = h * options.exaggeration + options.height_offset_m
        enu = frame.geodetic_to_enu(lat, lon, h)
        steps = np.linalg.norm(np.diff(enu, axis=0), axis=1)
        dist = distance_offset + np.concatenate([[0.0], np.cumsum(steps)])
        distance_offset = float(dist[-1])
        result.append((enu, dist))
    return result


def ribbon_vertices(segments: list[tuple[np.ndarray, np.ndarray]], origin: np.ndarray):
    """Vertex array (pos, prev, next, side, dist) and strip indices with primitive restart."""
    verts, indices = [], []
    base = 0
    for enu, dist in segments:
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
        for index, track in enumerate(self.tracks):
            segments = build_track_positions(track, self.frame, options, self.heights_at)
            if not segments:
                continue
            origin = segments[0][0][0].copy()
            verts, indices = ribbon_vertices(segments, origin)
            self._gpu.append(
                _GpuTrack(
                    track,
                    index,
                    origin,
                    self.ctx.buffer(verts.tobytes()),
                    self.ctx.buffer(indices.tobytes()),
                    float(segments[-1][1][-1]),
                    {},
                )
            )
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
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE  # emission adds up
        else:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        single = store is not None and store["tracks.color_mode"] == "single"
        for g in self._gpu:
            if not g.track.visible:
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
            color = store["tracks.color"] if single else track_color(g.index)
            _set(program, "u_color", tuple(c**2.2 for c in color))
            _set(program, "u_track_length", g.length_m)
            vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

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
