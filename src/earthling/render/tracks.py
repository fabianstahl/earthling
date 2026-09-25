"""GPU representation of GPS tracks (simple lines for now)."""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary

PRIMITIVE_RESTART = 0xFFFFFFFF

# Distinct, saturated colors for tracks / days.
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


def track_enu(track: Track, frame: LocalFrame) -> list[np.ndarray]:
    """ENU positions (float64) of every segment. Missing elevations become 0 m."""
    result = []
    for seg in track.segments:
        ele = np.nan_to_num(seg.ele, nan=0.0)
        result.append(frame.geodetic_to_enu(seg.lat, seg.lon, ele))
    return result


@dataclass
class _GpuTrack:
    track: Track
    origin: np.ndarray
    vao: moderngl.VertexArray
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    color: tuple[float, float, float]


class TrackLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self._gpu: list[_GpuTrack] = []
        self._program: moderngl.Program | None = None
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.visible = True

    def set_tracks(self, tracks: list[Track], frame: LocalFrame) -> None:
        self.release()
        program = self.shaders.get("track_line")
        self._program = program
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for index, track in enumerate(tracks):
            segments = track_enu(track, frame)
            if not segments:
                continue
            origin = segments[0][0].copy()
            verts = np.concatenate(segments) - origin
            lo = np.minimum(lo, verts.min(axis=0) + origin)
            hi = np.maximum(hi, verts.max(axis=0) + origin)
            indices, start = [], 0
            for seg in segments:
                indices.append(np.arange(start, start + len(seg), dtype=np.uint32))
                indices.append(np.array([PRIMITIVE_RESTART], dtype=np.uint32))
                start += len(seg)
            vbo = self.ctx.buffer(verts.astype("f4").tobytes())
            ibo = self.ctx.buffer(np.concatenate(indices).tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "3f", "in_pos")], ibo, index_element_size=4)
            self._gpu.append(_GpuTrack(track, origin, vao, vbo, ibo, track_color(index)))
        self.bounds = (lo, hi) if self._gpu else None

    def render(self, camera: Camera, view_proj) -> None:
        if not self._gpu or not self.visible:
            return
        program = self.shaders.get("track_line")
        if program is not self._program:
            # shader was hot-reloaded; VAOs must be rebuilt
            for g in self._gpu:
                g.vao.release()
                g.vao = self.ctx.vertex_array(
                    program, [(g.vbo, "3f", "in_pos")], g.ibo, index_element_size=4
                )
            self._program = program
        program["u_view_proj"].write(view_proj)
        program["u_log_depth_coef"] = camera.log_depth_coef
        for g in self._gpu:
            if not g.track.visible:
                continue
            program["u_offset"] = camera.relative(g.origin)
            program["u_color"] = (*g.color, 1.0)
            g.vao.render(moderngl.LINE_STRIP)

    def release(self) -> None:
        for g in self._gpu:
            g.vao.release()
            g.vbo.release()
            g.ibo.release()
        self._gpu = []
        self.bounds = None
