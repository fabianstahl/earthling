"""Simple colored polylines (AOI and zone outlines, debug geometry)."""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary


@dataclass
class _Line:
    origin: np.ndarray
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray | None
    color: tuple[float, float, float, float]
    closed: bool


class OutlineLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.visible = True
        self._lines: list[_Line] = []
        self._program: moderngl.Program | None = None

    def set_lines(self, lines: list[tuple[np.ndarray, tuple[float, ...], bool]]) -> None:
        """``lines``: (ENU points (n, 3) float64, rgba color, closed) triples."""
        self.release()
        for points, color, closed in lines:
            if len(points) < 2:
                continue
            origin = points[0].copy()
            vbo = self.ctx.buffer((points - origin).astype("f4").tobytes())
            self._lines.append(_Line(origin, vbo, None, tuple(color), closed))  # type: ignore[arg-type]

    def render(self, camera: Camera, view_proj) -> None:
        if not self.visible or not self._lines:
            return
        program = self.shaders.get("track_line")
        if program is not self._program:
            for line in self._lines:
                if line.vao is not None:
                    line.vao.release()
                line.vao = None
            self._program = program
        program["u_view_proj"].write(view_proj)
        program["u_log_depth_coef"] = camera.log_depth_coef
        for line in self._lines:
            if line.vao is None:
                line.vao = self.ctx.vertex_array(program, [(line.vbo, "3f", "in_pos")])
            program["u_offset"] = camera.relative(line.origin)
            program["u_color"] = line.color
            line.vao.render(moderngl.LINE_LOOP if line.closed else moderngl.LINE_STRIP)

    def release(self) -> None:
        for line in self._lines:
            if line.vao is not None:
                line.vao.release()
            line.vbo.release()
        self._lines = []
