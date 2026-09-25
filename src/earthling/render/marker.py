"""Hiker marker: a pulsing, glowing billboard at the head of the drawn track."""

from __future__ import annotations

import moderngl
import numpy as np

from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary


class MarkerLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.position: np.ndarray | None = None  # ENU, None = hidden
        self.color = (1.0, 0.95, 0.8)
        self._vaos: dict[int, moderngl.VertexArray] = {}

    def render(
        self,
        camera: Camera,
        view_proj,
        width: int,
        height: int,
        time_s: float,
        store=None,
        glow_pass: bool = False,
    ) -> None:
        if self.position is None:
            return
        defines = {"GLOW_PASS": 1} if glow_pass else None
        program = self.shaders.get("marker", defines=defines)
        vao = self._vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(program, [])
            self._vaos[program.glo] = vao
        program["u_view_proj"].write(view_proj)
        _set(program, "u_center", camera.relative(self.position))
        _set(program, "u_viewport", (float(width), float(height)))
        _set(program, "u_log_depth_coef", camera.log_depth_coef)
        _set(program, "u_time", float(time_s))
        _set(program, "u_color", tuple(c**2.2 for c in self.color))
        if store is not None:
            from earthling.core.properties import bind_uniforms

            bind_uniforms(program, store)
        self.ctx.enable(moderngl.BLEND)
        if glow_pass:
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        else:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


def _set(program: moderngl.Program, name: str, value) -> None:
    if name in program:
        program[name] = value
