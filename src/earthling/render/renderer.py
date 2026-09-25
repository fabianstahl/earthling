"""Top level scene renderer, independent of Qt.

The renderer draws into any moderngl framebuffer. The viewport widget passes the widget's
default framebuffer; the exporter later passes an offscreen framebuffer.
"""

from __future__ import annotations

import moderngl
import numpy as np

from earthling.render.shader_library import ShaderLibrary


class Renderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.shaders = ShaderLibrary(ctx)
        self.clear_color = (0.08, 0.09, 0.11, 1.0)
        vertices = np.array(
            [
                # x, y, r, g, b
                -0.6, -0.5, 1.0, 0.3, 0.2,
                0.6, -0.5, 0.2, 1.0, 0.3,
                0.0, 0.6, 0.2, 0.4, 1.0,
            ],
            dtype="f4",
        )  # fmt: skip
        self._vbo = ctx.buffer(vertices.tobytes())
        self._vao: moderngl.VertexArray | None = None
        self._vao_program: moderngl.Program | None = None

    def render(self, fbo: moderngl.Framebuffer, width: int, height: int, time: float) -> None:
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        fbo.clear(*self.clear_color, depth=1.0)
        program = self.shaders.get("test_triangle")
        if self._vao_program is not program:
            self._vao = self.ctx.vertex_array(program, [(self._vbo, "2f 3f", "in_pos", "in_color")])
            self._vao_program = program
        program["u_time"] = time
        assert self._vao is not None
        self._vao.render(moderngl.TRIANGLES)
