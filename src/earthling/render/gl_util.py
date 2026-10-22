"""Small moderngl helpers shared by the render layers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import moderngl


def set_uniform(program: moderngl.Program, name: str, value) -> None:
    """Set a uniform if the compiled program uses it (the compiler drops unused ones, and
    shader variants use different subsets)."""
    if name in program:
        program[name] = value


def set_uniforms(program: moderngl.Program, values: Mapping | Iterable[tuple[str, object]]) -> None:
    """:func:`set_uniform` for several (name, value) pairs or a mapping."""
    for name, value in values.items() if isinstance(values, Mapping) else values:
        if name in program:
            program[name] = value


class DynamicBuffer:
    """A vertex buffer rewritten every frame that grows when needed, with one vertex array per
    program (rebuilt when the buffer is replaced or a shader is hot-reloaded)."""

    def __init__(self, ctx: moderngl.Context, layout: str, *attributes: str,
                 min_bytes: int = 1 << 16) -> None:  # fmt: skip
        self.ctx = ctx
        self.layout = layout  # e.g. "2f 2f 3f"
        self.attributes = attributes  # shader inputs in layout order
        self.min_bytes = min_bytes
        self.buffer: moderngl.Buffer | None = None
        self._vaos: dict[int, moderngl.VertexArray] = {}

    def vertex_array(self, program: moderngl.Program, data: bytes) -> moderngl.VertexArray:
        """Upload ``data`` and return the vertex array of ``program`` reading it."""
        if self.buffer is None or self.buffer.size < len(data):
            self.release()
            self.buffer = self.ctx.buffer(reserve=max(len(data), self.min_bytes) * 2)
        self.buffer.write(data)
        vao = self._vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(program, [(self.buffer, self.layout, *self.attributes)])
            self._vaos[program.glo] = vao
        return vao

    def release(self) -> None:
        if self.buffer is not None:
            self.buffer.release()
            self.buffer = None
        for vao in self._vaos.values():
            vao.release()
        self._vaos.clear()


def draw_overlay(ctx: moderngl.Context, fbo, vao: moderngl.VertexArray, vertices: int,
                 width: int, height: int, origin: tuple[int, int] = (0, 0)) -> None:  # fmt: skip
    """Draw screen-space triangles with premultiplied alpha over ``fbo`` (no depth test), then
    restore the default blending. ``origin``: where the width x height area starts in the
    framebuffer (the export frame inside a wider viewport)."""
    fbo.use()
    ctx.viewport = (*origin, width, height)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    vao.render(moderngl.TRIANGLES, vertices=vertices)
    ctx.disable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
