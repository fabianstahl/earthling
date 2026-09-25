"""Bloom for emissive objects: mip-chain downsample + tent upsample (additive).

The emissive buffer (only glowing objects are drawn into it) is downsampled into a chain of
half-resolution levels and upsampled back, accumulating into level 0. The result is added to
the HDR image in the tonemap pass.
"""

from __future__ import annotations

import moderngl

MAX_LEVELS = 7


class Bloom:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.size = (0, 0)
        self.levels: list[tuple[moderngl.Texture, moderngl.Framebuffer]] = []

    def ensure(self, width: int, height: int) -> None:
        size = (max(2, width // 2), max(2, height // 2))
        if size == self.size and self.levels:
            return
        self.release()
        w, h = size
        for _ in range(MAX_LEVELS):
            tex = self.ctx.texture((w, h), 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self.levels.append((tex, self.ctx.framebuffer([tex])))
            if w <= 4 or h <= 4:
                break
            w, h = max(1, w // 2), max(1, h // 2)
        self.size = size

    def run(
        self, source: moderngl.Texture, fullscreen, radius: float, levels: int
    ) -> moderngl.Texture:
        """``fullscreen(name) -> program`` draws a fullscreen pass (the renderer's helper)."""
        levels = max(1, min(levels, len(self.levels)))
        down = fullscreen("bloom_down")
        src = source
        for tex, fbo in self.levels[:levels]:
            fbo.use()
            self.ctx.viewport = (0, 0, *tex.size)
            src.use(0)
            down["u_source"] = 0
            down["u_texel"] = (1.0 / src.size[0], 1.0 / src.size[1])
            fullscreen.draw("bloom_down")
            src = tex
        up = fullscreen("bloom_up")
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(levels - 1, 0, -1):
            small, _ = self.levels[i]
            _, target = self.levels[i - 1]
            target.use()
            self.ctx.viewport = (0, 0, *self.levels[i - 1][0].size)
            small.use(0)
            up["u_source"] = 0
            up["u_texel"] = (1.0 / small.size[0], 1.0 / small.size[1])
            up["u_radius"] = radius
            up["u_weight"] = 1.0
            fullscreen.draw("bloom_up")
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        return self.levels[0][0]

    def release(self) -> None:
        for tex, fbo in self.levels:
            fbo.release()
            tex.release()
        self.levels = []
        self.size = (0, 0)
