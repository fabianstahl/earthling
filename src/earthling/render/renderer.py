"""Top level scene renderer, independent of Qt.

The scene is drawn into the renderer's own offscreen target (color + depth texture) and then
copied to the output framebuffer. The viewport widget passes the widget's framebuffer, the
exporter an offscreen one. Having our own depth texture also makes picking independent of the
windowing system.
"""

from __future__ import annotations

import moderngl
import numpy as np

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.render.camera import Camera
from earthling.render.overlays import OutlineLayer
from earthling.render.shader_library import ShaderLibrary
from earthling.render.terrain import TerrainLayer
from earthling.render.tracks import TrackLayer


class SceneTarget:
    """Offscreen color + depth render target that follows the output size."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.size = (0, 0)
        self.color: moderngl.Texture | None = None
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None

    def ensure(self, width: int, height: int) -> moderngl.Framebuffer:
        size = (max(1, width), max(1, height))
        if size != self.size or self.fbo is None:
            self.release()
            self.color = self.ctx.texture(size, 4)
            self.depth = self.ctx.depth_texture(size)
            self.fbo = self.ctx.framebuffer([self.color], self.depth)
            self.size = size
        return self.fbo

    def read_depth(self, px: int, py: int) -> float | None:
        if self.fbo is None:
            return None
        w, h = self.size
        if not (0 <= px < w and 0 <= py < h):
            return None
        raw = self.fbo.read(viewport=(px, py, 1, 1), attachment=-1, dtype="f4")
        return float(np.frombuffer(raw, dtype="f4")[0])

    def release(self) -> None:
        for obj in (self.fbo, self.color, self.depth):
            if obj is not None:
                obj.release()
        self.fbo = self.color = self.depth = None


class Renderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.shaders = ShaderLibrary(ctx)
        self.clear_color = (0.55, 0.68, 0.82, 1.0)
        self.frame: LocalFrame | None = None
        self.target = SceneTarget(ctx)
        self.terrain = TerrainLayer(ctx, self.shaders)
        self.tracks = TrackLayer(ctx, self.shaders)
        self.outlines = OutlineLayer(ctx, self.shaders)

    def set_scene(self, frame: LocalFrame, tracks: list[Track]) -> None:
        self.frame = frame
        self.tracks.set_tracks(tracks, frame)

    def set_terrain_source(self, data, nodes) -> None:
        """``data``: TerrainData, ``nodes``: lod.NodeSet (or None to clear)."""
        if self.frame is None:
            return
        self.terrain.set_source(self.frame, data, nodes)

    def scene_bounds(self):
        return self.terrain.bounds() or self.tracks.bounds

    def render(self, fbo: moderngl.Framebuffer, width: int, height: int, camera: Camera) -> None:
        target = self.target.ensure(width, height)
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        target.clear(*self.clear_color, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        view_proj = camera.view_projection(width / max(1, height))
        self.terrain.render(camera, view_proj, height)
        self.tracks.render(camera, view_proj)
        self.outlines.render(camera, view_proj)
        self.ctx.copy_framebuffer(fbo, target)

    def read_depth(self, px: int, py: int) -> float | None:
        """Log depth of the last frame at pixel (px, py) (GL convention, y up)."""
        return self.target.read_depth(px, py)
