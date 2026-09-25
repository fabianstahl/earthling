"""Top level scene renderer, independent of Qt.

The renderer draws into any moderngl framebuffer. The viewport widget passes the widget's
default framebuffer; the exporter later passes an offscreen framebuffer.
"""

from __future__ import annotations

import moderngl

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.render.camera import Camera
from earthling.render.overlays import OutlineLayer
from earthling.render.shader_library import ShaderLibrary
from earthling.render.terrain import TerrainLayer
from earthling.render.tracks import TrackLayer


class Renderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.shaders = ShaderLibrary(ctx)
        self.clear_color = (0.55, 0.68, 0.82, 1.0)
        self.frame: LocalFrame | None = None
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
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        fbo.clear(*self.clear_color, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        view_proj = camera.view_projection(width / max(1, height))
        self.terrain.render(camera, view_proj, height)
        self.tracks.render(camera, view_proj)
        self.outlines.render(camera, view_proj)
