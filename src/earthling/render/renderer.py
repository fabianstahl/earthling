"""Top level scene renderer, independent of Qt.

Frame structure:

1. sky pass (single-scattering atmosphere, sun disc, stars) into the HDR target
2. scene geometry (terrain, tracks, overlays) into the HDR target (linear radiance)
3. tonemap pass (exposure, ACES/Reinhard, sRGB, dithering) into the output framebuffer

The viewport passes the widget's framebuffer as output, the exporter an offscreen one. The HDR
target's depth texture also serves depth picking independent of the windowing system.
"""

from __future__ import annotations

import moderngl
import numpy as np
from pyglm import glm

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.core.properties import PropertyStore, bind_uniforms
from earthling.render.atmosphere import optical_depth_lut
from earthling.render.camera import Camera
from earthling.render.layers import generate_glsl, required_tile_sources
from earthling.render.lighting import (
    OPTICAL_DEPTH_UNIT,
    Lighting,
    compute_lighting,
    enu_to_celestial,
    lighting_uniforms,
)
from earthling.render.overlays import OutlineLayer
from earthling.render.shader_library import ShaderLibrary
from earthling.render.shadows import SHADOW_UNIT, ShadowMaps, compute_cascades
from earthling.render.terrain import TerrainLayer
from earthling.render.tracks import TrackGeometryOptions, TrackLayer


class SceneTarget:
    """Offscreen HDR color + depth render target that follows the output size."""

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
            self.color = self.ctx.texture(size, 4, dtype="f2")  # linear HDR radiance
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
        self.shaders.register_virtual("layers_generated.glsl", generate_glsl())
        self.clear_color = (0.0, 0.0, 0.0, 1.0)
        self.frame: LocalFrame | None = None
        self.target = SceneTarget(ctx)
        self.terrain = TerrainLayer(ctx, self.shaders)
        self.tracks = TrackLayer(ctx, self.shaders)
        self.outlines = OutlineLayer(ctx, self.shaders)
        self.shadows = ShadowMaps(ctx)
        self.store: PropertyStore | None = None
        self.timezone = "UTC"
        self.lighting: Lighting | None = None
        self.camera_height = 1500.0
        self._fullscreen: dict[str, tuple[moderngl.Program, moderngl.VertexArray]] = {}
        lut = optical_depth_lut()
        self.optical_depth = ctx.texture((lut.shape[1], lut.shape[0]), 3, lut.tobytes(), dtype="f4")
        self.optical_depth.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.optical_depth.repeat_x = self.optical_depth.repeat_y = False

    # --- scene setup ------------------------------------------------------------------
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

    def apply_properties(self, camera: Camera) -> None:
        """Push the current property values into the render layers."""
        s = self.store
        if s is None:
            return
        camera.fov_y = s["view.fov"]
        if s["terrain.exaggeration"] != self.terrain.exaggeration:
            self.terrain.exaggeration = s["terrain.exaggeration"]
            self.terrain.invalidate_bounds()
        self.terrain.params.pixel_threshold = s["terrain.detail"]
        self.terrain.memory_budget_mb = s["terrain.memory_budget_mb"]
        self.terrain.debug_lod = s["terrain.debug_lod"]
        self.terrain.store = s
        self.terrain.required_sources = required_tile_sources(
            s["layers.a"], s["layers.b"], s["layers.mix"], s["borders.overlay"]
        )
        self.outlines.visible = s["view.show_outlines"]
        self.tracks.visible = s["tracks.visible"]
        if self.frame is not None:
            height = float(self.frame.enu_to_geodetic(camera.position)[2])
            self.camera_height = height
            self.lighting = compute_lighting(
                s, self.frame.lat, self.frame.lon, self.timezone, camera_height=height
            )
            self.terrain.lighting_uniforms = lighting_uniforms(self.lighting)

    # --- frame ------------------------------------------------------------------------
    def render(self, fbo: moderngl.Framebuffer, width: int, height: int, camera: Camera) -> None:
        self.apply_properties(camera)
        view_proj = camera.view_projection(width / max(1, height))
        # the camera selection comes first so shadow casters never starve the view
        selection = self.terrain.update(camera, view_proj, height) if self.terrain.visible else None
        shadow_uniforms = self._render_shadows(camera, width, height)
        target = self.target.ensure(width, height)
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        target.clear(*self.clear_color, depth=1.0)
        self._render_sky(camera, view_proj, height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.optical_depth.use(OPTICAL_DEPTH_UNIT)
        if selection is not None:
            extra = {"u_camera_forward": tuple(float(v) for v in camera.forward), **shadow_uniforms}
            self.terrain.draw(camera, view_proj, selection.draw, extra_uniforms=extra)
        self._render_tracks(camera, view_proj, width, height)
        self.outlines.render(camera, view_proj)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._tonemap(fbo, width, height)

    def _render_shadows(self, camera: Camera, width: int, height: int) -> dict[str, object]:
        s = self.store
        lit = self.lighting is not None and max(self.lighting.sun_radiance) > 0.0
        if s is None or not s["shadows.enabled"] or not lit or not self.terrain.visible:
            self.shadows.setup = None
            return self.shadows.uniforms()
        assert self.lighting is not None
        self.shadows.ensure(int(s["shadows.resolution"]))
        setup = compute_cascades(
            camera,
            width / max(1, height),
            self.lighting.sun_direction,
            s["shadows.distance"] * 1000.0,
            self.shadows.size,
        )
        casters = self.terrain.select_shadow_casters(camera, setup.cascades[-1].view_proj, height)
        fbo = self.shadows.fbo
        assert fbo is not None
        fbo.use()
        self.ctx.viewport = (0, 0, self.shadows.size, self.shadows.size)
        fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        for index, cascade in enumerate(setup.cascades):
            self.ctx.viewport = self.shadows.viewport(index)
            self.terrain.draw_shadow_casters(camera, cascade.view_proj, casters)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.shadows.setup = setup
        assert self.shadows.depth is not None
        self.shadows.depth.use(SHADOW_UNIT)
        return self.shadows.uniforms()

    def _render_tracks(self, camera: Camera, view_proj, width: int, height: int) -> None:
        s = self.store
        data = self.terrain.data
        self.tracks.heights_at = getattr(data, "heights_at", None)
        options = TrackGeometryOptions(
            elevation_source=s["tracks.elevation"] if s else "dem",
            height_offset_m=s["tracks.height_offset"] if s else 3.0,
            exaggeration=self.terrain.exaggeration,
            smoothing=s["tracks.smoothing"] if s else 5,
        )
        self.tracks.ensure_built(options, dem_version=id(data))
        if self.lighting is not None:
            self.tracks.uniforms = {**lighting_uniforms(self.lighting)}
        self.tracks.render(camera, view_proj, width, height, s)

    def read_depth(self, px: int, py: int) -> float | None:
        """Log depth of the last frame at pixel (px, py) (GL convention, y up)."""
        return self.target.read_depth(px, py)

    # --- passes -----------------------------------------------------------------------
    def _fullscreen_pass(self, name: str) -> moderngl.Program:
        """Program for a fullscreen triangle; the empty VAO is rebuilt after hot reload."""
        program = self.shaders.get(name, vertex="fullscreen")
        cached = self._fullscreen.get(name)
        if cached is None or cached[0] is not program:
            self._fullscreen[name] = (program, self.ctx.vertex_array(program, []))
        return program

    def _draw_fullscreen(self, name: str) -> None:
        self._fullscreen[name][1].render(moderngl.TRIANGLES, vertices=3)

    def _render_sky(self, camera: Camera, view_proj, height: int) -> None:
        if self.lighting is None or self.frame is None:
            return
        program = self._fullscreen_pass("sky")
        program["u_inv_view_proj"].write(glm.inverse(view_proj))
        for name, value in lighting_uniforms(self.lighting).items():
            if name in program:
                program[name] = value
        program["u_night"] = self.lighting.night
        program["u_pixel_angle"] = float(np.radians(camera.fov_y)) / max(1, height)
        m = enu_to_celestial(self.lighting.when, self.frame.lat, self.frame.lon)
        program["u_to_celestial"].write(m.T.astype("f4").tobytes())  # column-major
        self.optical_depth.use(OPTICAL_DEPTH_UNIT)
        if self.store is not None:
            bind_uniforms(program, self.store)
        self._draw_fullscreen("sky")

    def _tonemap(self, fbo: moderngl.Framebuffer, width: int, height: int) -> None:
        program = self._fullscreen_pass("tonemap")
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        assert self.target.color is not None
        self.target.color.use(0)
        program["u_hdr"] = 0
        if self.store is not None:
            bind_uniforms(program, self.store)
            program["u_exposure"] = 2.0 ** self.store["post.exposure"]
        self._draw_fullscreen("tonemap")
