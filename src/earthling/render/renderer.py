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
from earthling.render.bloom import Bloom
from earthling.render.camera import Camera
from earthling.render.hud import HudLayer
from earthling.render.labels import LabelLayer
from earthling.render.layers import generate_glsl, required_tile_sources
from earthling.render.lighting import (
    OPTICAL_DEPTH_UNIT,
    Lighting,
    compute_lighting,
    enu_to_celestial,
    lighting_uniforms,
)
from earthling.render.marker import MarkerLayer
from earthling.render.overlays import OutlineLayer
from earthling.render.shader_library import ShaderLibrary
from earthling.render.shadows import SHADOW_UNIT, ShadowMaps, compute_cascades
from earthling.render.terrain import TerrainLayer
from earthling.render.tracks import TrackGeometryOptions, TrackLayer


class FullscreenPasses:
    """Fullscreen-triangle programs sharing one vertex shader: ``passes(name)`` returns the
    program, ``passes.draw(name)`` draws it. Empty VAOs are rebuilt after hot reloads."""

    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self._cache: dict[str, tuple[moderngl.Program, moderngl.VertexArray]] = {}

    def __call__(self, name: str) -> moderngl.Program:
        program = self.shaders.get(name, vertex="fullscreen")
        cached = self._cache.get(name)
        if cached is None or cached[0] is not program:
            self._cache[name] = (program, self.ctx.vertex_array(program, []))
        return program

    def draw(self, name: str) -> None:
        self._cache[name][1].render(moderngl.TRIANGLES, vertices=3)


class SceneTarget:
    """Offscreen HDR color + depth render target that follows the output size."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.size = (0, 0)
        self.color: moderngl.Texture | None = None
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.glow: moderngl.Texture | None = None
        self.glow_fbo: moderngl.Framebuffer | None = None  # emissive objects, shares depth
        self.glow_clear_fbo: moderngl.Framebuffer | None = None

    def ensure(self, width: int, height: int) -> moderngl.Framebuffer:
        size = (max(1, width), max(1, height))
        if size != self.size or self.fbo is None:
            self.release()
            self.color = self.ctx.texture(size, 4, dtype="f2")  # linear HDR radiance
            self.depth = self.ctx.depth_texture(size)
            self.fbo = self.ctx.framebuffer([self.color], self.depth)
            self.glow = self.ctx.texture(size, 4, dtype="f2")
            self.glow.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.glow.repeat_x = self.glow.repeat_y = False
            # drawing: shares the scene depth (glowing objects redraw at their own depth,
            # so depth writes are harmless); clearing: colour only
            self.glow_fbo = self.ctx.framebuffer([self.glow], self.depth)
            self.glow_clear_fbo = self.ctx.framebuffer([self.glow])
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
        for obj in (
            self.glow_clear_fbo,
            self.glow_fbo,
            self.glow,
            self.fbo,
            self.color,
            self.depth,
        ):
            if obj is not None:
                obj.release()
        self.fbo = self.color = self.depth = self.glow = self.glow_fbo = self.glow_clear_fbo = None


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
        self.camera_path = OutlineLayer(ctx, self.shaders)  # animated camera path gizmo
        self.marker = MarkerLayer(ctx, self.shaders)
        self.labels = LabelLayer(ctx, self.shaders)
        self.hud = HudLayer(ctx, self.shaders, self.labels.get_atlas)
        self.shadows = ShadowMaps(ctx)
        self.time = 0.0  # animation time in seconds (drives pulsing effects)
        self.jitter_px = (0.0, 0.0)  # sub-pixel projection offset (export anti-aliasing)
        # temporary replacements of property values (preview / export quality)
        self.overrides: dict[str, object] = {}
        self.store: PropertyStore | None = None
        self.timezone = "UTC"
        self.lighting: Lighting | None = None
        self.camera_height = 1500.0
        self.fullscreen = FullscreenPasses(ctx, self.shaders)
        self.bloom = Bloom(ctx)
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

    def set_labels(self, data, session=None) -> None:
        """Label features (LabelData) plus what enriches them: DEM heights and the tracks."""
        self.labels.data = data
        if data is not None:  # the Data > on-demand switch also covers the label download
            data.allow_download = bool(getattr(self.terrain.data, "on_demand", True))
        self.labels.enricher = self.labels.make_enricher(
            self.frame, self.terrain.data, self.tracks.tracks
        )

    def scene_bounds(self):
        return self.terrain.bounds() or self.tracks.bounds

    def value(self, pid: str):
        """A property value, honouring quality overrides."""
        if pid in self.overrides:
            return self.overrides[pid]
        return self.store[pid]

    def apply_properties(self, camera: Camera) -> None:
        """Push the current property values into the render layers."""
        s = self.store
        if s is None:
            return
        camera.fov_y = s["view.fov"]
        if s["terrain.exaggeration"] != self.terrain.exaggeration:
            self.terrain.exaggeration = s["terrain.exaggeration"]
            self.terrain.invalidate_bounds()
        self.terrain.params.pixel_threshold = self.value("terrain.detail")
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
    def reset_state(self) -> None:
        """Known GL state: the context is shared with Qt's widget compositing, which leaves
        blending enabled between frames."""
        self.ctx.enable_only(moderngl.NOTHING)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.depth_func = "<"

    def prepare(self, camera: Camera, width: int, height: int, timeout_s: float = 300.0) -> bool:
        """Load everything the frame needs at full detail (camera view and shadow casters),
        so an exported frame never shows lower-resolution tiles or pop-in."""
        self.reset_state()
        self.apply_properties(camera)
        if self.store is not None and self.store["labels.visible"]:
            self.labels.load_sync()
        if not self.terrain.visible:
            return True
        view_proj = camera.view_projection(width / max(1, height))
        ok = self.terrain.finish_loading(camera, view_proj, height, timeout_s)
        s = self.store
        lit = self.lighting is not None and max(self.lighting.sun_radiance) > 0.0
        if ok and s is not None and s["shadows.enabled"] and lit:
            setup = compute_cascades(
                camera,
                width / max(1, height),
                self.lighting.sun_direction,
                s["shadows.distance"] * 1000.0,
                int(self.value("shadows.resolution")),
            )
            light_vp = setup.cascades[-1].view_proj
            ok = self.terrain.finish_loading(camera, light_vp, height, timeout_s, coarse=True)
        return ok

    def render(
        self,
        fbo: moderngl.Framebuffer,
        width: int,
        height: int,
        camera: Camera,
        scale: float = 1.0,
    ) -> None:
        """Render into ``fbo`` (width x height); ``scale`` < 1 renders the scene at a lower
        internal resolution (preview quality) and upscales in the tonemap pass."""
        output = (width, height)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
        self.reset_state()
        self.apply_properties(camera)
        view_proj = camera.view_projection(width / max(1, height))
        if self.jitter_px != (0.0, 0.0):
            jx, jy = self.jitter_px
            shift = glm.vec3(2.0 * jx / width, 2.0 * jy / height, 0.0)
            view_proj = glm.translate(glm.mat4(1.0), shift) * view_proj
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
        self.ctx.disable(moderngl.DEPTH_TEST)  # the camera path is drawn on top
        self.camera_path.render(camera, view_proj)
        bloom = self._render_glow(camera, view_proj, width, height)
        self._tonemap(fbo, *output, bloom)
        self.labels.render(fbo, camera, view_proj, self.frame, self.terrain.exaggeration,
                           *output, self.target.depth, (width, height), self.store)  # fmt: skip
        self.hud.render(fbo, *output, self.store, self.tracks.path, self.tracks.head_m,
                        self.timezone)  # fmt: skip
        self.reset_state()  # leave the context clean for Qt

    def _render_shadows(self, camera: Camera, width: int, height: int) -> dict[str, object]:
        s = self.store
        lit = self.lighting is not None and max(self.lighting.sun_radiance) > 0.0
        if s is None or not s["shadows.enabled"] or not lit or not self.terrain.visible:
            self.shadows.setup = None
            return self.shadows.uniforms()
        assert self.lighting is not None
        self.shadows.ensure(int(self.value("shadows.resolution")))
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
        self._update_progress()
        if self.lighting is not None:
            self.tracks.uniforms = {**lighting_uniforms(self.lighting)}
        self.tracks.render(camera, view_proj, width, height, s)
        if self.tracks.visible:
            self.marker.render(camera, view_proj, width, height, self.time, s)

    def _update_progress(self) -> None:
        """Visible range of the hike and the marker position from the progress properties."""
        s = self.store
        path = self.tracks.path
        if s is None or path is None or path.total_m <= 0:
            self.tracks.head_m, self.tracks.tail_m = float("inf"), 0.0
            self.marker.position = None
            return
        mode = s["progress.mode"]
        head = path.distance_for(s["progress.head"], mode)
        tail = min(path.distance_for(s["progress.tail"], mode), head)
        self.tracks.head_m = head if s["progress.head"] < 1.0 else float("inf")
        self.tracks.tail_m = tail
        self.marker.color = s["marker.color"]
        self.marker.position = path.position_at(head) if s["marker.visible"] else None

    def read_depth(self, px: int, py: int) -> float | None:
        """Log depth of the last frame at pixel (px, py) (GL convention, y up)."""
        return self.target.read_depth(px, py)

    # --- passes -----------------------------------------------------------------------
    def _render_sky(self, camera: Camera, view_proj, height: int) -> None:
        if self.lighting is None or self.frame is None:
            return
        program = self.fullscreen("sky")
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
        self.fullscreen.draw("sky")

    def _render_glow(self, camera: Camera, view_proj, width: int, height: int):
        """Emissive objects into the glow buffer, then bloom. Returns the bloom texture."""
        s = self.store
        if s is None or not s["glow.enabled"] or self.target.glow_fbo is None:
            return None
        self.target.glow_clear_fbo.clear(0.0, 0.0, 0.0, 0.0)
        glow_fbo = self.target.glow_fbo
        glow_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST)  # test against the scene depth ...
        self.ctx.depth_func = "<="  # ... which already contains the tracks themselves
        self.tracks.render(camera, view_proj, width, height, s, glow_pass=True)
        if self.tracks.visible:
            self.marker.render(camera, view_proj, width, height, self.time, s, glow_pass=True)
        self.ctx.depth_func = "<"
        self.ctx.disable(moderngl.DEPTH_TEST)
        assert self.target.glow is not None
        return self.bloom_pass(self.target.glow, width, height)

    def bloom_pass(self, source: moderngl.Texture, width: int, height: int) -> moderngl.Texture:
        s = self.store
        self.bloom.ensure(width, height)
        return self.bloom.run(source, self.fullscreen, s["glow.radius"], s["glow.levels"])

    def _tonemap(self, fbo: moderngl.Framebuffer, width: int, height: int, bloom=None) -> None:
        program = self.fullscreen("tonemap")
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        assert self.target.color is not None
        self.target.color.use(0)
        program["u_hdr"] = 0
        if bloom is not None:
            bloom.use(1)
        _set(program, "u_bloom", 1)
        _set(program, "u_has_bloom", bloom is not None)
        if self.store is not None:
            bind_uniforms(program, self.store)
            program["u_exposure"] = 2.0 ** self.store["post.exposure"]
        self.fullscreen.draw("tonemap")


def _set(program: moderngl.Program, name: str, value) -> None:
    if name in program:
        program[name] = value
