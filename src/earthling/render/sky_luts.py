"""Precomputed atmospheric scattering (Hillaire 2020) on the GPU.

* transmittance LUT (256 x 64) and multiple-scattering LUT (32 x 32): depend only on the medium
  (air density and haze), recomputed when those change;
* sky-view LUT (192 x 108): the whole sky around the camera, every frame;
* aerial perspective volume (32^3 froxels over the view frustum, in-scattering and
  transmittance up to AP_MAX_DIST), every frame; terrain and tracks look it up per pixel.

Shaders: atmosphere_lut.glsl and atm_*.{frag,comp}.
"""

from __future__ import annotations

import moderngl

TRANSMITTANCE_UNIT = 7
MULTISCATTER_UNIT = 8
SKYVIEW_UNIT = 9
AP_INSCATTER_UNIT = 10
AP_TRANSMITTANCE_UNIT = 11
AP_SIZE = 32
AP_MAX_DIST = 300e3  # metres covered by the aerial perspective volume


def _texture2d(ctx: moderngl.Context, size: tuple[int, int]) -> moderngl.Texture:
    tex = ctx.texture(size, 4, dtype="f2")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    return tex


class AtmosphereLuts:
    def __init__(self, ctx: moderngl.Context, shaders, passes) -> None:
        """``passes``: the renderer's FullscreenPasses."""
        self.ctx = ctx
        self.shaders = shaders
        self.passes = passes
        self.transmittance = _texture2d(ctx, (256, 64))
        self.multiscatter = _texture2d(ctx, (32, 32))
        self.skyview = _texture2d(ctx, (192, 108))
        self._fbos = {
            name: ctx.framebuffer([tex])
            for name, tex in (
                ("transmittance", self.transmittance),
                ("multiscatter", self.multiscatter),
                ("skyview", self.skyview),
            )
        }
        self.ap_inscatter = ctx.texture3d((AP_SIZE,) * 3, 4, dtype="f2")
        self.ap_transmittance = ctx.texture3d((AP_SIZE,) * 3, 4, dtype="f2")
        for tex in (self.ap_inscatter, self.ap_transmittance):
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = tex.repeat_z = False
        self._medium: tuple[float, float] | None = None
        self.static_updates = 0  # how often the medium LUTs were rebuilt (tests)

    def uniforms(self) -> dict[str, object]:
        return {
            "u_transmittance_lut": TRANSMITTANCE_UNIT,
            "u_multiscatter_lut": MULTISCATTER_UNIT,
            "u_skyview_lut": SKYVIEW_UNIT,
            "u_ap_inscatter": AP_INSCATTER_UNIT,
            "u_ap_transmittance": AP_TRANSMITTANCE_UNIT,
            "u_ap_max_dist": AP_MAX_DIST,
        }

    def bind(self) -> None:
        self.transmittance.use(TRANSMITTANCE_UNIT)
        self.multiscatter.use(MULTISCATTER_UNIT)
        self.skyview.use(SKYVIEW_UNIT)
        self.ap_inscatter.use(AP_INSCATTER_UNIT)
        self.ap_transmittance.use(AP_TRANSMITTANCE_UNIT)

    def _pass(self, name: str, uniforms: dict[str, object]) -> None:
        program = self.passes(f"atm_{name}")
        for key, value in {**self.uniforms(), **uniforms}.items():
            if key in program:
                program[key] = value
        fbo = self._fbos[name]
        fbo.use()
        self.ctx.viewport = (0, 0, *fbo.size)
        self.passes.draw(f"atm_{name}")

    def update(self, rayleigh: float, mie: float, camera_height: float, sun_dir, inv_view_proj):
        """Refresh the LUTs for this frame (``inv_view_proj``: camera-relative, glm.mat4)."""
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        medium = {"u_rayleigh_scale": float(rayleigh), "u_mie_scale": float(mie)}
        key = (float(rayleigh), float(mie))
        if key != self._medium:
            self._pass("transmittance", medium)
            self.transmittance.use(TRANSMITTANCE_UNIT)
            self._pass("multiscatter", medium)
            self._medium = key
            self.static_updates += 1
        self.bind()
        view = {**medium, "u_camera_height": float(camera_height),
                "u_sun_dir": tuple(float(v) for v in sun_dir)}  # fmt: skip
        self._pass("skyview", view)
        program = self.shaders.compute("atm_aerial")
        for name, value in {**self.uniforms(), **view}.items():
            if name in program:
                program[name] = value
        program["u_inv_view_proj"].write(inv_view_proj)
        self.ap_inscatter.bind_to_image(0, read=False, write=True)
        self.ap_transmittance.bind_to_image(1, read=False, write=True)
        groups = (AP_SIZE + 3) // 4
        program.run(groups, groups, groups)
        self.ctx.memory_barrier()
        self.bind()

    def release(self) -> None:
        for obj in (*self._fbos.values(), self.transmittance, self.multiscatter, self.skyview,
                    self.ap_inscatter, self.ap_transmittance):  # fmt: skip
            obj.release()
