"""Weather: volumetric cloud / fog layers (with cloud shadows), rain and thunderstorms.

The cloud noise (tileable 3D Perlin-Worley) and the 2D coverage map are generated once on the
GPU. Every frame the enabled layers are raymarched against the scene depth (clouds.frag) and
composited onto the HDR target; the terrain samples the same density for moving cloud shadows.
Wind moves the noise in world space with the timeline time, so exports are deterministic.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from earthling.render.shader_library import ShaderLibrary

CLOUD_NOISE_UNIT = 12
WEATHER_MAP_UNIT = 13
NOISE_SIZE = 128
MAP_SIZE = 512
MAX_LAYERS = 4
CLOUD_LAYERS = 3  # user layers; the rain deck is the 4th
EXTINCTION_PER_DENSITY = 0.03  # 1/m at density 1


def equalize(values: np.ndarray) -> np.ndarray:
    """uint8 values -> uniformly distributed uint8 (rank based)."""
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values))
    return (ranks / max(len(values) - 1, 1) * 255.0 + 0.5).astype(np.uint8)


def wind_vector(speed: float, direction_deg: float) -> tuple[float, float]:
    """Velocity (east, north) of a wind blowing *towards* ``direction_deg`` (0 = north)."""
    a = math.radians(direction_deg)
    return speed * math.sin(a), speed * math.cos(a)


class WeatherSystem:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, passes) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.passes = passes
        self.noise: moderngl.Texture3D | None = None
        self.map: moderngl.Texture | None = None
        self.layers: list[tuple] = []  # (shape, noise, color) per active layer
        self.time = 0.0
        self.shadow_strength = 0.0

    # --- resources ----------------------------------------------------------------------
    def _ensure_noise(self) -> None:
        if self.noise is not None:
            return
        # (mipmap filters only after the mipmaps exist: image stores into an incomplete
        # texture are discarded)
        self.noise = self.ctx.texture3d((NOISE_SIZE,) * 3, 4, dtype="f1")
        self.noise.repeat_x = self.noise.repeat_y = self.noise.repeat_z = True
        program = self.shaders.compute("weather_noise")
        self.noise.bind_to_image(0, read=False, write=True)
        groups = NOISE_SIZE // 4
        program.run(groups, groups, groups)
        self.map = self.ctx.texture((MAP_SIZE, MAP_SIZE), 1, dtype="f1")
        self.map.repeat_x = self.map.repeat_y = True
        program = self.shaders.compute("weather_map")
        self.map.bind_to_image(0, read=False, write=True)
        program.run(MAP_SIZE // 8, MAP_SIZE // 8, 1)
        self.ctx.memory_barrier()
        # histogram equalisation of the base shapes and the coverage map: "coverage" then is
        # the cloudy fraction of the sky
        noise = np.frombuffer(self.noise.read(), dtype=np.uint8).reshape(-1, 4).copy()
        noise[:, 0] = equalize(noise[:, 0])
        self.noise.write(noise.tobytes())
        cover = np.frombuffer(self.map.read(), dtype=np.uint8).copy()
        self.map.write(equalize(cover).tobytes())
        self.noise.build_mipmaps()
        self.map.build_mipmaps()
        self.noise.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.map.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

    def bind(self) -> None:
        if self.noise is not None and self.map is not None:
            self.noise.use(CLOUD_NOISE_UNIT)
            self.map.use(WEATHER_MAP_UNIT)

    # --- per frame ----------------------------------------------------------------------
    def active(self, store) -> bool:
        return store is not None and bool(store["weather.enabled"] or store["rain.enabled"])

    def raining(self, store) -> bool:
        return store is not None and bool(store["rain.enabled"]) and store["rain.intensity"] > 0

    def update(self, store, time: float) -> None:
        """Collect the enabled layers for this frame."""
        self.layers = []
        self.time = float(time)
        self.shadow_strength = 0.0
        if not self.active(store):
            return
        self._ensure_noise()
        for i in range(1, CLOUD_LAYERS + 1 if store["weather.enabled"] else 1):
            p = f"weather.layer{i}."
            if not store[p + "enabled"] or store[p + "density"] <= 0.0:
                continue
            wx, wy = wind_vector(store[p + "wind_speed"], store[p + "wind_dir"])
            self.layers.append((
                (store[p + "base"], store[p + "thickness"], store[p + "coverage"],
                 store[p + "density"] * EXTINCTION_PER_DENSITY),
                (store[p + "scale"], wx * time, wy * time, store[p + "detail"]),
                (0.95, 0.96, 1.0, store[p + "brightness"]),
            ))  # fmt: skip
        self.layers.extend(self.extra_layers(store, time))
        self.layers = self.layers[:MAX_LAYERS]
        self.shadow_strength = store["weather.cloud_shadows"] if self.layers else 0.0

    def extra_layers(self, store, time: float) -> list[tuple]:
        """The rain cloud deck (a dense, dark layer)."""
        if not store["rain.enabled"]:
            return []
        wx, wy = wind_vector(store["rain.wind"] * 2.0, store["rain.wind_dir"])
        albedo = 1.0 - 0.75 * store["rain.darkness"]
        return [(
            (store["rain.cloud_base"], store["rain.cloud_thickness"], store["rain.coverage"],
             store["rain.cloud_density"] * EXTINCTION_PER_DENSITY),
            (4500.0, wx * time, wy * time, 0.5),
            (albedo * 0.92, albedo * 0.94, albedo, 1.0),
        )]  # fmt: skip

    def rain_uniforms(self, store) -> dict[str, object]:
        """Wet ground and rain haze for terrain, tracks and clouds."""
        if not self.raining(store):
            return {"u_wetness": 0.0, "u_rain_haze": 0.0}
        visibility = max(store["rain.visibility_km"] * 1000.0, 50.0)
        return {
            "u_wetness": float(store["rain.wetness"] * min(1.0, store["rain.intensity"] * 1.5)),
            "u_rain_haze": float(3.0 / visibility * store["rain.intensity"]),
        }

    def dim_lighting(self, store, lighting) -> None:
        """Less sky light under the rain deck (the sun is handled by the cloud shadows)."""
        if not self.raining(store):
            return
        factor = 1.0 - 0.55 * store["rain.darkness"] * store["rain.coverage"]
        lighting.sky_ambient = tuple(c * factor for c in lighting.sky_ambient)
        lighting.ground_ambient = tuple(c * factor for c in lighting.ground_ambient)
        grey = sum(lighting.sky_ambient) / 3.0
        lighting.fog_ambient = (grey * 0.9, grey * 0.92, grey)

    def render_rain(self, fbo, depth, camera, view_proj, lighting, store, width: int,
                    height: int, time: float) -> None:  # fmt: skip
        if not self.raining(store):
            return
        from pyglm import glm

        program = self.passes("rain")
        sky = lighting.sky_ambient
        light = tuple(1.6 * c + 0.12 * s for c, s in zip(sky, lighting.sun_radiance, strict=True))
        wx, wy = wind_vector(store["rain.wind"], store["rain.wind_dir"])
        for name, value in {
            "u_depth": 0,
            "u_log_depth_coef": camera.log_depth_coef,
            "u_cam_forward": tuple(float(v) for v in camera.forward),
            "u_camera_height": float(lighting.camera_height),
            "u_time": float(time),
            "u_rain_intensity": float(store["rain.intensity"]),
            "u_rain_base": float(store["rain.cloud_base"]),
            "u_rain_wind": (wx, wy),
            "u_rain_light": light,
            "u_pixel_angle": float(math.radians(camera.fov_y)) / max(height, 1),
        }.items():
            if name in program:
                program[name] = value
        program["u_inv_view_proj"].write(glm.inverse(view_proj))
        previous = depth.compare_func
        depth.compare_func = ""
        depth.use(0)
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.passes.draw("rain")
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        depth.compare_func = previous

    def uniforms(self) -> dict[str, object]:
        """Layer uniforms for the cloud pass and the terrain (cloud shadows)."""
        shapes = [layer[0] for layer in self.layers] + [(0.0, 1.0, 0.0, 0.0)] * MAX_LAYERS
        noises = [layer[1] for layer in self.layers] + [(1000.0, 0.0, 0.0, 0.0)] * MAX_LAYERS
        colors = [layer[2] for layer in self.layers] + [(1.0, 1.0, 1.0, 1.0)] * MAX_LAYERS
        return {
            "u_layer_count": len(self.layers),
            "u_layer_shape": [tuple(map(float, v)) for v in shapes[:MAX_LAYERS]],
            "u_layer_noise": [tuple(map(float, v)) for v in noises[:MAX_LAYERS]],
            "u_layer_color": [tuple(map(float, v)) for v in colors[:MAX_LAYERS]],
            "u_weather_time": self.time,
            "u_cloud_shadow_strength": float(self.shadow_strength),
            "u_cloud_noise": CLOUD_NOISE_UNIT,
            "u_weather_map": WEATHER_MAP_UNIT,
        }

    def render_clouds(self, fbo, depth, camera, view_proj, uniforms: dict, width: int,
                      height: int, jitter: float = 0.0) -> None:  # fmt: skip
        """Composite the layers onto ``fbo`` (colour only; ``depth`` is sampled)."""
        if not self.layers:
            return
        from pyglm import glm

        program = self.passes("clouds")
        values = {
            **uniforms,
            **self.uniforms(),
            "u_depth": 0,
            "u_log_depth_coef": camera.log_depth_coef,
            "u_cam_forward": tuple(float(v) for v in camera.forward),
            "u_jitter": float(jitter),
            "u_pixel_angle": float(math.radians(camera.fov_y)) / max(height, 1),
        }
        for name, value in values.items():
            if name in program:
                program[name] = value
        program["u_inv_view_proj"].write(glm.inverse(view_proj))
        previous = depth.compare_func
        depth.compare_func = ""
        depth.use(0)
        self.bind()
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.SRC_ALPHA  # rgb + dst * transmittance
        self.passes.draw("clouds")
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        depth.compare_func = previous

    def release(self) -> None:
        for tex in (self.noise, self.map):
            if tex is not None:
                tex.release()


def layer_defaults() -> list[dict]:
    """Default look of the three layers: valley fog, mid clouds, high clouds."""
    return [
        {"enabled": True, "base": 700.0, "thickness": 600.0, "coverage": 0.5, "density": 0.35,
         "scale": 1500.0, "wind_speed": 2.0, "wind_dir": 60.0, "detail": 0.6, "brightness": 1.0},
        {"enabled": False, "base": 2600.0, "thickness": 900.0, "coverage": 0.4, "density": 0.6,
         "scale": 3000.0, "wind_speed": 8.0, "wind_dir": 70.0, "detail": 0.7, "brightness": 1.0},
        {"enabled": False, "base": 4500.0, "thickness": 1500.0, "coverage": 0.3, "density": 0.35,
         "scale": 6000.0, "wind_speed": 15.0, "wind_dir": 80.0, "detail": 0.5,
         "brightness": 1.1},
    ]  # fmt: skip


def weather_properties() -> list:
    from earthling.render.parameters import boolean, flt, section

    defs = [
        boolean("weather.enabled", "Weather", False,
                tooltip="Volumetric cloud and fog layers, rain and thunderstorms"),
        flt("weather.cloud_shadows", "Cloud shadows", 0.8, 0.0, 1.0),
    ]  # fmt: skip
    names = ["Low layer (fog)", "Middle layer", "High layer"]
    for i, (name, d) in enumerate(zip(names, layer_defaults(), strict=True), start=1):
        p = f"weather.layer{i}."
        defs += [
            boolean(p + "enabled", f"{name}", d["enabled"]),
            flt(p + "base", f"{name}: base", d["base"], -500.0, 9000.0, step=50.0, decimals=0,
                unit="m", tooltip="Metres above sea level"),
            flt(p + "thickness", f"{name}: thickness", d["thickness"], 10.0, 6000.0, step=50.0,
                decimals=0, unit="m"),
            flt(p + "coverage", f"{name}: coverage", d["coverage"], 0.0, 1.0),
            flt(p + "density", f"{name}: density", d["density"], 0.0, 3.0),
            flt(p + "scale", f"{name}: size of the patches", d["scale"], 100.0, 30000.0,
                step=100.0, decimals=0, unit="m", logarithmic=True),
            flt(p + "wind_speed", f"{name}: wind", d["wind_speed"], 0.0, 60.0, step=0.5,
                unit="m/s"),
            flt(p + "wind_dir", f"{name}: wind towards", d["wind_dir"], 0.0, 360.0, step=5.0,
                decimals=0, unit="°"),
            flt(p + "detail", f"{name}: fraying", d["detail"], 0.0, 1.0),
            flt(p + "brightness", f"{name}: brightness", d["brightness"], 0.0, 3.0),
        ]  # fmt: skip
    from earthling.render.lightning import lightning_properties

    return section("Weather: Clouds & Fog", *defs) + rain_properties() + lightning_properties()


def rain_properties() -> list:
    from earthling.render.parameters import boolean, flt, section

    return section(
        "Weather: Rain & Thunder",
        boolean("rain.enabled", "Rain", False),
        flt("rain.intensity", "Intensity", 0.7, 0.0, 1.0),
        flt("rain.cloud_base", "Rain clouds: base", 2200.0, 0.0, 8000.0, step=50.0, decimals=0,
            unit="m"),
        flt("rain.cloud_thickness", "Rain clouds: thickness", 3000.0, 100.0, 8000.0, step=50.0,
            decimals=0, unit="m"),
        flt("rain.coverage", "Rain clouds: coverage", 0.85, 0.0, 1.0),
        flt("rain.cloud_density", "Rain clouds: density", 1.0, 0.0, 3.0),
        flt("rain.darkness", "Rain clouds: darkness", 0.6, 0.0, 1.0),
        flt("rain.wind", "Wind", 3.0, 0.0, 30.0, step=0.5, unit="m/s",
            tooltip="Slants the rain and moves the rain clouds"),
        flt("rain.wind_dir", "Wind towards", 90.0, 0.0, 360.0, step=5.0, decimals=0, unit="°"),
        flt("rain.visibility_km", "Visibility", 6.0, 0.2, 50.0, step=0.1, decimals=1, unit="km",
            logarithmic=True),
        flt("rain.wetness", "Wet ground", 0.8, 0.0, 1.0),
    )  # fmt: skip
