import math

import numpy as np
import pytest

from earthling.core.geo import lonlat_to_tile
from earthling.core.scene import Scene
from earthling.render import lod
from earthling.render.camera import Camera, OrbitController
from earthling.render.renderer import Renderer
from earthling.render.weather import equalize, wind_vector
from test_terrain import FRAME, FakeTerrainData


def test_helpers():
    assert wind_vector(10.0, 90.0) == pytest.approx((10.0, 0.0), abs=1e-9)  # towards east
    assert wind_vector(10.0, 0.0) == pytest.approx((0.0, 10.0), abs=1e-9)
    values = np.array([5, 5, 5, 200, 7, 9], dtype=np.uint8)
    out = equalize(values)
    assert out.min() == 0 and out.max() == 255 and out[3] == 255


def test_noise_is_generated_and_equalised(gl_ctx):
    renderer = Renderer(gl_ctx)
    renderer.weather._ensure_noise()
    noise = np.frombuffer(renderer.weather.noise.read(), dtype=np.uint8).reshape(-1, 4) / 255.0
    base = noise[:, 0]
    assert base.mean() == pytest.approx(0.5, abs=0.01)
    assert (base > 0.7).mean() == pytest.approx(0.3, abs=0.02)  # "coverage" = cloudy fraction
    assert noise[:, 1].std() > 0.1  # detail channel present
    cover = np.frombuffer(renderer.weather.map.read(), dtype=np.uint8) / 255.0
    assert cover.mean() == pytest.approx(0.5, abs=0.01)


def test_layers_follow_the_properties(gl_ctx):
    renderer = Renderer(gl_ctx)
    store = Scene().store
    weather = renderer.weather
    weather.update(store, 10.0)
    assert weather.layers == []  # off by default
    store.set("weather.enabled", True)
    weather.update(store, 10.0)
    assert len(weather.layers) == 1  # the fog layer is on by default
    store.set("weather.layer2.enabled", True)
    store.set("weather.layer2.wind_speed", 5.0)
    store.set("weather.layer2.wind_dir", 90.0)
    weather.update(store, 10.0)
    shape, noise, _ = weather.layers[1]
    assert shape[0] == store["weather.layer2.base"]
    assert noise[1] == pytest.approx(50.0) and noise[2] == pytest.approx(0.0, abs=1e-9)
    uniforms = weather.uniforms()
    assert uniforms["u_layer_count"] == 2 and len(uniforms["u_layer_shape"]) == 4


def render(gl_ctx, distance=1.0, **props):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    scene = Scene()
    for pid, value in props.items():
        scene.store.set(pid.replace("__", "."), value)
    renderer.store = scene.store
    renderer.timezone = "UTC"
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 96)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.distance *= distance
    orbit.pitch = -35.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 96)
    fbo = gl_ctx.simple_framebuffer((96, 96))
    renderer.render(fbo, 96, 96, camera)
    return np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(96, 96, 3).astype(float)


def test_cloud_layer_renders_and_casts_shadows(gl_ctx):
    base = {"sun__datetime": __import__("datetime").datetime(2026, 7, 1, 12, 0),
            "haze__aerial": 0.0}  # fmt: skip
    clear = render(gl_ctx, **base)
    # a thick layer between the camera and the ground (the terrain is at 1500-2500 m)
    cloudy = render(gl_ctx, **base, weather__enabled=True, weather__layer1__base=2700.0,
                    weather__layer1__thickness=2000.0, weather__layer1__coverage=0.8,
                    weather__layer1__density=1.0)  # fmt: skip
    assert np.abs(cloudy - clear).mean() > 10.0
    none = render(gl_ctx, **base, weather__enabled=True, weather__layer1__coverage=0.0,
                  weather__cloud_shadows=0.0)  # fmt: skip
    assert np.abs(none - clear).max() <= 2.0  # nothing to draw
    # shadows only: the camera (~7 km up, looking down) is below the layer, which stays out
    # of view but darkens the sunlit terrain
    high = {**base, "weather__enabled": True, "weather__layer1__base": 12000.0,
            "weather__layer1__thickness": 3000.0, "weather__layer1__coverage": 1.0,
            "weather__layer1__density": 1.0}  # fmt: skip
    shaded = render(gl_ctx, 0.2, **high, weather__cloud_shadows=1.0)
    unshaded = render(gl_ctx, 0.2, **high, weather__cloud_shadows=0.0)
    assert shaded.mean() < unshaded.mean() - 3.0
    assert math.isfinite(shaded.mean())


def test_rain_dims_the_light_and_wets_the_ground(gl_ctx):
    from earthling.render.lighting import compute_lighting

    renderer = Renderer(gl_ctx)
    store = Scene().store
    weather = renderer.weather
    assert weather.rain_uniforms(store) == {"u_wetness": 0.0, "u_rain_haze": 0.0}
    store.set("rain.enabled", True)
    store.set("rain.intensity", 1.0)
    store.set("rain.visibility_km", 3.0)
    u = weather.rain_uniforms(store)
    assert u["u_wetness"] == pytest.approx(store["rain.wetness"])
    assert u["u_rain_haze"] == pytest.approx(1.0 / 1000.0)
    weather.update(store, 5.0)
    assert len(weather.layers) == 1  # only the rain deck (weather layers are off)
    assert weather.layers[0][0][0] == store["rain.cloud_base"]
    lighting = compute_lighting(store, 46.0, 7.0, "UTC")
    before = lighting.sky_ambient
    weather.dim_lighting(store, lighting)
    assert all(a < b for a, b in zip(lighting.sky_ambient, before, strict=True))


def test_rain_renders_streaks(gl_ctx):
    import datetime

    base = {"sun__datetime": datetime.datetime(2026, 7, 1, 12, 0), "haze__aerial": 0.0}
    dry = render(gl_ctx, 0.08, **base)
    # rain clouds far above the (low) camera, no deck in view: only streaks, haze, wet ground
    wet = render(gl_ctx, 0.08, **base, rain__enabled=True, rain__intensity=1.0,
                 rain__cloud_base=40000.0, rain__coverage=0.0)  # fmt: skip
    assert np.abs(wet - dry).mean() > 2.0
    streaks_only = render(gl_ctx, 0.08, **base, rain__enabled=True, rain__intensity=1.0,
                          rain__cloud_base=40000.0, rain__coverage=0.0, rain__wetness=0.0,
                          rain__visibility_km=50.0)  # fmt: skip
    # thin vertical streaks: high horizontal contrast compared with the dry image
    dx = lambda img: np.abs(np.diff(img.mean(axis=2), axis=1)).mean()  # noqa: E731
    assert dx(streaks_only) > dx(dry)


def test_layers_above_shade_the_clouds_below(gl_ctx):
    base = {"sun__datetime": __import__("datetime").datetime(2026, 7, 1, 12, 0),
            "haze__aerial": 0.0, "weather__enabled": True, "weather__cloud_shadows": 0.0,
            "weather__layer1__base": 2700.0, "weather__layer1__thickness": 2000.0,
            "weather__layer1__coverage": 0.8, "weather__layer1__density": 1.0}  # fmt: skip
    # a dense deck above the camera (out of view, and no ground shadows)
    deck = {"weather__layer3__enabled": True, "weather__layer3__base": 12000.0,
            "weather__layer3__thickness": 3000.0, "weather__layer3__coverage": 1.0,
            "weather__layer3__density": 1.0}  # fmt: skip
    lit = render(gl_ctx, **base)
    shaded = render(gl_ctx, **base, **deck)
    assert shaded.mean() < lit.mean() - 5.0


def test_clouds_fade_into_the_rain_haze(gl_ctx):
    import datetime

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    scene = Scene()
    for pid, value in {"sun.datetime": datetime.datetime(2026, 7, 1, 12, 0), "rain.enabled": True,
                       "rain.visibility_km": 2.0}.items():  # fmt: skip
        scene.store.set(pid, value)
    renderer.store = scene.store
    renderer.timezone = "UTC"
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.apply()
    renderer.render(gl_ctx.simple_framebuffer((64, 64)), 64, 64, camera)
    program = renderer.weather.passes("clouds")
    assert program["u_rain_haze"].value == pytest.approx(3.0 / 2000.0 * 0.7, rel=1e-3)
    sky = renderer.fullscreen("sky")  # the sky vanishes in the haze too (no dark holes)
    assert sky["u_rain_haze"].value == pytest.approx(3.0 / 2000.0 * 0.7, rel=1e-3)


def test_cloud_shadows_are_continuous_at_sunrise(gl_ctx):
    """No cut-off at low sun: images just below and above ~0.6 degrees differ only slightly."""
    import datetime

    from earthling.render.lighting import solar_position

    when = datetime.datetime(2026, 7, 1, 5, 30)
    elevation = solar_position(when.replace(tzinfo=datetime.UTC), 46.0, 7.0).elevation
    base = {"sun__datetime": when, "haze__aerial": 0.0, "sun__intensity": 6.0,
            "weather__enabled": True, "weather__cloud_shadows": 1.0,
            "weather__layer1__base": 2700.0, "weather__layer1__thickness": 2000.0,
            "weather__layer1__coverage": 0.8, "weather__layer1__density": 1.0,
            "weather__layer3__enabled": True, "weather__layer3__base": 12000.0,
            "weather__layer3__thickness": 3000.0, "weather__layer3__coverage": 1.0,
            "weather__layer3__density": 1.0}  # fmt: skip
    below = render(gl_ctx, **base, sun__elevation_offset=0.45 - elevation)
    above = render(gl_ctx, **base, sun__elevation_offset=0.75 - elevation)
    assert np.abs(above - below).mean() < 10.0  # was ~38 with the cut-off
