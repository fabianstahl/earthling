"""Scene lighting derived from the property store (sun position, colors, atmosphere)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from earthling.core.easing import smoothstep
from earthling.core.properties import PropertyStore
from earthling.core.sun import SunPosition, local_to_aware, solar_position
from earthling.render.atmosphere import sun_transmittance


def srgb_to_linear(c) -> tuple[float, float, float]:
    return tuple(float(v) ** 2.2 for v in c)  # type: ignore[return-value]


@dataclass
class Lighting:
    sun: SunPosition  # after artistic offsets
    sun_direction: tuple[float, float, float]  # ENU, towards the sun
    sun_radiance: tuple[float, float, float]  # linear RGB at the camera, 0 below horizon
    sky_ambient: tuple[float, float, float]
    ground_ambient: tuple[float, float, float]
    night: float  # 0 = day, 1 = full night
    when: datetime  # aware
    camera_height: float = 1500.0
    fog_ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fog_sun: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fog_density: float = 0.0  # per metre
    # world ENU -> ENU at the camera (planet "up" = +z for the atmosphere), column-major mat3
    atm_frame: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # the parts of the ambient light by daylight (sky_day, ground_day) and at night, the sun's
    # colour before the atmosphere and the camera's daylight factor: views from far up light
    # every point by its own sun elevation (day and night side, terminator; terrain.frag)
    sun_base: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sky_day: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sky_night: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_day: tuple[float, float, float] = (0.0, 0.0, 0.0)  # with the sun fully up
    ground_night: tuple[float, float, float] = (0.0, 0.0, 0.0)
    day: float = 1.0


def compute_lighting(
    store: PropertyStore,
    lat: float,
    lon: float,
    timezone: str,
    camera_height: float = 1500.0,
    when: datetime | None = None,
) -> Lighting:
    """``when`` (aware) replaces the ``sun.datetime`` property (e.g. the GPX time)."""
    if when is None:
        when = local_to_aware(store["sun.datetime"], timezone)
    real = solar_position(when, lat, lon)
    sun = SunPosition(
        (real.azimuth + store["sun.azimuth_offset"]) % 360.0,
        max(-90.0, min(90.0, real.elevation + store["sun.elevation_offset"])),
    )
    direction = sun.direction_enu()
    transmittance = sun_transmittance(
        sun.elevation, camera_height, store["sky.rayleigh"], store["sky.haze"]
    )
    # fade the direct light around the horizon (the disc sets over ~1 degree)
    visibility = smoothstep(-1.0, 2.0, sun.elevation)
    base = np.array(srgb_to_linear(store["sun.color"])) * store["sun.intensity"]
    radiance = base * transmittance * visibility
    # ambient: daylight falls off through civil twilight down to a dim night level; the sky
    # light turns warmer and weaker when the sun light has to cross more atmosphere
    day = smoothstep(-8.0, 8.0, sun.elevation)
    night = 1.0 - smoothstep(-16.0, -4.0, sun.elevation)
    # (the tint of a sun below the horizon is that of the sun on it: the transmittance there
    # underflows to equal tiny values, which turned the tint white and made it jump to red at
    # sunrise - seen from space the whole day side changed colour in one frame)
    tint_t = transmittance
    if sun.elevation < 0.5:
        tint_t = sun_transmittance(0.5, camera_height, store["sky.rayleigh"], store["sky.haze"])
    tint = 0.35 + 0.65 * tint_t / max(float(tint_t.max()), 1e-6)
    night_sky = np.array([0.012, 0.016, 0.03])
    sky_day = np.array(srgb_to_linear(store["light.sky_color"])) * store["light.ambient"] * tint
    sky_night = night_sky * store["light.night_ambient"]
    sky = sky_day * day + sky_night * (1.0 - day)
    ground_full = np.array(srgb_to_linear(store["light.ground_color"])) * store["light.ambient"]
    ground_day = ground_full * (0.3 + 0.7 * float(visibility))
    ground_night = night_sky * 0.3
    ground = ground_day * day + ground_night * (1.0 - day)
    fog_color = np.array(srgb_to_linear(store["fog.color"]))
    fog_ambient = fog_color * (sky * 0.8 + ground * 0.2)
    fog_sun = fog_color * radiance * store["fog.sun_scatter"] * 0.25

    def vec(a) -> tuple[float, float, float]:
        return tuple(float(v) for v in a)  # type: ignore[return-value]

    return Lighting(
        sun,
        direction,
        vec(radiance),
        vec(sky),
        vec(ground),
        float(night),
        when,
        camera_height,
        vec(fog_ambient),
        vec(fog_sun),
        store["fog.density"] / 1000.0,
        sun_base=vec(base),
        sky_day=vec(sky_day),
        sky_night=vec(sky_night),
        ground_day=vec(ground_full),  # (the sun's visibility: per point, see terrain.frag)
        ground_night=vec(ground_night),
        day=float(day),
    )


def lighting_uniforms(lighting: Lighting) -> dict[str, object]:
    return {
        "u_sun_dir": lighting.sun_direction,
        "u_sun_radiance": lighting.sun_radiance,
        "u_sky_ambient": lighting.sky_ambient,
        "u_ground_ambient": lighting.ground_ambient,
        "u_camera_height": lighting.camera_height,
        "u_fog_ambient": lighting.fog_ambient,
        "u_fog_sun": lighting.fog_sun,
        "u_fog_density": lighting.fog_density,
        "u_atm_frame": lighting.atm_frame,
        "u_sun_base": lighting.sun_base,
        "u_sky_day": lighting.sky_day,
        "u_sky_night": lighting.sky_night,
        "u_ground_day": lighting.ground_day,
        "u_ground_night": lighting.ground_night,
        "u_cam_day": lighting.day,
        # views from far up: every point lit by its own sun elevation (0 near the ground,
        # where the camera's light is the same everywhere)
        "u_local_light": smoothstep(20e3, 200e3, lighting.camera_height),
    }
