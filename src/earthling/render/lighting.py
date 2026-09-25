"""Scene lighting derived from the property store (sun position, colors, atmosphere)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from earthling.core.properties import PropertyStore
from earthling.core.sun import SunPosition, local_to_aware, solar_position
from earthling.render.atmosphere import sun_transmittance


def srgb_to_linear(c) -> tuple[float, float, float]:
    return tuple(float(v) ** 2.2 for v in c)  # type: ignore[return-value]


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def greenwich_sidereal_angle(when: datetime) -> float:
    """Greenwich mean sidereal time as an angle in radians."""
    jd = when.astimezone(UTC).timestamp() / 86400.0 + 2440587.5
    d = jd - 2451545.0
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return math.radians(gmst_deg)


def enu_to_celestial(when: datetime, lat: float, lon: float) -> np.ndarray:
    """3x3 matrix mapping ENU directions to the (Earth-centred) equatorial frame."""
    phi, lam = math.radians(lat), math.radians(lon)
    east = np.array([-math.sin(lam), math.cos(lam), 0.0])
    north = np.array(
        [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)]
    )
    up = np.array([math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)])
    enu_to_ecef = np.column_stack([east, north, up])
    g = greenwich_sidereal_angle(when)
    rot = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return rot @ enu_to_ecef


@dataclass
class Lighting:
    sun: SunPosition  # after artistic offsets
    sun_direction: tuple[float, float, float]  # ENU, towards the sun
    sun_radiance: tuple[float, float, float]  # linear RGB at the camera, 0 below horizon
    sky_ambient: tuple[float, float, float]
    ground_ambient: tuple[float, float, float]
    night: float  # 0 = day, 1 = full night
    when: datetime  # aware


def compute_lighting(
    store: PropertyStore, lat: float, lon: float, timezone: str, camera_height: float = 1500.0
) -> Lighting:
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
    tint = 0.35 + 0.65 * transmittance / max(float(transmittance.max()), 1e-6)
    night_sky = np.array([0.012, 0.016, 0.03])
    sky = np.array(srgb_to_linear(store["light.sky_color"])) * store[
        "light.ambient"
    ] * day * tint + night_sky * store["light.night_ambient"] * (1.0 - day)
    ground = np.array(srgb_to_linear(store["light.ground_color"])) * store["light.ambient"] * day
    ground = ground * (0.3 + 0.7 * float(visibility)) + night_sky * 0.3 * (1.0 - day)
    return Lighting(
        sun,
        direction,
        tuple(float(v) for v in radiance),  # type: ignore[arg-type]
        tuple(float(v) for v in sky),  # type: ignore[arg-type]
        tuple(float(v) for v in ground),  # type: ignore[arg-type]
        float(night),
        when,
    )


def lighting_uniforms(lighting: Lighting) -> dict[str, object]:
    return {
        "u_sun_dir": lighting.sun_direction,
        "u_sun_radiance": lighting.sun_radiance,
        "u_sky_ambient": lighting.sky_ambient,
        "u_ground_ambient": lighting.ground_ambient,
    }
