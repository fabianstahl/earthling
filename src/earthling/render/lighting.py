"""Scene lighting derived from the property store (sun position, colors)."""

from __future__ import annotations

from dataclasses import dataclass

from earthling.core.properties import PropertyStore
from earthling.core.sun import SunPosition, local_to_aware, solar_position


def srgb_to_linear(c: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(v**2.2 for v in c)  # type: ignore[return-value]


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Lighting:
    sun: SunPosition  # after artistic offsets
    sun_direction: tuple[float, float, float]  # ENU, towards the sun
    sun_radiance: tuple[float, float, float]  # linear RGB, 0 when below the horizon
    sky_ambient: tuple[float, float, float]
    ground_ambient: tuple[float, float, float]


def compute_lighting(store: PropertyStore, lat: float, lon: float, timezone: str) -> Lighting:
    when = local_to_aware(store["sun.datetime"], timezone)
    real = solar_position(when, lat, lon)
    sun = SunPosition(
        (real.azimuth + store["sun.azimuth_offset"]) % 360.0,
        max(-90.0, min(90.0, real.elevation + store["sun.elevation_offset"])),
    )
    direction = sun.direction_enu()
    # fade the direct light around the horizon (the disc sets over ~1°, penumbra of terrain)
    visibility = smoothstep(-1.0, 3.0, sun.elevation)
    radiance = tuple(
        c * store["sun.intensity"] * visibility for c in srgb_to_linear(store["sun.color"])
    )
    # ambient: daylight falls off through civil twilight down to a dim night level
    day = smoothstep(-8.0, 8.0, sun.elevation)
    night_level = 0.03
    ambient = store["light.ambient"] * (night_level + (1.0 - night_level) * day)
    sky = tuple(c * ambient for c in srgb_to_linear(store["light.sky_color"]))
    ground = tuple(c * ambient * max(0.2, day) for c in srgb_to_linear(store["light.ground_color"]))
    return Lighting(sun, direction, radiance, sky, ground)  # type: ignore[arg-type]


def lighting_uniforms(lighting: Lighting) -> dict[str, object]:
    return {
        "u_sun_dir": lighting.sun_direction,
        "u_sun_radiance": lighting.sun_radiance,
        "u_sky_ambient": lighting.sky_ambient,
        "u_ground_ambient": lighting.ground_ambient,
    }
