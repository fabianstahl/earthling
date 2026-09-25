"""Declarations of all scene/rendering properties, grouped by UI section.

Each render feature adds its section here. Keeping them in one module keeps ids unique and the
order of the UI sections stable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from earthling.core.properties import PropertyDef, PropertyRegistry, PType


def flt(pid, label, default, lo, hi, step=0.01, decimals=2, unit="", **kw) -> PropertyDef:
    return PropertyDef(pid, label, PType.FLOAT, default, minimum=lo, maximum=hi, step=step,
                       decimals=decimals, unit=unit, **kw)  # fmt: skip


def integer(pid, label, default, lo, hi, step=1, unit="", **kw) -> PropertyDef:
    return PropertyDef(pid, label, PType.INT, default, minimum=lo, maximum=hi, step=step,
                       unit=unit, **kw)  # fmt: skip


def boolean(pid, label, default, **kw) -> PropertyDef:
    return PropertyDef(pid, label, PType.BOOL, default, **kw)


def enum(pid, label, default, options: list[tuple[str, str]], **kw) -> PropertyDef:
    return PropertyDef(pid, label, PType.ENUM, default, options=tuple(options), **kw)


def color(pid, label, default: tuple[float, float, float], **kw) -> PropertyDef:
    return PropertyDef(pid, label, PType.COLOR, default, **kw)


def section(title: str, *defs: PropertyDef) -> list[PropertyDef]:
    """Assign the UI section title to a group of definitions."""
    out = []
    for d in defs:
        fields: dict[str, Any] = {**d.__dict__, "section": title}
        out.append(PropertyDef(**fields))
    return out


VIEW = section(
    "View",
    flt("view.fov", "Field of view", 50.0, 10.0, 120.0, step=0.5, decimals=1, unit="°"),
    boolean("view.show_outlines", "Area outlines", True, animatable=False,
            tooltip="Show the area of interest and resolution zone outlines"),
)  # fmt: skip

TERRAIN = section(
    "Terrain",
    flt("terrain.exaggeration", "Vertical exaggeration", 1.0, 0.1, 5.0, step=0.05, unit="×",
        uniform="u_exaggeration"),
    boolean("terrain.imagery", "Show imagery", True, uniform="u_show_imagery"),
    flt("terrain.detail", "Detail threshold", 1.0, 0.25, 8.0, step=0.05, unit="px/texel",
        logarithmic=True, animatable=False,
        tooltip="Refine terrain until one imagery texel covers at most this many screen pixels"),
    integer("terrain.memory_budget_mb", "GPU memory budget", 3000, 256, 16000, step=128,
            unit="MB", animatable=False),
    boolean("terrain.debug_lod", "Debug: color LOD levels", False, uniform="u_debug_lod",
            animatable=False),
)  # fmt: skip

SUN = section(
    "Sun & Time",
    PropertyDef("sun.datetime", "Local date & time", PType.DATETIME, datetime(2026, 7, 1, 12, 0),
                tooltip="Local time in the project's timezone; drives the sun position"),
    flt("sun.azimuth_offset", "Azimuth offset", 0.0, -180.0, 180.0, step=1.0, decimals=1,
        unit="°", tooltip="Artistic rotation of the computed sun position"),
    flt("sun.elevation_offset", "Elevation offset", 0.0, -45.0, 45.0, step=0.5, decimals=1,
        unit="°"),
    flt("sun.intensity", "Sun intensity", 1.4, 0.0, 6.0),
    color("sun.color", "Sun color", (1.0, 0.96, 0.9)),
)  # fmt: skip

LIGHT = section(
    "Lighting",
    flt("light.ambient", "Sky ambient", 0.45, 0.0, 3.0),
    flt("light.night_ambient", "Night ambient", 1.0, 0.0, 10.0,
        tooltip="Brightness of moon/starlight on the terrain at night"),
    color("light.sky_color", "Sky ambient color", (0.62, 0.72, 0.92)),
    color("light.ground_color", "Ground bounce color", (0.45, 0.4, 0.33)),
)  # fmt: skip

SKY = section(
    "Sky & Atmosphere",
    flt("sky.illuminance", "Sky brightness", 20.0, 0.0, 100.0, step=0.5, decimals=1,
        uniform="u_sun_illuminance"),
    flt("sky.rayleigh", "Air density", 1.0, 0.0, 4.0, uniform="u_rayleigh_scale",
        tooltip="Rayleigh scattering: blue sky, red sunsets"),
    flt("sky.haze", "Haze", 1.0, 0.0, 10.0, uniform="u_mie_scale",
        tooltip="Mie scattering by aerosols: whitish haze and glow around the sun"),
    flt("sky.sun_disc", "Sun disc", 1.0, 0.0, 5.0, uniform="u_sun_disc"),
    flt("sky.stars", "Star brightness", 1.0, 0.0, 10.0, uniform="u_star_brightness"),
)  # fmt: skip

FOG = section(
    "Fog & Haze",
    flt("haze.aerial", "Aerial perspective", 1.0, 0.0, 4.0, uniform="u_aerial_strength",
        tooltip="Physically based haze between camera and terrain (uses the sky model)"),
    boolean("fog.enabled", "Height fog", False, uniform="u_fog_enabled"),
    flt("fog.density", "Fog density", 0.3, 0.0, 20.0, step=0.01, unit="/km",
        logarithmic=False, tooltip="Extinction per kilometre at the fog base height"),
    flt("fog.base", "Fog base height", 1200.0, -500.0, 6000.0, step=10.0, decimals=0, unit="m",
        uniform="u_fog_base"),
    flt("fog.falloff", "Fog falloff", 250.0, 10.0, 5000.0, step=10.0, decimals=0, unit="m",
        logarithmic=True, uniform="u_fog_falloff",
        tooltip="Height over which the fog density drops to 37 %"),
    color("fog.color", "Fog color", (0.85, 0.88, 0.92)),
    flt("fog.sun_scatter", "Sun glow", 1.0, 0.0, 5.0,
        tooltip="How strongly sunlight scatters in the fog (glow towards the sun)"),
    flt("fog.anisotropy", "Glow sharpness", 0.6, 0.0, 0.95, uniform="u_fog_glow"),
)  # fmt: skip

POST = section(
    "Camera & Tonemapping",
    flt("post.exposure", "Exposure", 0.0, -8.0, 8.0, step=0.1, decimals=1, unit="EV"),
    enum("post.tonemap", "Tonemapping", "aces",
         [("aces", "ACES filmic"), ("reinhard", "Reinhard"), ("none", "None (clip)")],
         uniform="u_tonemap"),
)  # fmt: skip

TRACKS = section(
    "Tracks",
    boolean("tracks.visible", "Show tracks", True),
)


def build_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    for group in (VIEW, POST, TERRAIN, SUN, LIGHT, SKY, FOG, TRACKS):
        registry.extend(group)
    return registry
