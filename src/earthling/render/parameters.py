"""Declarations of all scene/rendering properties, grouped by UI section.

Each render feature adds its section here. Keeping them in one module keeps ids unique and the
order of the UI sections stable.
"""

from __future__ import annotations

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
    color("view.background", "Background", (0.55, 0.68, 0.82)),
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

LIGHT = section(
    "Lighting",
    flt("light.azimuth", "Light azimuth", 315.0, 0.0, 360.0, step=1.0, decimals=0, unit="°"),
    flt("light.elevation", "Light elevation", 40.0, -10.0, 90.0, step=0.5, decimals=1, unit="°"),
    flt("light.ambient", "Ambient", 0.55, 0.0, 2.0, uniform="u_ambient"),
    flt("light.relief", "Relief shading", 0.75, 0.0, 3.0, uniform="u_relief"),
)  # fmt: skip

TRACKS = section(
    "Tracks",
    boolean("tracks.visible", "Show tracks", True),
)


def build_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    for group in (VIEW, TERRAIN, LIGHT, TRACKS):
        registry.extend(group)
    return registry
