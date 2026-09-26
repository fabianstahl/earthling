"""Points of interest: positioned icons with captions and keyframeable effects (Qt independent).

A POI belongs to the scene (saved in the scene file). Its static data (position, icon, size,
caption) lives in :class:`Poi`; everything animatable is a scene property ``poi.<id>.*``
registered per POI (see :func:`poi_properties`), so POIs get keys, the dope sheet and the graph
editor like every other property.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Any

EFFECTS = [
    ("none", "None"),
    ("bounce", "Bounce"),
    ("pulse", "Pulse"),
    ("wobble", "Wobble"),
    ("spin", "Spin"),
    ("float", "Float"),
]


@dataclass
class Poi:
    id: str  # unique slug, used in property ids
    name: str
    lon: float
    lat: float
    icon: str = "builtin:pin"  # file path (absolute or relative to the scene) or builtin:<name>
    size_px: float = 64.0  # icon height at 1080p output
    caption: str = ""
    lift_px: float = 24.0  # icon floats this far above its ground point (pin line below)
    height_offset_m: float = 0.0  # anchor above the terrain
    sprite_cols: int = 1  # sprite sheet layout (PNG); GIF/APNG frames are read directly
    sprite_rows: int = 1
    fps: float = 12.0  # frame animation speed of sprite sheets (GIF/APNG use their timing)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Poi:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "poi"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def poi_properties(poi: Poi) -> list:
    """Keyframeable properties of one POI ("poi.<id>.*")."""
    from earthling.render.parameters import boolean, enum, flt, section

    p = f"poi.{poi.id}."
    return section(
        f"POI: {poi.name}",
        boolean(p + "visible", "Show", True,
                tooltip="Keys on this property drive the pop-in / pop-out animation"),
        flt(p + "opacity", "Opacity", 1.0, 0.0, 1.0),
        flt(p + "scale", "Scale", 1.0, 0.0, 5.0, step=0.05),
        boolean(p + "pop", "Pop in / out", True,
                tooltip="Grow in with a little overshoot when shown, shrink when hidden"),
        enum(p + "effect", "Effect", "none", EFFECTS),
        flt(p + "effect_strength", "Effect strength", 1.0, 0.0, 5.0),
        flt(p + "effect_speed", "Effect speed", 1.0, 0.0, 10.0, step=0.05, unit="Hz"),
        flt(p + "anim_speed", "Animation speed", 1.0, 0.0, 10.0, step=0.05,
            tooltip="Playback speed of animated icons (GIF, APNG, sprite sheets)"),
        boolean(p + "caption", "Show caption", True),
        boolean(p + "pin", "Pin line", True),
    )  # fmt: skip


def all_poi_properties(pois: list[Poi]) -> list:
    return [d for poi in pois for d in poi_properties(poi)]


# --- time helpers (deterministic effects from the keys of "visible") --------------------------
POP_IN_S = 0.45
POP_OUT_S = 0.3


def visibility_state(animation, poi_id: str, time: float, visible: bool) -> tuple[float, float]:
    """(scale factor from the pop animation, seconds since the POI appeared).

    The appearance time is the last key at or before ``time`` where "visible" switched on (or
    the start of the timeline); a switch-off shrinks the icon over POP_OUT_S."""
    curve = animation.curves.get(f"poi.{poi_id}.visible") if animation is not None else None
    since = time
    if curve is not None and curve.keys:
        on_at, off_at, previous = None, None, None
        for key in curve.keys:
            if key.time > time:
                break
            if key.value and previous is not True:
                on_at = key.time
            if not key.value and previous is True:  # a switch-off (not a hidden start)
                off_at = key.time
            previous = bool(key.value)
        if visible:
            since = time - on_at if on_at is not None else time
            return pop_in(since), since
        if off_at is not None and time - off_at < POP_OUT_S:
            since_on = off_at - on_at if on_at is not None else off_at
            return 1.0 - smooth01((time - off_at) / POP_OUT_S), since_on + (time - off_at)
        return 0.0, 0.0
    return (1.0 if visible else 0.0), since


def smooth01(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def pop_in(since: float) -> float:
    """0 -> 1 with a little overshoot ("ease out back") over POP_IN_S seconds."""
    if since <= 0.0:
        return 0.0
    if since >= POP_IN_S:
        return 1.0
    x = since / POP_IN_S - 1.0
    c1 = 1.70158
    return 1.0 + (c1 + 1.0) * x**3 + c1 * x**2
