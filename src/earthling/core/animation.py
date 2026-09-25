"""Keyframe animation of properties (Qt independent).

Every animated property has a :class:`Curve` of :class:`Keyframe` objects. The interpolation
stored on a key applies to the segment *starting* at that key. Bezier handles are normalised
to the segment: ``out_handle = (x, y)`` places the first control point at
``(t0 + x * duration, v0 + y * delta)``, ``in_handle = (x, y)`` the second one at
``(t1 - x * duration, v1 - y * delta)``. That works for any value type (scalars, colours,
vectors, datetimes) and matches CSS ``cubic-bezier`` for the presets.

Type rules
----------
* float / int: interpolated (ints are rounded)
* color: interpolated in OKLab (perceptually even blends)
* vec3: per component
* datetime: as continuous time
* bool / enum: always step (hold until the next key)
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any

from earthling.core.properties import PropertyDef, PropertyStore, PType


class Interp(StrEnum):
    STEP = "step"
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BEZIER = "bezier"


# CSS cubic-bezier control points (x1, y1, x2, y2) of the presets
PRESETS: dict[Interp, tuple[float, float, float, float]] = {
    Interp.LINEAR: (0.0, 0.0, 1.0, 1.0),
    Interp.EASE_IN: (0.42, 0.0, 1.0, 1.0),
    Interp.EASE_OUT: (0.0, 0.0, 0.58, 1.0),
    Interp.EASE_IN_OUT: (0.42, 0.0, 0.58, 1.0),
}

INTERP_LABELS = {
    Interp.STEP: "Constant (step)",
    Interp.LINEAR: "Linear",
    Interp.EASE_IN: "Ease in",
    Interp.EASE_OUT: "Ease out",
    Interp.EASE_IN_OUT: "Ease in-out",
    Interp.BEZIER: "Bézier (custom handles)",
}

DEFAULT_HANDLE = (1.0 / 3.0, 0.0)  # smooth ease-in-out-like default for new Bezier keys


def cubic_bezier_ease(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """y(x) of the cubic Bezier (0,0),(x1,y1),(x2,y2),(1,1). x1, x2 are clamped to [0, 1]."""
    x1 = min(max(x1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def bx(u: float) -> float:
        return 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u**2 * x2 + u**3

    def by(u: float) -> float:
        return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u**2 * y2 + u**3

    def dbx(u: float) -> float:
        return 3 * (1 - u) ** 2 * x1 + 6 * (1 - u) * u * (x2 - x1) + 3 * u**2 * (1 - x2)

    # Newton iterations with bisection fallback (x(u) is monotonic for x1, x2 in [0, 1])
    u = x
    for _ in range(8):
        err = bx(u) - x
        if abs(err) < 1e-7:
            return by(u)
        d = dbx(u)
        if abs(d) < 1e-6:
            break
        u = min(max(u - err / d, 0.0), 1.0)
    lo, hi = 0.0, 1.0
    u = x
    for _ in range(40):
        if bx(u) < x:
            lo = u
        else:
            hi = u
        u = (lo + hi) / 2
    return by(u)


# --- colour spaces (OKLab, Björn Ottosson) ----------------------------------------------
def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = max(c, 0.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def srgb_to_oklab(rgb) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(float(c)) for c in rgb)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab_to_srgb(lab) -> tuple[float, float, float]:
    lightness, a, b = lab
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    lc, mc, sc = l_**3, m_**3, s_**3
    r = 4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc
    g = -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc
    bl = -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc
    return tuple(min(1.0, max(0.0, _linear_to_srgb(c))) for c in (r, g, bl))  # type: ignore[return-value]


# --- keyframes ---------------------------------------------------------------------------
@dataclass
class Keyframe:
    time: float
    value: Any
    interp: Interp = Interp.LINEAR
    out_handle: tuple[float, float] = DEFAULT_HANDLE  # used when interp is BEZIER
    in_handle: tuple[float, float] = DEFAULT_HANDLE  # of the segment *ending* at this key

    def copy(self) -> Keyframe:
        return Keyframe(self.time, self.value, self.interp, self.out_handle, self.in_handle)


def _is_stepped(ptype: PType) -> bool:
    return ptype in (PType.BOOL, PType.ENUM)


def _blend(definition: PropertyDef, a: Any, b: Any, f: float) -> Any:
    """Type-aware interpolation between two values (f may exceed [0, 1] for overshoot)."""
    t = definition.type
    if t is PType.FLOAT:
        return a + (b - a) * f
    if t is PType.INT:
        return int(round(a + (b - a) * f))
    if t is PType.VEC3:
        return tuple(x + (y - x) * f for x, y in zip(a, b, strict=True))
    if t is PType.COLOR:
        la, lb = srgb_to_oklab(a), srgb_to_oklab(b)
        return oklab_to_srgb(tuple(x + (y - x) * f for x, y in zip(la, lb, strict=True)))
    if t is PType.DATETIME:
        return a + timedelta(seconds=(b - a).total_seconds() * f)
    return a if f < 1.0 else b


@dataclass
class Curve:
    definition: PropertyDef
    keys: list[Keyframe] = field(default_factory=list)

    @property
    def pid(self) -> str:
        return self.definition.id

    def times(self) -> list[float]:
        return [k.time for k in self.keys]

    def key_at(self, time: float, tolerance: float = 1e-6) -> Keyframe | None:
        for k in self.keys:
            if abs(k.time - time) <= tolerance:
                return k
        return None

    def set_key(self, time: float, value: Any, interp: Interp | None = None) -> Keyframe:
        """Insert or replace the key at ``time`` (value coerced to the property type)."""
        value = self.definition.coerce(value)
        existing = self.key_at(time)
        if existing is not None:
            existing.value = value
            if interp is not None:
                existing.interp = interp
            return existing
        if interp is None:
            interp = Interp.STEP if _is_stepped(self.definition.type) else Interp.LINEAR
        key = Keyframe(time, value, interp)
        bisect.insort(self.keys, key, key=lambda k: k.time)
        return key

    def remove_key(self, time: float) -> bool:
        key = self.key_at(time)
        if key is None:
            return False
        self.keys.remove(key)
        return True

    def sort(self) -> None:
        self.keys.sort(key=lambda k: k.time)

    def evaluate(self, time: float) -> Any:
        keys = self.keys
        if not keys:
            raise ValueError(f"curve {self.pid} has no keys")
        if time <= keys[0].time:
            return keys[0].value
        if time >= keys[-1].time:
            return keys[-1].value
        i = bisect.bisect_right([k.time for k in keys], time) - 1
        k0, k1 = keys[i], keys[i + 1]
        duration = k1.time - k0.time
        x = (time - k0.time) / duration if duration > 0 else 1.0
        if k0.interp is Interp.STEP or _is_stepped(self.definition.type):
            return k0.value
        if k0.interp is Interp.BEZIER:
            (ox, oy), (ix, iy) = k0.out_handle, k1.in_handle
            f = cubic_bezier_ease(x, ox, oy, 1.0 - ix, 1.0 - iy)
        else:
            f = cubic_bezier_ease(x, *PRESETS[k0.interp])
        return self.definition.coerce(_blend(self.definition, k0.value, k1.value, f))

    # --- serialisation ----------------------------------------------------------------
    def to_json(self) -> list[dict[str, Any]]:
        out = []
        for k in self.keys:
            entry: dict[str, Any] = {
                "t": k.time,
                "v": self.definition.to_json(k.value),
                "interp": str(k.interp),
            }
            if k.interp is Interp.BEZIER or k.in_handle != DEFAULT_HANDLE:
                entry["out"] = list(k.out_handle)
                entry["in"] = list(k.in_handle)
            out.append(entry)
        return out

    @classmethod
    def from_json(cls, definition: PropertyDef, data: list[dict[str, Any]]) -> Curve:
        curve = cls(definition)
        for entry in data:
            key = Keyframe(
                float(entry["t"]),
                definition.from_json(entry["v"]),
                Interp(entry.get("interp", "linear")),
                tuple(entry.get("out", DEFAULT_HANDLE)),  # type: ignore[arg-type]
                tuple(entry.get("in", DEFAULT_HANDLE)),  # type: ignore[arg-type]
            )
            curve.keys.append(key)
        curve.sort()
        return curve


def snap_to_frame(time: float, fps: float) -> float:
    return round(time * fps) / fps


class Animation:
    """All curves of a scene plus the timeline settings."""

    def __init__(self, store: PropertyStore, duration: float = 30.0, fps: float = 60.0) -> None:
        self.store = store
        self.duration = duration
        self.fps = fps
        self.curves: dict[str, Curve] = {}

    # --- editing ----------------------------------------------------------------------
    def curve(self, pid: str, create: bool = False) -> Curve | None:
        curve = self.curves.get(pid)
        if curve is None and create:
            definition = self.store.registry[pid]
            if not definition.animatable:
                raise ValueError(f"property {pid} cannot be animated")
            curve = Curve(definition)
            self.curves[pid] = curve
        return curve

    def set_key(self, pid: str, time: float, value: Any = None, interp: Interp | None = None):
        """Key the property at ``time`` (default: its current value in the store)."""
        time = snap_to_frame(time, self.fps)
        value = self.store[pid] if value is None else value
        return self.curve(pid, create=True).set_key(time, value, interp)  # type: ignore[union-attr]

    def remove_key(self, pid: str, time: float) -> bool:
        curve = self.curves.get(pid)
        if curve is None:
            return False
        removed = curve.remove_key(snap_to_frame(time, self.fps))
        if not curve.keys:
            del self.curves[pid]
        return removed

    def is_animated(self, pid: str) -> bool:
        return pid in self.curves and bool(self.curves[pid].keys)

    def has_key(self, pid: str, time: float) -> bool:
        curve = self.curves.get(pid)
        return curve is not None and curve.key_at(snap_to_frame(time, self.fps)) is not None

    def key_times(self) -> list[float]:
        return sorted({k.time for c in self.curves.values() for k in c.keys})

    # --- evaluation -------------------------------------------------------------------
    def values_at(self, time: float) -> dict[str, Any]:
        return {pid: c.evaluate(time) for pid, c in self.curves.items() if c.keys}

    def apply(self, time: float, notify: bool = True) -> None:
        """Write all animated values at ``time`` into the store."""
        for pid, value in self.values_at(time).items():
            self.store.set(pid, value, notify=notify)

    @property
    def frame_count(self) -> int:
        return int(math.floor(self.duration * self.fps + 1e-9)) + 1

    def frame_time(self, frame: int) -> float:
        return frame / self.fps

    # --- serialisation ----------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "fps": self.fps,
            "curves": {pid: c.to_json() for pid, c in self.curves.items() if c.keys},
        }

    def load_json(self, data: dict[str, Any]) -> list[str]:
        self.duration = float(data.get("duration", 30.0))
        self.fps = float(data.get("fps", 60.0))
        self.curves.clear()
        problems = []
        for pid, entries in data.get("curves", {}).items():
            if pid not in self.store.registry:
                problems.append(f"animation of unknown property '{pid}'")
                continue
            try:
                self.curves[pid] = Curve.from_json(self.store.registry[pid], entries)
            except (KeyError, TypeError, ValueError) as exc:
                problems.append(f"animation of '{pid}': {exc}")
        return problems
