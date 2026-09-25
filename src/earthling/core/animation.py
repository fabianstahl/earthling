"""Keyframe animation of properties (Qt independent).

Every animated property has a :class:`Curve` of :class:`Keyframe` objects. The interpolation
stored on a key applies to the segment *starting* at that key. Bezier handles are normalised
to the segment: ``out_handle = (x, y)`` places the first control point at
``(t0 + x * duration, v0 + y * delta)``, ``in_handle = (x, y)`` the second one at
``(t1 - x * duration, v1 - y * delta)``. That works for any value type (scalars, colours,
vectors, datetimes) and matches CSS ``cubic-bezier`` for the presets.

Handle modes (per key, like Blender): ``auto`` handles are computed from the neighbouring keys
(auto-clamped: smooth, flat at extremes, no overshoot), ``vector`` handles point at the
neighbour (straight), ``aligned`` and ``free`` handles are the stored ones (aligned: the editor
keeps both sides collinear). The graph editor works on each curve's *scalar channel*: the value
for float/int/datetime properties, otherwise the "progress" key index + blend factor.

Type rules
----------
* float / int: interpolated (ints are rounded)
* color: interpolated in OKLab (perceptually even blends)
* vec3: per component
* datetime: as continuous time
* bool / enum: always step (hold until the next key)
* camera: centripetal Catmull-Rom positions + quaternion squad rotations through all keys; the
  key's interpolation shapes the timing within the segment. With ``constant_speed`` the keys
  become waypoints travelled at constant speed between the first and last key time.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any

import numpy as np

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
VECTOR_HANDLE = (1.0 / 3.0, 1.0 / 3.0)  # pointing at the neighbouring key


class HandleMode(StrEnum):
    AUTO = "auto"
    ALIGNED = "aligned"
    FREE = "free"
    VECTOR = "vector"


HANDLE_LABELS = {
    HandleMode.AUTO: "Auto (clamped)",
    HandleMode.ALIGNED: "Aligned",
    HandleMode.FREE: "Free",
    HandleMode.VECTOR: "Vector",
}


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
    handle_mode: HandleMode = HandleMode.AUTO

    def copy(self) -> Keyframe:
        return Keyframe(
            self.time, self.value, self.interp, self.out_handle, self.in_handle, self.handle_mode
        )


def _usable_handle(handle: tuple[float, float], towards: tuple[float, float]):
    """Presets have zero-length handles (linear, ease-out start); give them a third of the
    way to the other control point, which keeps the tangent (linear stays exactly linear)."""
    if abs(handle[0]) + abs(handle[1]) > 1e-6:
        return handle
    return towards[0] / 3.0, towards[1] / 3.0


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
    constant_speed: bool = False  # camera curves only
    _spline_cache: tuple | None = field(default=None, repr=False, compare=False)

    def _spline(self):
        from earthling.core.camera_path import CameraSpline

        signature = tuple((k.time, k.value) for k in self.keys)
        if self._spline_cache is None or self._spline_cache[0] != signature:
            spline = CameraSpline([k.value for k in self.keys])
            self._spline_cache = (signature, spline, spline.arc_table())
        return self._spline_cache[1], self._spline_cache[2]

    def _evaluate_camera(self, time: float, i: int, f: float) -> Any:
        spline, (g_table, lengths) = self._spline()
        if self.constant_speed and len(self.keys) > 1 and lengths[-1] > 0:
            t0, t1 = self.keys[0].time, self.keys[-1].time
            s = (time - t0) / max(t1 - t0, 1e-9) * lengths[-1]
            g = float(np.interp(s, lengths, g_table))
            segment = min(int(g), len(self.keys) - 2)
            return spline.pose(segment, g - segment)
        return spline.pose(i, f)

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

    # --- scalar channel and handles -----------------------------------------------------
    @property
    def graph_kind(self) -> str | None:
        """'value' (plotted by value), 'progress' (key index + blend) or None (stepped)."""
        t = self.definition.type
        if _is_stepped(t):
            return None
        return "value" if t in (PType.FLOAT, PType.INT, PType.DATETIME) else "progress"

    def to_scalar(self, value: Any, index: float = 0.0) -> float:
        """Scalar channel of a value (``index`` is used for progress channels)."""
        t = self.definition.type
        if t in (PType.FLOAT, PType.INT):
            return float(value)
        if t is PType.DATETIME:
            return value.timestamp()
        return float(index)

    def from_scalar(self, scalar: float, like: Any) -> Any:
        """Inverse of :meth:`to_scalar` for value channels (keeps a datetime's timezone)."""
        t = self.definition.type
        if t is PType.DATETIME:
            return like + timedelta(seconds=scalar - like.timestamp())
        if t in (PType.FLOAT, PType.INT):
            return self.definition.coerce(scalar)
        return like

    def scalars(self) -> list[float]:
        return [self.to_scalar(k.value, i) for i, k in enumerate(self.keys)]

    def _auto_slope(self, i: int, values: list[float]) -> float:
        """Auto-clamped slope (scalar units per second) at key ``i``."""
        if i == 0 or i == len(self.keys) - 1:
            return 0.0
        vp, vk, vn = values[i - 1], values[i], values[i + 1]
        if (vk - vp) * (vn - vk) <= 0.0:  # extreme or flat: horizontal handle
            return 0.0
        slope = (vn - vp) / max(self.keys[i + 1].time - self.keys[i - 1].time, 1e-9)
        # clamp so neither neighbouring segment overshoots (a third of the segment's handle
        # may rise at most by the segment's delta); one slope for both sides keeps it smooth
        limit = min(
            3.0 * abs(vk - vp) / max(self.keys[i].time - self.keys[i - 1].time, 1e-9),
            3.0 * abs(vn - vk) / max(self.keys[i + 1].time - self.keys[i].time, 1e-9),
        )
        return math.copysign(min(abs(slope), limit), slope)

    def effective_handles(
        self, i: int, values: list[float] | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """(in_handle, out_handle) of key ``i`` after applying its handle mode."""
        key = self.keys[i]
        mode = key.handle_mode
        if mode is HandleMode.VECTOR:
            return VECTOR_HANDLE, VECTOR_HANDLE
        if mode is not HandleMode.AUTO:
            return key.in_handle, key.out_handle
        values = self.scalars() if values is None else values
        slope = self._auto_slope(i, values)
        in_h = out_h = DEFAULT_HANDLE
        third = 1.0 / 3.0
        if i > 0:
            dur, delta = key.time - self.keys[i - 1].time, values[i] - values[i - 1]
            y = slope * third * dur / delta if delta else 0.0
            in_h = (third, min(max(y, 0.0), 1.0))
        if i < len(self.keys) - 1:
            dur, delta = self.keys[i + 1].time - key.time, values[i + 1] - values[i]
            y = slope * third * dur / delta if delta else 0.0
            out_h = (third, min(max(y, 0.0), 1.0))
        return in_h, out_h

    def segment_ease(self, i: int, values: list[float] | None = None):
        """(x1, y1, x2, y2) cubic-bezier timing of the segment starting at key ``i``."""
        k0 = self.keys[i]
        if k0.interp is not Interp.BEZIER:
            return PRESETS.get(k0.interp, PRESETS[Interp.LINEAR])
        needs_values = HandleMode.AUTO in (k0.handle_mode, self.keys[i + 1].handle_mode)
        if needs_values and values is None:
            values = self.scalars()
        ox, oy = self.effective_handles(i, values)[1]
        ix, iy = self.effective_handles(i + 1, values)[0]
        return ox, oy, 1.0 - ix, 1.0 - iy

    def blend_at(self, time: float) -> tuple[int, float]:
        """Segment index and blend factor at ``time`` (f may leave [0, 1] with overshoot)."""
        keys = self.keys
        if time <= keys[0].time or len(keys) == 1:
            return 0, 0.0
        if time >= keys[-1].time:
            return len(keys) - 2, 1.0
        i = bisect.bisect_right([k.time for k in keys], time) - 1
        k0, k1 = keys[i], keys[i + 1]
        duration = k1.time - k0.time
        x = (time - k0.time) / duration if duration > 0 else 1.0
        if k0.interp is Interp.STEP or _is_stepped(self.definition.type):
            return i, 0.0
        return i, cubic_bezier_ease(x, *self.segment_ease(i))

    def freeze_handles(self, i: int, values: list[float] | None = None) -> None:
        """Store the effective handles of key ``i`` so a mode change does not move them."""
        key = self.keys[i]
        key.in_handle, key.out_handle = self.effective_handles(i, values)

    def drag_handle(self, i: int, side: str, time: float, scalar: float) -> None:
        """Move the ``side`` ('in' or 'out') handle of key ``i`` to the absolute point
        (time, scalar channel value). Automatic handles become aligned, the segment becomes
        Bézier (keeping the other end's shape), aligned handles stay collinear."""
        keys = self.keys
        seg = i if side == "out" else i - 1
        if not 0 <= seg < len(keys) - 1 or self.graph_kind is None:
            return
        values = self.scalars()
        key = keys[i]
        if key.handle_mode in (HandleMode.AUTO, HandleMode.VECTOR):
            self.freeze_handles(i, values)
            key.handle_mode = HandleMode.ALIGNED
        start, end = keys[seg], keys[seg + 1]
        if start.interp is not Interp.BEZIER:
            x1, y1, x2, y2 = self.segment_ease(seg, values)
            other_index = seg + 1 if side == "out" else seg
            other = keys[other_index]
            if other.handle_mode in (HandleMode.AUTO, HandleMode.VECTOR):
                self.freeze_handles(other_index, values)
                other.handle_mode = HandleMode.FREE
            if side == "out":
                end.in_handle = _usable_handle((1.0 - x2, 1.0 - y2), (1.0 - x1, 1.0 - y1))
            else:
                start.out_handle = _usable_handle((x1, y1), (x2, y2))
            start.interp = Interp.BEZIER
        duration = max(end.time - start.time, 1e-9)
        delta = values[seg + 1] - values[seg]
        if side == "out":
            hx = min(max((time - key.time) / duration, 1e-3), 1.0)
            hy = (scalar - values[i]) / delta if delta else key.out_handle[1]
            key.out_handle = (hx, hy)
        else:
            hx = min(max((key.time - time) / duration, 1e-3), 1.0)
            hy = (values[i] - scalar) / delta if delta else key.in_handle[1]
            key.in_handle = (hx, hy)
        if key.handle_mode is HandleMode.ALIGNED:
            self._align(i, side, values)

    def _align(self, i: int, side: str, values: list[float]) -> None:
        """Make the handle opposite to ``side`` collinear with it (keeping its length)."""
        keys = self.keys
        key = keys[i]
        if side == "out":
            dur, delta = keys[i + 1].time - key.time, values[i + 1] - values[i]
            hx, hy = key.out_handle
        else:
            dur, delta = key.time - keys[i - 1].time, values[i] - values[i - 1]
            hx, hy = key.in_handle
        slope = hy * delta / (hx * dur)
        j = i - 1 if side == "out" else i + 1
        if not 0 <= j < len(keys):
            return
        other_dur = abs(keys[j].time - key.time)
        other_delta = values[max(i, j)] - values[min(i, j)]
        if not other_delta:
            return
        if side == "out":
            ix = key.in_handle[0]
            key.in_handle = (ix, slope * ix * other_dur / other_delta)
        else:
            ox = key.out_handle[0]
            key.out_handle = (ox, slope * ox * other_dur / other_delta)

    def handle_points(self, i: int, values: list[float] | None = None):
        """Absolute (time, scalar) positions of the visible handles of key ``i``:
        {'in': (t, v), 'out': (t, v)} for the adjacent Bézier segments."""
        keys = self.keys
        values = self.scalars() if values is None else values
        (ix, iy), (ox, oy) = self.effective_handles(i, values)
        out = {}
        key = keys[i]
        if i < len(keys) - 1 and key.interp is Interp.BEZIER:
            dur, delta = keys[i + 1].time - key.time, values[i + 1] - values[i]
            out["out"] = (key.time + ox * dur, values[i] + oy * delta)
        if i > 0 and keys[i - 1].interp is Interp.BEZIER:
            dur, delta = key.time - keys[i - 1].time, values[i] - values[i - 1]
            out["in"] = (key.time - ix * dur, values[i] - iy * delta)
        return out

    def scalar_at(self, time: float) -> float:
        """The scalar channel of the curve at ``time`` (what the graph editor plots)."""
        if self.graph_kind == "progress":
            if len(self.keys) == 1:
                return 0.0
            i, f = self.blend_at(time)
            return i + f
        return self.to_scalar(self.evaluate(time))

    def evaluate(self, time: float) -> Any:
        keys = self.keys
        if not keys:
            raise ValueError(f"curve {self.pid} has no keys")
        if time <= keys[0].time:
            return keys[0].value
        if time >= keys[-1].time:
            return keys[-1].value
        i, f = self.blend_at(time)
        k0, k1 = keys[i], keys[i + 1]
        if k0.interp is Interp.STEP or _is_stepped(self.definition.type):
            return k0.value
        if self.definition.type is PType.CAMERA:
            return self._evaluate_camera(time, i, f)
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
            if k.handle_mode is not HandleMode.AUTO or k.interp is Interp.BEZIER:
                entry["handles"] = str(k.handle_mode)
            if k.handle_mode in (HandleMode.FREE, HandleMode.ALIGNED):
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
                # files from before handle modes stored explicit handles only
                HandleMode(entry.get("handles", "free" if "out" in entry else "auto")),
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
        constant = bool(self.store.get("camera.constant_speed", False))
        out = {}
        for pid, curve in self.curves.items():
            if curve.keys:
                curve.constant_speed = constant
                out[pid] = curve.evaluate(time)
        return out

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


def layer_switch_warnings(animation: Animation) -> list[tuple[str, float, str]]:
    """Keys that switch a texture-layer slot while it is visible (a visible jump).

    Slot A is visible while ``layers.mix < 1``, slot B while ``layers.mix > 0``. Returns
    (property id, key time, message) triples.
    """
    store = animation.store
    mix_curve = animation.curves.get("layers.mix")

    def mix_at(t: float) -> float:
        if mix_curve is not None and mix_curve.keys:
            return float(mix_curve.evaluate(t))
        return float(store.get("layers.mix", 0.0))

    warnings = []
    for pid, visible, slot in (
        ("layers.a", lambda m: m < 1.0 - 1e-6, "A"),
        ("layers.b", lambda m: m > 1e-6, "B"),
    ):
        curve = animation.curves.get(pid)
        if curve is None:
            continue
        for k0, k1 in zip(curve.keys, curve.keys[1:], strict=False):
            if k1.value != k0.value and visible(mix_at(k1.time)):
                warnings.append(
                    (pid, k1.time, f"layer slot {slot} switches from '{k0.value}' to "
                                   f"'{k1.value}' at {k1.time:.2f} s while it is visible")
                )  # fmt: skip
    return warnings
