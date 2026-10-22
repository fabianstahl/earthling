"""Declarative, typed properties (Qt independent).

Every adjustable value of the scene (terrain exaggeration, sun time, fog density, ...) is
declared once as a :class:`PropertyDef`. From that declaration the UI builds its widgets, the
renderer binds shader uniforms, the scene file serialises values and (later) the timeline
animates them. A :class:`PropertyStore` holds the current values and notifies listeners.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class PType(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    ENUM = "enum"
    COLOR = "color"  # (r, g, b) floats 0..1, sRGB
    VEC3 = "vec3"
    DATETIME = "datetime"  # naive local datetime
    CAMERA = "camera"  # (x, y, z, heading, pitch, roll): ENU metres and degrees
    TEXT = "text"  # a line of text (keys switch it, like enums)


Condition = tuple[str, tuple, bool]  # (property id, values, negate)


def shown_if(pid: str, *values: Any) -> tuple[Condition, ...]:
    """``when=``: show a property only while ``pid`` has one of ``values`` (default True)."""
    return ((pid, values or (True,), False),)


def shown_unless(pid: str, *values: Any) -> tuple[Condition, ...]:
    """``when=``: hide a property while ``pid`` has one of ``values``."""
    return ((pid, values, True),)


@dataclass(frozen=True)
class PropertyDef:
    id: str  # "section.name", unique
    label: str
    type: PType
    default: Any
    section: str = "General"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 2
    unit: str = ""
    logarithmic: bool = False  # slider maps logarithmically (minimum must be > 0)
    options: tuple[tuple[str, str], ...] = ()  # enum: (value, label)
    uniform: str | None = None  # shader uniform name bound automatically
    animatable: bool = True
    tooltip: str = ""
    # how the inspector shows it (no effect on rendering)
    advanced: bool = False  # behind the section's "More settings"
    when: tuple[Condition, ...] = ()  # shown only while all conditions hold (shown_if ...)
    switch: bool = False  # a boolean that is its section's on/off switch
    keywords: str = ""  # extra search words

    def shown(self, values: dict[str, Any]) -> bool:
        """Whether the inspector shows this property for the current ``values``."""
        return all((values.get(pid) in allowed) != negate for pid, allowed, negate in self.when)

    def coerce(self, value: Any) -> Any:
        """Convert/validate ``value``; raises ValueError for unusable input."""
        t = self.type
        if t is PType.FLOAT:
            v = float(value)
            if math.isnan(v):
                raise ValueError(f"{self.id}: NaN")
            return self._clamp(v)
        if t is PType.INT:
            return int(self._clamp(int(round(float(value)))))
        if t is PType.BOOL:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if t is PType.ENUM:
            value = str(value)
            if value not in {v for v, _ in self.options}:
                raise ValueError(
                    f"{self.id}: '{value}' is not one of {[v for v, _ in self.options]}"
                )
            return value
        if t in (PType.COLOR, PType.VEC3):
            items = tuple(float(v) for v in value)
            if len(items) != 3:
                raise ValueError(f"{self.id}: expected 3 components")
            if t is PType.COLOR:
                items = tuple(min(1.0, max(0.0, v)) for v in items)
            return items
        if t is PType.CAMERA:
            items = tuple(float(v) for v in value)
            if len(items) != 6:
                raise ValueError(f"{self.id}: expected 6 components")
            return items
        if t is PType.TEXT:
            return "" if value is None else str(value)
        if t is PType.DATETIME:
            if isinstance(value, datetime):
                return value.replace(tzinfo=None, microsecond=0)
            return datetime.fromisoformat(str(value)).replace(tzinfo=None, microsecond=0)
        raise ValueError(f"unsupported type {t}")

    def _clamp(self, v: float) -> float:
        if self.minimum is not None:
            v = max(self.minimum, v)
        if self.maximum is not None:
            v = min(self.maximum, v)
        return v

    def to_json(self, value: Any) -> Any:
        if self.type is PType.DATETIME:
            return value.isoformat()
        if self.type in (PType.COLOR, PType.VEC3, PType.CAMERA):
            return list(value)
        return value

    def from_json(self, value: Any) -> Any:
        return self.coerce(value)


class PropertyRegistry:
    def __init__(self, defs: Iterable[PropertyDef] = ()) -> None:
        self._defs: dict[str, PropertyDef] = {}
        for d in defs:
            self.add(d)

    def add(self, d: PropertyDef) -> PropertyDef:
        if d.id in self._defs:
            raise ValueError(f"duplicate property id '{d.id}'")
        d.coerce(d.default)  # validate default early
        self._defs[d.id] = d
        return d

    def remove(self, pid: str) -> None:
        self._defs.pop(pid, None)

    def extend(self, defs: Iterable[PropertyDef]) -> None:
        for d in defs:
            self.add(d)

    def __getitem__(self, pid: str) -> PropertyDef:
        return self._defs[pid]

    def __contains__(self, pid: str) -> bool:
        return pid in self._defs

    def __iter__(self):
        return iter(self._defs.values())

    def __len__(self) -> int:
        return len(self._defs)

    def sections(self) -> list[tuple[str, list[PropertyDef]]]:
        """Sections in declaration order with their properties."""
        out: dict[str, list[PropertyDef]] = {}
        for d in self._defs.values():
            out.setdefault(d.section, []).append(d)
        return list(out.items())


# --- declaration shortcuts (used by every module that declares properties) ------------------
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


Listener = Callable[[str, Any], None]


@dataclass
class PropertyStore:
    registry: PropertyRegistry
    values: dict[str, Any] = field(default_factory=dict)
    _listeners: list[Listener] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        for d in self.registry:
            self.values.setdefault(d.id, d.coerce(d.default))

    def __getitem__(self, pid: str) -> Any:
        return self.values[pid]

    def get(self, pid: str, default: Any = None) -> Any:
        return self.values.get(pid, default)

    def set(self, pid: str, value: Any, notify: bool = True) -> bool:
        """Set a value (coerced). Returns True if it changed."""
        value = self.registry[pid].coerce(value)
        if self.values.get(pid) == value:
            return False
        self.values[pid] = value
        if notify:
            for listener in list(self._listeners):
                listener(pid, value)
        return True

    def reset(self, pid: str) -> None:
        self.set(pid, self.registry[pid].default)

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    # --- serialisation ----------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {pid: self.registry[pid].to_json(v) for pid, v in self.values.items()}

    def load_json(self, data: dict[str, Any], notify: bool = True) -> list[str]:
        """Load values; unknown ids / invalid values are skipped and reported."""
        problems = []
        for pid, raw in data.items():
            if pid not in self.registry:
                problems.append(f"unknown property '{pid}'")
                continue
            try:
                self.set(pid, self.registry[pid].from_json(raw), notify=notify)
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))
        return problems


def bind_uniforms(program, store: PropertyStore) -> None:
    """Write every property with a ``uniform`` present in ``program``. Colours are passed as
    sRGB; the shaders linearise them (once) where they shade with them."""
    for d in store.registry:
        if d.uniform is None:
            continue
        try:
            uniform = program[d.uniform]
        except KeyError:
            continue
        value = store.values[d.id]
        if d.type is PType.BOOL:
            uniform.value = bool(value)
        elif d.type is PType.ENUM:
            uniform.value = [v for v, _ in d.options].index(value)
        elif d.type is PType.COLOR:
            uniform.value = tuple(value)
        elif d.type in (PType.DATETIME, PType.CAMERA):
            continue
        else:
            uniform.value = value
