"""Vehicle markers: an icon at the head of the drawn track instead of the glowing dot (a ship
on the ferry crossing, a plane on the flight, a cyclist on the ride).

Top views (ship, plane, train, car, canoe) point in the direction of travel on screen; side
views (bike, runner, hiker) face right and are mirrored when the track runs to the left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VEHICLE_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons" / "vehicles"
VEHICLES = {"ship": "rotate", "plane": "rotate", "train": "rotate", "car": "rotate",
            "canoe": "rotate", "bike": "flip", "runner": "flip", "hiker": "flip"}  # fmt: skip
MARKER_OPTIONS = [("dot", "Glowing dot")] + [(name, name.title()) for name in VEHICLES]


@dataclass
class VehicleMarker:
    name: str  # key of VEHICLES
    position: np.ndarray  # ENU of the head
    ahead: np.ndarray  # ENU a little further along the track (the direction of travel)
    size_px: float  # at 1080p
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # multiplies the icon (sRGB)
    label: str = ""  # a caption moving with the vehicle (e.g. "2024")

    @property
    def icon_path(self) -> Path:
        return VEHICLE_DIR / f"{self.name}.png"

    @property
    def mode(self) -> str:
        return VEHICLES[self.name]


def screen_direction(view_proj, camera_position, a, b, width: int, height: int):
    """Direction from ENU point a to b on screen (pixels, y down), or None if not visible."""
    from pyglm import glm

    out = []
    for point in (a, b):
        rel = np.asarray(point, dtype=np.float64) - np.asarray(camera_position, dtype=np.float64)
        clip = view_proj * glm.vec4(float(rel[0]), float(rel[1]), float(rel[2]), 1.0)
        if clip.w <= 1e-6:
            return None
        out.append((clip.x / clip.w * 0.5 * width, -clip.y / clip.w * 0.5 * height))
    d = np.array(out[1]) - np.array(out[0])
    n = float(np.hypot(*d))
    return d / n if n > 1e-6 else None


def icon_rotation(direction, mode: str) -> tuple[float, bool]:
    """(angle in radians for the icon quad, mirrored) for a screen direction (y down)."""
    if direction is None:
        return 0.0, False
    dx, dy = float(direction[0]), float(direction[1])
    if mode == "flip":
        return 0.0, dx < 0.0
    return math.atan2(dx, -dy), False  # the icon's "up" turns to the direction
