"""Camera model and interactive controllers.

Positions are float64 ENU meters. Rendering is camera-relative: the view matrix only contains
the rotation, and every drawable passes ``origin - camera.position`` as an offset uniform. That
keeps float32 precision high even hundreds of kilometres away from the frame origin.

Orientation is given by heading (deg, 0 = north, clockwise), pitch (deg, positive = up) and roll.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pyglm import glm


def forward_vector(heading: float, pitch: float) -> np.ndarray:
    h, p = math.radians(heading), math.radians(pitch)
    return np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, -5000.0, 3000.0]))
    heading: float = 0.0
    pitch: float = -30.0
    roll: float = 0.0
    fov_y: float = 50.0
    near: float = 0.5
    far: float = 2_000_000.0

    def copy(self) -> Camera:
        return Camera(
            self.position.copy(),
            self.heading,
            self.pitch,
            self.roll,
            self.fov_y,
            self.near,
            self.far,
        )

    @property
    def forward(self) -> np.ndarray:
        return forward_vector(self.heading, self.pitch)

    def rotation_matrix(self) -> glm.mat4:
        """View rotation (world -> camera) with the camera at the origin."""
        f = glm.vec3(*self.forward)
        world_up = glm.vec3(0.0, 0.0, 1.0)
        if abs(glm.dot(f, world_up)) > 0.999:
            world_up = glm.vec3(0.0, 1.0, 0.0)
        view = glm.lookAt(glm.vec3(0.0), f, world_up)
        if self.roll:
            view = glm.rotate(glm.mat4(1.0), math.radians(self.roll), glm.vec3(0, 0, 1)) * view
        return view

    def projection_matrix(self, aspect: float) -> glm.mat4:
        return glm.perspective(math.radians(self.fov_y), aspect, self.near, self.far)

    def view_projection(self, aspect: float) -> glm.mat4:
        return self.projection_matrix(aspect) * self.rotation_matrix()

    def relative(self, origin) -> tuple[float, float, float]:
        """Offset uniform for a drawable whose vertices are relative to ``origin``."""
        d = np.asarray(origin, dtype=np.float64) - self.position
        return float(d[0]), float(d[1]), float(d[2])

    def look_at(self, target) -> None:
        d = np.asarray(target, dtype=np.float64) - self.position
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return
        self.heading = math.degrees(math.atan2(d[0], d[1])) % 360.0
        self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2] / dist))))

    @property
    def log_depth_coef(self) -> float:
        return 1.0 / math.log2(self.far + 1.0)


class OrbitController:
    """Orbits around a target point. Left drag rotates, right/middle drag pans, wheel zooms."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self.target = np.zeros(3)
        self.distance = 10_000.0
        self.heading = camera.heading
        self.pitch = -35.0
        self.apply()

    def apply(self) -> None:
        self.pitch = max(-89.5, min(89.5, self.pitch))
        self.distance = max(10.0, min(5_000_000.0, self.distance))
        f = forward_vector(self.heading, self.pitch)
        self.camera.heading = self.heading % 360.0
        self.camera.pitch = self.pitch
        self.camera.position = self.target - f * self.distance

    def rotate(self, dx_px: float, dy_px: float) -> None:
        self.heading += dx_px * 0.3
        self.pitch -= dy_px * 0.3
        self.apply()

    def pan(self, dx_px: float, dy_px: float, viewport_height: int) -> None:
        scale = 2.0 * self.distance * math.tan(math.radians(self.camera.fov_y) / 2)
        scale /= max(1, viewport_height)
        h = math.radians(self.heading)
        right = np.array([math.cos(h), -math.sin(h), 0.0])
        ahead = np.array([math.sin(h), math.cos(h), 0.0])
        self.target = self.target - right * dx_px * scale + ahead * dy_px * scale
        self.apply()

    def zoom(self, steps: float) -> None:
        self.distance *= 0.85**steps
        self.apply()

    def frame_bounds(self, lo: np.ndarray, hi: np.ndarray) -> None:
        """Place the camera so the axis-aligned box [lo, hi] is fully visible."""
        self.target = (np.asarray(lo) + np.asarray(hi)) / 2.0
        radius = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) / 2.0
        half_fov = math.radians(self.camera.fov_y) / 2.0
        self.distance = max(500.0, radius / math.sin(half_fov) * 1.1)
        self.apply()

    def sync_from_camera(self, target) -> None:
        """Take over the current camera pose, orbiting around ``target``."""
        self.target = np.asarray(target, dtype=np.float64).copy()
        d = self.target - self.camera.position
        self.distance = max(10.0, float(np.linalg.norm(d)))
        self.camera.look_at(self.target)
        self.heading = self.camera.heading
        self.pitch = self.camera.pitch
        self.apply()


class FlyController:
    """First-person flight: WASD move, Q/E down/up, mouse drag looks around.

    Speed scales with the height above ground so that both low valley flights and
    high overview moves feel natural.
    """

    FORWARD, BACK, LEFT, RIGHT, DOWN, UP = "forward", "back", "left", "right", "down", "up"

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self.pressed: set[str] = set()
        self.speed_multiplier = 1.0  # adjusted with the mouse wheel
        self.boost = 1.0  # Shift / Ctrl
        self.look_sensitivity = 0.15

    def look(self, dx_px: float, dy_px: float) -> None:
        self.camera.heading = (self.camera.heading + dx_px * self.look_sensitivity) % 360.0
        self.camera.pitch = max(-89.5, min(89.5, self.camera.pitch - dy_px * self.look_sensitivity))

    def speed(self, height_above_ground: float) -> float:
        return max(15.0, abs(height_above_ground) * 0.9) * self.speed_multiplier * self.boost

    def step(self, dt: float, height_above_ground: float) -> bool:
        """Advance the camera; returns True if it moved."""
        if not self.pressed or dt <= 0:
            return False
        f = self.camera.forward
        h = math.radians(self.camera.heading)
        right = np.array([math.cos(h), -math.sin(h), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        move = np.zeros(3)
        if self.FORWARD in self.pressed:
            move += f
        if self.BACK in self.pressed:
            move -= f
        if self.RIGHT in self.pressed:
            move += right
        if self.LEFT in self.pressed:
            move -= right
        if self.UP in self.pressed:
            move += up
        if self.DOWN in self.pressed:
            move -= up
        norm = np.linalg.norm(move)
        if norm < 1e-9:
            return False
        self.camera.position = self.camera.position + move / norm * self.speed(
            height_above_ground
        ) * min(dt, 0.1)
        return True
