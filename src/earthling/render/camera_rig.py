"""Final camera pose for a point in time: keyframed path, look-at target or follow-the-hiker.

The camera mode (``camera.mode``) is a stepped property, so each timeline segment has one mode.
When it switches, the previous and the new mode's poses are blended over
``camera.transition`` seconds, so cuts only happen when the transition is 0.
"""

from __future__ import annotations

import math

import numpy as np

from earthling.core.animation import Animation
from earthling.core.camera_path import orientation_to_quat, quat_to_orientation, slerp
from earthling.core.properties import PropertyStore

Pose = tuple[float, float, float, float, float, float]


def look_orientation(
    eye: np.ndarray, target: np.ndarray, roll: float = 0.0
) -> tuple[float, float, float]:
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 0.0, 0.0, roll
    heading = math.degrees(math.atan2(d[0], d[1])) % 360.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2] / dist))))
    return heading, pitch, roll


def blend_poses(a: Pose, b: Pose, f: float) -> Pose:
    pa, pb = np.array(a[:3]), np.array(b[:3])
    p = pa + (pb - pa) * f
    q = slerp(orientation_to_quat(*a[3:]), orientation_to_quat(*b[3:]), f)
    h, pitch, roll = quat_to_orientation(q)
    return (float(p[0]), float(p[1]), float(p[2]), h, pitch, roll)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class CameraRig:
    def __init__(self, animation: Animation, store: PropertyStore) -> None:
        self.animation = animation
        self.store = store
        self.path = None  # tracks.ProgressPath of the renderer (for follow mode)

    def value(self, pid: str, time: float):
        """Property value at an arbitrary time (animated or the current static value)."""
        curve = self.animation.curves.get(pid)
        if curve is not None and curve.keys:
            curve.constant_speed = bool(self.store.get("camera.constant_speed", False))
            return curve.evaluate(time)
        return self.store[pid]

    # --- modes ------------------------------------------------------------------------
    def pose_for_mode(self, mode: str, time: float) -> Pose:
        keyed = self.value("camera.pose", time)
        if mode == "look_at":
            eye = np.array(keyed[:3])
            h, p, r = look_orientation(eye, np.array(self.value("camera.target", time)), keyed[5])
            return (keyed[0], keyed[1], keyed[2], h, p, r)
        if mode == "follow":
            pose = self.follow_pose(time)
            return pose if pose is not None else keyed
        return keyed

    def follow_pose(self, time: float) -> Pose | None:
        path = self.path
        if path is None or path.total_m <= 0:
            return None
        lag = max(0.0, self.value("follow.lag", time))
        mode = self.value("progress.mode", time)
        head = path.distance_for(self.value("progress.head", max(0.0, time - lag)), mode)
        here = path.position_at(head)
        if here is None:
            return None
        back = path.position_at(max(0.0, head - 150.0))
        direction = here - back
        direction[2] = 0.0
        n = float(np.linalg.norm(direction))
        direction = direction / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        angle = math.radians(self.value("follow.angle", time))
        c, s = math.cos(angle), math.sin(angle)
        rotated = np.array(
            [direction[0] * c - direction[1] * s, direction[0] * s + direction[1] * c, 0.0]
        )
        eye = here - rotated * self.value("follow.distance", time)
        eye[2] += self.value("follow.height", time)
        ahead = path.position_at(head + self.value("follow.look_ahead", time))
        target = ahead if ahead is not None else here
        h, p, r = look_orientation(eye, target)
        return (float(eye[0]), float(eye[1]), float(eye[2]), h, p, r)

    # --- final pose -------------------------------------------------------------------
    def pose(self, time: float) -> Pose:
        mode = self.value("camera.mode", time)
        current = self.pose_for_mode(mode, time)
        transition = float(self.store.get("camera.transition", 0.0))
        curve = self.animation.curves.get("camera.mode")
        if transition <= 0 or curve is None or len(curve.keys) < 2:
            return current
        # the most recent mode switch at or before `time`
        previous_mode, switch_time = None, None
        for k0, k1 in zip(curve.keys, curve.keys[1:], strict=False):
            if k1.time <= time and k1.value != k0.value:
                previous_mode, switch_time = k0.value, k1.time
        if previous_mode is None or time - switch_time >= transition:
            return current
        f = smoothstep((time - switch_time) / transition)
        return blend_poses(self.pose_for_mode(previous_mode, time), current, f)


MIN_CLEARANCE_M = 5.0


def clamp_above_ground(pose: Pose, frame, terrain_data, exaggeration: float) -> Pose:
    """Keep a pose above the terrain (follow/look-at paths may cut through ridges)."""
    height_at = getattr(terrain_data, "height_at", None)
    if frame is None or height_at is None:
        return pose
    lat, lon, h = frame.enu_to_geodetic(np.array(pose[:3]))
    ground = height_at(float(lon), float(lat))
    if ground is None:
        return pose
    lift = ground * exaggeration + MIN_CLEARANCE_M - float(h)
    if lift <= 0:
        return pose
    return (pose[0], pose[1], pose[2] + lift, *pose[3:])
