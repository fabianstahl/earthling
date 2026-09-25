"""Camera path interpolation: centripetal Catmull-Rom positions, quaternion squad rotations.

A camera pose is ``(x, y, z, heading, pitch, roll)`` in ENU metres and degrees (heading 0 =
north, clockwise; pitch positive = up), matching :class:`earthling.render.camera.Camera`.
"""

from __future__ import annotations

import math

import numpy as np

Pose = tuple[float, float, float, float, float, float]


# --- orientation <-> quaternion ----------------------------------------------------------
def orientation_matrix(heading: float, pitch: float, roll: float) -> np.ndarray:
    """Columns: right, forward, up (ENU) of a camera."""
    h, p, r = math.radians(heading), math.radians(pitch), math.radians(roll)
    forward = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    up = np.cross(right, forward)
    if roll:
        c, s = math.cos(r), math.sin(r)
        right, up = right * c + up * s, up * c - right * s
    return np.column_stack([right, forward, up])


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation_to_quat(heading: float, pitch: float, roll: float) -> np.ndarray:
    return matrix_to_quat(orientation_matrix(heading, pitch, roll))


def quat_to_orientation(q: np.ndarray) -> tuple[float, float, float]:
    m = quat_to_matrix(q)
    right, forward = m[:, 0], m[:, 1]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, forward[2]))))
    heading = math.degrees(math.atan2(forward[0], forward[1])) % 360.0
    # roll: angle between the actual right vector and the unrolled one
    h = math.radians(heading)
    flat_right = np.array([math.cos(h), -math.sin(h), 0.0])
    flat_up = np.cross(flat_right, forward)
    roll = math.degrees(math.atan2(np.dot(right, flat_up), np.dot(right, flat_right)))
    return heading, pitch, roll


# --- quaternion interpolation ------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_log(q: np.ndarray) -> np.ndarray:
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(4)
    angle = math.atan2(n, q[0])
    return np.concatenate([[0.0], v / n * angle])


def quat_exp(q: np.ndarray) -> np.ndarray:
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(n)], v / n * math.sin(n)])


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + (b - a) * t
        return q / np.linalg.norm(q)
    theta = math.acos(d)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


def squad_tangent(q_prev: np.ndarray, q: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    inv = quat_conj(q)
    a = quat_log(quat_mul(inv, q_next))
    b = quat_log(quat_mul(inv, q_prev))
    return quat_mul(q, quat_exp(-(a + b) / 4.0))


def align_hemispheres(quats: list[np.ndarray]) -> list[np.ndarray]:
    """Flip signs so consecutive quaternions take the short way round."""
    out = [quats[0]]
    for q in quats[1:]:
        out.append(-q if np.dot(out[-1], q) < 0 else q)
    return out


# --- positions ---------------------------------------------------------------------------
def catmull_rom(p0, p1, p2, p3, u: float, alpha: float = 0.5) -> np.ndarray:
    """Centripetal (alpha = 0.5) Catmull-Rom point between p1 and p2 at u in [0, 1]."""

    def knot(t: float, a: np.ndarray, b: np.ndarray) -> float:
        return t + max(float(np.linalg.norm(b - a)), 1e-6) ** alpha

    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    t = t1 + (t2 - t1) * u
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


class CameraSpline:
    """Interpolates a list of poses (at their key indices); ``u`` runs 0..n-1 across keys."""

    def __init__(self, poses: list[Pose]) -> None:
        self.poses = [tuple(float(v) for v in p) for p in poses]
        self.points = [np.array(p[:3]) for p in self.poses]
        self.quats = align_hemispheres([orientation_to_quat(*p[3:]) for p in self.poses])
        n = len(self.poses)
        self.tangents = []
        for i in range(n):
            q_prev = self.quats[max(i - 1, 0)]
            q_next = self.quats[min(i + 1, n - 1)]
            self.tangents.append(squad_tangent(q_prev, self.quats[i], q_next))

    def _point(self, i: int) -> np.ndarray:
        n = len(self.points)
        if i < 0:  # mirrored end points keep the curve natural at the ends
            return 2 * self.points[0] - self.points[1] if n > 1 else self.points[0]
        if i >= n:
            return 2 * self.points[-1] - self.points[-2] if n > 1 else self.points[-1]
        return self.points[i]

    def position(self, segment: int, u: float) -> np.ndarray:
        if len(self.points) == 1:
            return self.points[0].copy()
        i = min(max(segment, 0), len(self.points) - 2)
        return catmull_rom(
            self._point(i - 1), self._point(i), self._point(i + 1), self._point(i + 2), u
        )

    def rotation(self, segment: int, u: float) -> np.ndarray:
        if len(self.quats) == 1:
            return self.quats[0]
        i = min(max(segment, 0), len(self.quats) - 2)
        q1, q2 = self.quats[i], self.quats[i + 1]
        s1, s2 = self.tangents[i], self.tangents[i + 1]
        return slerp(slerp(q1, q2, u), slerp(s1, s2, u), 2 * u * (1 - u))

    def pose(self, segment: int, u: float) -> Pose:
        p = self.position(segment, u)
        heading, pitch, roll = quat_to_orientation(self.rotation(segment, u))
        return (float(p[0]), float(p[1]), float(p[2]), heading, pitch, roll)

    # --- arc length (constant speed) ------------------------------------------------------
    def arc_table(self, samples_per_segment: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """(global parameter g = segment + u, cumulative length) samples."""
        g_list, lengths = [0.0], [0.0]
        prev = self.position(0, 0.0)
        for seg in range(max(1, len(self.points) - 1)):
            for k in range(1, samples_per_segment + 1):
                u = k / samples_per_segment
                p = self.position(seg, u)
                lengths.append(lengths[-1] + float(np.linalg.norm(p - prev)))
                g_list.append(seg + u)
                prev = p
        return np.array(g_list), np.array(lengths)


def camera_path_lines(curve, samples: int = 400):
    """Gizmo polylines for an animated camera curve: the path plus a small frustum per key.

    Returns [(ENU points (n, 3), rgba, closed)] for an outline layer; empty without keys.
    """
    if curve is None or not curve.keys:
        return []
    keys = curve.keys
    lines = []
    if len(keys) > 1:
        t0, t1 = keys[0].time, keys[-1].time
        times = np.linspace(t0, t1, samples)
        pts = np.array([curve.evaluate(t)[:3] for t in times])
        lines.append((pts, (1.0, 0.85, 0.3, 0.9), False))
        size = max(30.0, float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) * 0.02)
    else:
        size = 100.0
    for k in keys:
        x, y, z, heading, pitch, roll = k.value
        m = orientation_matrix(heading, pitch, roll)
        right, forward, up = m[:, 0], m[:, 1], m[:, 2]
        eye = np.array([x, y, z])
        c = eye + forward * size
        corners = [c + right * sx * size * 0.6 + up * sy * size * 0.35 for sx, sy in
                   ((-1, -1), (1, -1), (1, 1), (-1, 1))]  # fmt: skip
        lines.append((np.array(corners), (0.4, 0.8, 1.0, 0.9), True))
        for corner in corners:
            lines.append((np.array([eye, corner]), (0.4, 0.8, 1.0, 0.7), False))
    return lines
