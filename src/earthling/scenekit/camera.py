"""Camera helpers for scene scripts: poses, orbits, smooth camera plans from timed views or
(time, eye, target) shots, chase / map / travel cameras. Shots are lists of (time, eye, target)
in the scene's ENU frame; :func:`key_shots` turns them into camera keys."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from earthling.core.animation import Interp
from earthling.core.easing import gaussian, pchip, pchip_array, smoother
from earthling.core.geo import LocalFrame
from earthling.core.scene import Scene
from earthling.render.camera import Camera
from earthling.render.camera_rig import look_orientation
from earthling.scenekit.journey import Journey, current


def level_roll(eye, target, heading: float, pitch: float) -> float:
    """Roll (degrees) that keeps the true local vertical at ``target`` upright on screen. Far
    from the frame origin (views from space) the ENU "up" is not the local up any more."""
    journey = current()
    if journey is None or not journey.level_horizon:
        return 0.0
    frame = journey.frame
    lat, lon, _ = frame.enu_to_geodetic(np.asarray(target, dtype=np.float64))
    up_t = frame.geodetic_to_enu(lat, lon, 1000.0) - frame.geodetic_to_enu(lat, lon, 0.0)
    up_t = np.asarray(up_t, dtype=np.float64) / np.linalg.norm(up_t)
    f = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    f /= np.linalg.norm(f)
    want = up_t - f * np.dot(up_t, f)
    tilt = float(np.linalg.norm(want))  # sin of the angle between view and vertical
    if tilt < 0.35:  # (nearly) straight down: local north points up on screen
        north = frame.geodetic_to_enu(lat + 0.01, lon, 0.0) - frame.geodetic_to_enu(lat, lon, 0.0)
        north = np.asarray(north, dtype=np.float64)
        north -= f * np.dot(north, f)
        north /= np.linalg.norm(north)
        w = (tilt / 0.35) ** 2
        want = (want / tilt if tilt > 1e-9 else 0.0) * w + north * (1.0 - w)
    want /= np.linalg.norm(want)

    def screen_up(roll: float) -> np.ndarray:
        m = Camera(np.zeros(3), heading, pitch, roll).rotation_matrix()
        return np.array([m[0][1], m[1][1], m[2][1]])  # camera up axis in world space

    base = screen_up(0.0)
    angle = math.degrees(math.atan2(np.dot(np.cross(base, want), f), np.dot(base, want)))
    for sign in (1.0, -1.0):
        if np.dot(screen_up(sign * angle), want) > 0.9999:
            return sign * angle
    return 0.0


def pose(eye, target, roll: float = 0.0) -> tuple:
    """camera.pose value (x, y, z, heading, pitch, roll) looking from eye at target."""
    h, p, r = look_orientation(np.asarray(eye), np.asarray(target), roll)
    r += level_roll(eye, target, h, p)
    return (float(eye[0]), float(eye[1]), float(eye[2]), h, p, r)


def orbit_eye(target, heading: float, pitch: float, distance: float) -> np.ndarray:
    """Eye position looking at ``target`` along heading/pitch (degrees, pitch < 0 looks down)."""
    h, p = math.radians(heading), math.radians(pitch)
    forward = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    return np.asarray(target, dtype=np.float64) - forward * distance


def geo_orbit_eye(hike: Journey, lat: float, lon: float, h: float, heading: float, pitch: float,
                  distance: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(eye, target) in ENU for an orbit around the geodetic point, with heading / pitch in the
    point's own local frame (views from space, far from the frame origin)."""
    target = np.asarray(hike.frame.geodetic_to_enu(lat, lon, h), dtype=np.float64)
    hr, pr = math.radians(heading), math.radians(pitch)
    forward = np.array([math.sin(hr) * math.cos(pr), math.cos(hr) * math.cos(pr), math.sin(pr)])
    offset_ecef = LocalFrame(lat, lon, 0.0).rotation.T @ (-forward * distance)
    eye = target + hike.frame.rotation @ offset_ecef
    return eye, target


def bearing(a, b) -> float:
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return math.degrees(math.atan2(d[0], d[1]))


def heading_of(eye, target) -> float:
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    return math.degrees(math.atan2(d[0], d[1]))


@dataclass
class View:
    """Orbit state: look at ``target`` from ``distance`` along heading/pitch (degrees)."""

    target: tuple
    heading: float
    pitch: float
    distance: float
    roll: float = 0.0


def plan_camera(scene: Scene, hike: Journey | None, views: list[tuple[float, View]],
                step: float = 1.0 / 30.0, clearance: float = 120.0) -> None:  # fmt: skip
    """Sample a smooth camera path through timed views into linear camera keys.

    Headings are unwrapped, distances interpolated in log space. With ``hike`` the eye is
    kept ``clearance`` metres above the terrain."""
    times = [t for t, _ in views]
    heading = np.unwrap(np.radians([v.heading for _, v in views]))
    channels = {
        "tx": [v.target[0] for _, v in views],
        "ty": [v.target[1] for _, v in views],
        "tz": [v.target[2] for _, v in views],
        "h": list(np.degrees(heading)),
        "p": [v.pitch for _, v in views],
        "d": [math.log(v.distance) for _, v in views],
        "r": [v.roll for _, v in views],
    }
    ts = np.arange(times[0], times[-1] + 1e-9, step)
    eyes, targets, rolls = [], [], []
    for t in ts:
        c = {k: pchip(times, ys, t) for k, ys in channels.items()}
        target = np.array([c["tx"], c["ty"], c["tz"]])
        eyes.append(orbit_eye(target, c["h"], c["p"], math.exp(c["d"])))
        targets.append(target)
        rolls.append(c["r"])
    eyes = np.array(eyes)
    if hike is not None:
        # lift the camera above the terrain smoothly: a lift per sample follows every bump
        # of the ground and shakes the view
        lift = np.zeros(len(ts))
        for i, eye in enumerate(eyes):
            ground = hike.ground_z(eye)
            if math.isfinite(ground):
                lift[i] = max(0.0, ground + clearance - eye[2])
        reach = max(1, int(round(1.5 / step)))
        held = np.array([lift[max(0, i - reach) : i + reach + 1].max() for i in range(len(lift))])
        eyes[:, 2] += np.maximum(gaussian(held, 0.6 / step), lift)
    for t, eye, target, roll in zip(ts, eyes, targets, rolls, strict=True):
        scene.animation.set_key("camera.pose", round(float(t), 4), pose(eye, target, roll),
                                Interp.LINEAR)  # fmt: skip


def line_of_sight(hike: Journey, eye, target, samples: int = 24, margin: float = 20.0) -> bool:
    eye, target = np.asarray(eye), np.asarray(target)
    for f in np.linspace(0.08, 0.92, samples):
        p = eye + (target - eye) * f
        z = hike.ground_z(p)
        if math.isfinite(z) and z + margin > p[2]:
            return False
    return True


def follow_shots(hike: Journey, times, head_of, *, distance: float, pitch: float,
                 side: float, window: float = 1200.0, sigma_s: float = 2.0,
                 heading_sigma_s: float = 5.0, clearance: float = 150.0,
                 look_ahead: float = 0.0, companion=None,
                 companion_weight: float = 0.4) -> list[tuple]:  # fmt: skip
    """Soft chase camera: (time, eye, target) for ``times`` (evenly spaced).

    ``head_of(t)`` is the path distance of the hiker at video time t. The target is the
    path smoothed over ``window`` metres and ``sigma_s`` seconds, the camera looks from
    ``side`` degrees off the walking direction (positive: from the right).
    ``companion(centre)``: an optional second point to keep in view (e.g. the nearest point of
    the planned route); the camera aims between both and backs off when they are far apart."""
    times = np.asarray(times, dtype=np.float64)
    dt_s = float(times[1] - times[0])
    centres, headings = [], []
    for t in times:
        d = head_of(t)
        c, direction = hike.smoothed(d + look_ahead, window)
        centres.append(c)
        headings.append(math.atan2(direction[0], direction[1]))
    centres = gaussian(np.array(centres), sigma_s / dt_s)
    headings = gaussian(np.unwrap(headings), heading_sigma_s / dt_s)
    distances = np.full(len(centres), distance)
    if companion is not None:
        others = np.array([companion(c) for c in centres])
        gap = np.linalg.norm((others - centres)[:, :2], axis=1)
        centres = gaussian(centres + (others - centres) * companion_weight, sigma_s / dt_s)
        distances = gaussian(np.maximum(distance, 1.5 * gap), 2.0 * sigma_s / dt_s)
    eyes = []
    for c, h, dist in zip(centres, headings, distances, strict=True):
        cam_heading = math.degrees(h) + side
        p = pitch
        eye = orbit_eye(c, cam_heading, p, dist)
        while not line_of_sight(hike, eye, c) and p > -70.0:
            p -= 4.0
            eye = orbit_eye(c, cam_heading, p, dist)
        ground = hike.ground_z(eye)
        if math.isfinite(ground):
            eye[2] = max(eye[2], ground + clearance)
        eyes.append(eye)
    eyes = np.array(eyes)
    # smooth the corrections, then make sure the smoothed path still clears the ground
    eyes = gaussian(eyes, sigma_s / dt_s)
    for i, eye in enumerate(eyes):
        ground = hike.ground_z(eye)
        if math.isfinite(ground) and eye[2] < ground + clearance * 0.6:
            eyes[i, 2] = ground + clearance * 0.6
    return [(float(t), e, c) for t, e, c in zip(times, eyes, centres, strict=True)]


def map_shots(
    hike: Journey,
    times,
    head_of,
    *,
    distance_of,
    heading_of=None,
    pitch_of=None,
    sigma_s: float = 2.5,
    fit: float = 0.28,
    points_of=None,
    clearance: float = 300.0,
):
    """Calm map camera: (time, eye, target) with a fixed (or slowly keyed) heading, north up by
    default, looking down steeply at a heavily smoothed point near the head.

    ``distance_of(t)``: the wished distance; it grows where the head (and the extra points of
    ``points_of(t)``, e.g. the next POI) would leave the picture: all stay within ``fit`` x the
    distance of the centre. No turns, no chasing: viewers can follow the route like on a map."""
    times = np.asarray(times, dtype=np.float64)
    step = sigma_s / float(times[1] - times[0])
    heads = np.array([hike.path.position_at(head_of(t)) for t in times])
    centres = gaussian(heads, step)
    reach = np.linalg.norm((heads - centres)[:, :2], axis=1)
    if points_of is not None:
        for i, t in enumerate(times):
            for q in points_of(t):
                q = np.asarray(q, dtype=np.float64)
                reach[i] = max(reach[i], float(np.linalg.norm((q - centres[i])[:2])))
    wished = np.array([distance_of(t) for t in times])
    dist = np.maximum(wished, reach / fit)
    # (a running maximum over the smoothing window first: the zoom-out starts before it is
    # needed, not when the head is already at the edge)
    window = max(1, int(step))
    padded = np.pad(dist, window, mode="edge")
    dist = np.array([padded[i : i + 2 * window + 1].max() for i in range(len(dist))])
    dist = np.exp(gaussian(np.log(dist), step))
    shots = []
    for t, c, d in zip(times, centres, dist, strict=True):
        lat, lon, h = hike.frame.enu_to_geodetic(c)  # (h: the ENU z is not a height far away)
        heading = heading_of(t) if heading_of is not None else 0.0
        pitch = pitch_of(t) if pitch_of is not None else -60.0
        eye, target = geo_orbit_eye(hike, float(lat), float(lon), float(h), heading, pitch,
                                    float(d))  # fmt: skip
        ground = hike.ground_z(eye)
        if math.isfinite(ground):
            eye[2] = max(eye[2], ground + clearance)
        shots.append((float(t), eye, target))
    return shots


def blend_shots(first, second, t_a: float, t_b: float):
    """Shots of ``first`` until t_a, of ``second`` from t_b, smoothly mixed in between (both
    lists sampled at the same times)."""
    out = []
    for (t, e1, g1), (_, e2, g2) in zip(first, second, strict=True):
        w = smoother((t - t_a) / max(t_b - t_a, 1e-6))
        out.append((t, e1 + (e2 - e1) * w, g1 + (g2 - g1) * w))
    return out


def still_shots(times, eye, target):
    return [(float(t), np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64))
            for t in times]  # fmt: skip


KEY_STEP = 1.0 / 30.0  # camera keys every 2 frames: the centripetal spline between keys adds
# no wobble of its own (with sparse keys its speed jumped at every key during fast zooms)


def key_shots(scene: Scene, shots, every: int | None = None) -> None:
    """Camera keys from (time, eye, target) samples, resampled smoothly to KEY_STEP."""
    ts = np.array([t for t, _, _ in shots])
    eyes = np.array([e for _, e, _ in shots])
    targets = np.array([g for _, _, g in shots])
    fine = np.arange(ts[0], ts[-1] + 1e-9, KEY_STEP)
    if fine[-1] < ts[-1] - 1e-6:
        fine = np.append(fine, ts[-1])
    for t, eye, target in zip(fine, pchip_array(ts, eyes, fine), pchip_array(ts, targets, fine),
                              strict=True):  # fmt: skip
        scene.animation.set_key("camera.pose", round(float(t), 4), pose(eye, target), Interp.LINEAR)


def plan_look(scene: Scene, hike: Journey | None, points, step: float = 0.25,
              clearance: float = 150.0) -> list:  # fmt: skip
    """Smooth camera through timed (t, eye, target) control points (PCHIP on both)."""
    times = [p[0] for p in points]
    eyes = np.array([p[1] for p in points], dtype=np.float64)
    targets = np.array([p[2] for p in points], dtype=np.float64)
    shots = []
    t = times[0]
    while t <= times[-1] + 1e-9:
        eye = np.array([pchip(times, eyes[:, k], t) for k in range(3)])
        target = np.array([pchip(times, targets[:, k], t) for k in range(3)])
        if hike is not None:
            ground = hike.ground_z(eye)
            if math.isfinite(ground) and eye[2] < ground + clearance:
                eye[2] = ground + clearance
        shots.append((t, eye, target))
        t += step
    key_shots(scene, shots)
    return shots
