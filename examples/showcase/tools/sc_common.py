"""Shared helpers for the showcase scenes (adapted from the helpers of the Thailand and
Hexatrek projects): load a scene project, smooth camera plans, POIs, holds.

Run the scene scripts with ``python -m uv run python examples/showcase/tools/<script>.py``
from the repository; each writes ``<project>/scenes/<name>.json``.
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earthling.core.animation import Interp
from earthling.core.config import Project
from earthling.core.scene import Scene
from earthling.core.session import Session
from earthling.render.camera_rig import look_orientation

TOOLS = Path(__file__).resolve().parent
SHOWCASE = TOOLS.parent
REPO = SHOWCASE.parent.parent
RENDERS = REPO / "renders" / "showcase"
PROJECT = SHOWCASE  # set by load_hike: the scene project folder
sys.path.insert(0, str(TOOLS))
TZ = dt.UTC  # set by load_hike: the project's time zone


@dataclass
class Hike:
    session: Session
    path: object  # earthling.render.tracks.ProgressPath (walked tracks)
    gpu: list  # all built tracks (walked and planned)
    data: object  # TerrainData

    @property
    def frame(self):
        return self.session.frame

    # --- coordinates ---------------------------------------------------------------------
    def enu(self, lat: float, lon: float, h: float = 0.0) -> np.ndarray:
        return np.asarray(self.frame.geodetic_to_enu(lat, lon, h), dtype=np.float64)

    def lonlat(self, p) -> tuple[float, float]:
        lat, lon, _ = self.frame.enu_to_geodetic(np.asarray(p, dtype=np.float64))
        return float(lon), float(lat)

    def ground_z(self, p) -> float:
        """ENU z of the terrain below the ENU point ``p`` (NaN without DEM data)."""
        lon, lat = self.lonlat(p)
        h = self.data.height_at(lon, lat)
        if h is None:
            return float("nan")
        return float(self.enu(lat, lon, h)[2])

    def max_ground_z(self, a, b, samples: int = 40, radius: float = 0.0) -> float:
        """Highest terrain along the segment a-b (for clearance checks)."""
        best = -1e9
        for f in np.linspace(0.0, 1.0, samples):
            p = np.asarray(a) + (np.asarray(b) - np.asarray(a)) * f
            for dx, dy in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)):
                z = self.ground_z(p + np.array([dx, dy, 0.0]))
                if math.isfinite(z):
                    best = max(best, z)
        return best

    # --- tracks (the journey: walked tracks, bus rides, ferries and flights in time order) --
    def day_index(self, day) -> int:
        """Index on the journey path of track ``day``: its number (1 = the first in time) or
        its file stem."""
        if isinstance(day, int):
            return day - 1
        for i, g in enumerate(self.path.tracks):
            if g.track.source.stem == day:
                return i
        raise KeyError(day)

    def day(self, day):
        return self.path.tracks[self.day_index(day)]

    def day_name(self, day) -> str:
        return self.day(day).track.name

    def day_times(self, day) -> tuple[float, float]:
        t = self.day(day).time
        t = t[np.isfinite(t)]
        return float(t[0]), float(t[-1])

    def start(self, day) -> np.ndarray:
        g = self.day(day)
        return g.enu[0].copy()

    def end(self, day) -> np.ndarray:
        g = self.day(day)
        return g.enu[-1].copy()

    def head_at_time(self, epoch: float) -> float:
        """progress.head (mode "time") for a UTC epoch (clamped)."""
        return min(max(self.time_progress(epoch), 0.0), 1.0)

    def time_progress(self, seconds: float) -> float:
        """progress.head (mode "time") for a UTC epoch."""
        t = self.path.time[np.isfinite(self.path.time)]
        return float((seconds - t[0]) / (t[-1] - t[0]))

    def distance_progress(self, distance: float) -> float:
        return float(distance / self.path.total_m)

    def smoothed(self, distance: float, window: float) -> tuple[np.ndarray, np.ndarray]:
        """Mean position and direction of the walked path around ``distance``."""
        ds = np.linspace(distance - window, distance + window, 41)
        ds = np.clip(ds, 0.0, self.path.total_m)
        pts = np.array([self.path.position_at(d) for d in ds])
        weights = np.hanning(len(ds) + 2)[1:-1]
        centre = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        direction = pts[-1] - pts[0]
        direction[2] = 0.0
        n = float(np.linalg.norm(direction))
        direction = direction / n if n > 1e-3 else np.array([0.0, 1.0, 0.0])
        return centre, direction


def load_hike(project: str, origin: tuple[float, float] | None = None) -> Hike:
    """Scene project ``project`` (folder name in examples/showcase), with the local frame centred
    on ``origin`` (lat, lon) when given (the scene gets the same origin: :func:`new_scene`)."""
    import zoneinfo

    import moderngl

    global TZ, PROJECT

    from earthling.render.renderer import Renderer
    from earthling.render.tracks import TrackGeometryOptions

    PROJECT = SHOWCASE / project
    session = Session(Project.load(PROJECT))
    TZ = zoneinfo.ZoneInfo(session.config.project.timezone)
    session.set_origin(origin)
    ctx = moderngl.create_standalone_context(require=430)
    renderer = Renderer(ctx)
    renderer.set_scene(session.frame, session.all_tracks)
    data = session.terrain_data()
    renderer.tracks.heights_at = data.heights_at
    renderer.tracks.ensure_built(
        TrackGeometryOptions(elevation_source="dem", height_offset_m=3.0, exaggeration=1.0,
                             smoothing=5), dem_version=id(data))  # fmt: skip
    global _FRAME, _HIKE
    _FRAME = session.frame
    _HIKE = Hike(session, renderer.tracks.path, list(renderer.tracks._gpu), data)
    return _HIKE


# --- scenes ------------------------------------------------------------------------------
def new_scene(hike: Hike, duration: float, fps: float = 60.0) -> Scene:
    scene = Scene()
    frame = hike.frame
    scene.origin = (frame.lat, frame.lon)
    scene.set_dynamic("track_groups", hike.session.track_group_properties())
    scene.animation.duration = duration
    scene.animation.fps = fps
    s = scene.store
    s.set("export.resolution", "1920x1080")
    s.set("export.samples", "4")
    s.set("stats.language", "en")
    s.set("labels.visible", False)
    s.set("stats.visible", False)
    s.set("camera.mode", "keys")
    s.set("marker.glow", 1.2)
    s.set("marker.size", 10.0)
    return scene


_FRAME = None  # set by load_hike: the local frame (for the level horizon in pose())


def level_roll(eye, target, heading: float, pitch: float) -> float:
    """Roll (degrees) that keeps the true local vertical at ``target`` upright on screen. Far
    from the frame origin (views from space) the ENU "up" is not the local up any more."""
    if _FRAME is None:
        return 0.0
    from earthling.render.camera import Camera

    lat, lon, _ = _FRAME.enu_to_geodetic(np.asarray(target, dtype=np.float64))
    up_t = _FRAME.geodetic_to_enu(lat, lon, 1000.0) - _FRAME.geodetic_to_enu(lat, lon, 0.0)
    up_t = np.asarray(up_t, dtype=np.float64) / np.linalg.norm(up_t)
    f = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    f /= np.linalg.norm(f)
    want = up_t - f * np.dot(up_t, f)
    tilt = float(np.linalg.norm(want))  # sin of the angle between view and vertical
    if tilt < 0.35:  # (nearly) straight down: local north points up on screen
        north = _FRAME.geodetic_to_enu(lat + 0.01, lon, 0.0) - _FRAME.geodetic_to_enu(lat, lon, 0.0)
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
    h, p, r = look_orientation(np.asarray(eye), np.asarray(target), roll)
    r += level_roll(eye, target, h, p)
    return (float(eye[0]), float(eye[1]), float(eye[2]), h, p, r)


def orbit_eye(target, heading: float, pitch: float, distance: float) -> np.ndarray:
    """Eye position looking at ``target`` along heading/pitch (degrees, pitch < 0 looks down)."""
    h, p = math.radians(heading), math.radians(pitch)
    forward = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    return np.asarray(target, dtype=np.float64) - forward * distance


def key(scene: Scene, pid: str, time: float, value, interp: Interp | None = Interp.EASE_IN_OUT):
    scene.animation.set_key(pid, time, value, interp)


def local(epoch: float) -> dt.datetime:
    """Naive local datetime (project time zone) for sun.datetime."""
    return dt.datetime.fromtimestamp(epoch, TZ).replace(tzinfo=None)


HOLD_S = 1.0  # V2: a still second at both ends of every video
HOLD_YAW = 1.2  # degrees of slow rotation during a hold


def add_holds(scene: Scene, seconds: float = HOLD_S, yaw: float = HOLD_YAW) -> None:
    """Shift the whole animation by ``seconds`` and add a still second at both ends: the
    camera stays where it starts / ends and only turns very slowly (eased into the motion)."""
    anim = scene.animation
    old = anim.duration
    for curve in anim.curves.values():
        for k in curve.keys:
            if k.time >= 0.0:  # (keys before the timeline stay: "visible from the start")
                k.time = round(k.time + seconds, 4)
    anim.duration = old + 2.0 * seconds
    cam = anim.curves.get("camera.pose")
    if cam is None or not cam.keys:
        return
    first, last = cam.keys[0].value, cam.keys[-1].value
    t_first, t_last = cam.keys[0].time, cam.keys[-1].time
    n = max(2, int(round(seconds * 30)))
    for i in range(n):
        u = i / n
        x, y, z, h, pt, r = first
        key(scene, "camera.pose", round(t_first - seconds + u * seconds, 4),
            (x, y, z, h - yaw * (1.0 - u) ** 2, pt, r), Interp.LINEAR)  # fmt: skip
    for i in range(1, n + 1):
        u = i / n
        x, y, z, h, pt, r = last
        key(scene, "camera.pose", round(t_last + u * seconds, 4),
            (x, y, z, h + yaw * u * u, pt, r), Interp.LINEAR)  # fmt: skip


def save(scene: Scene, name: str, holds: bool = False) -> Path:
    """Write ``<project>/scenes/<name>.json`` (set POIs with scene.set_pois before keying their
    properties). With ``holds`` a still second is added at both ends (see add_holds)."""
    if holds:
        add_holds(scene)
    path = scene.save(PROJECT / "scenes" / f"{name}.json")
    print(f"wrote {path}")
    return path


# --- smooth camera plans ---------------------------------------------------------------------
def pchip(xs, ys, x: float) -> float:
    """Monotone cubic (Fritsch-Carlson) interpolation: C1 smooth, no overshoot."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.zeros_like(ys)
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # ends: zero slope (the camera eases in and out)
    i = int(np.searchsorted(xs, x) - 1)
    t = (x - xs[i]) / h[i]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return float(h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])


@dataclass
class View:
    """Orbit state: look at ``target`` from ``distance`` along heading/pitch (degrees)."""

    target: tuple
    heading: float
    pitch: float
    distance: float
    roll: float = 0.0


def plan_camera(scene: Scene, hike: Hike | None, views: list[tuple[float, View]],
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
        held = np.array([lift[max(0, i - reach): i + reach + 1].max() for i in range(len(lift))])
        eyes[:, 2] += np.maximum(gaussian(held, 0.6 / step), lift)
    for t, eye, target, roll in zip(ts, eyes, targets, rolls, strict=True):
        scene.animation.set_key("camera.pose", round(float(t), 4), pose(eye, target, roll),
                                Interp.LINEAR)  # fmt: skip


def lerp(a, b, f):
    return tuple(float(x + (y - x) * f) for x, y in zip(a, b, strict=True))


def gaussian(values, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 (edges padded with the end values)."""
    values = np.asarray(values, dtype=np.float64)
    if sigma_samples <= 0:
        return values
    radius = int(3 * sigma_samples) + 1
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    kernel /= kernel.sum()
    pad = [(radius, radius)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pad, mode="edge")
    out = np.empty_like(values)
    for i in range(len(values)):
        window = padded[i : i + 2 * radius + 1]
        out[i] = np.tensordot(kernel, window, axes=(0, 0))
    return out


def line_of_sight(hike: Hike, eye, target, samples: int = 24, margin: float = 20.0) -> bool:
    eye, target = np.asarray(eye), np.asarray(target)
    for f in np.linspace(0.08, 0.92, samples):
        p = eye + (target - eye) * f
        z = hike.ground_z(p)
        if math.isfinite(z) and z + margin > p[2]:
            return False
    return True


def follow_shots(hike: Hike, times, head_of, *, distance: float, pitch: float,
                 side: float, window: float = 1200.0, sigma_s: float = 2.0,
                 heading_sigma_s: float = 5.0, clearance: float = 150.0,
                 look_ahead: float = 0.0, companion=None,
                 companion_weight: float = 0.4) -> list[tuple[float, np.ndarray, np.ndarray]]:  # fmt: skip
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


def pchip_array(xs, ys, xq) -> np.ndarray:
    """Vectorised monotone cubic interpolation; ys (n, ...) sampled at xs, queried at xq."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    flat = ys.reshape(len(xs), -1)
    h = np.diff(xs)
    delta = np.diff(flat, axis=0) / h[:, None]
    m = np.zeros_like(flat)
    w1 = (2 * h[1:] + h[:-1])[:, None]
    w2 = (h[1:] + 2 * h[:-1])[:, None]
    same = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    m[1:-1] = np.where(same, harmonic, 0.0)
    xq = np.clip(np.asarray(xq, dtype=np.float64), xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xq) - 1, 0, len(xs) - 2)
    t = ((xq - xs[i]) / h[i])[:, None]
    hi = h[i][:, None]
    out = ((2 * t**3 - 3 * t**2 + 1) * flat[i] + (t**3 - 2 * t**2 + t) * hi * m[i]
           + (-2 * t**3 + 3 * t**2) * flat[i + 1] + (t**3 - t**2) * hi * m[i + 1])  # fmt: skip
    return out.reshape((len(xq),) + ys.shape[1:])


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


def plan_look(scene: Scene, hike: Hike | None, points, step: float = 0.25,
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


def nearest_on(points: np.ndarray, max_gap: float = 3000.0):
    """companion() for follow_shots: the nearest of ``points`` (ENU), or the centre itself when
    nothing is within ``max_gap`` metres."""
    pts = np.asarray(points, dtype=np.float64)

    def companion(centre):
        d = np.linalg.norm(pts[:, :2] - centre[:2], axis=1)
        i = int(np.argmin(d))
        return pts[i] if d[i] <= max_gap else np.asarray(centre)

    return companion


def stops(hike: Hike, day: int, min_minutes: float = 4.0, radius_m: float = 40.0):
    """Breaks of a day: [(start epoch, end epoch, ENU position)] where we stayed within
    ``radius_m`` for at least ``min_minutes``."""
    g = hike.day(day)
    ok = np.isfinite(g.time)
    t, p = g.time[ok], g.enu[ok]
    out, i = [], 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and np.linalg.norm((p[j + 1] - p[i])[:2]) < radius_m:
            j += 1
        if t[j] - t[i] >= min_minutes * 60.0:
            out.append((float(t[i]), float(t[j]), p[i:j + 1].mean(axis=0)))
            i = j + 1
        else:
            i += 1
    return out


def stop_near(hike: Hike, day: int, epoch: float, **kw):
    """The break closest in time to ``epoch``."""
    best = None
    for start, end, pos in stops(hike, day, **kw):
        gap = 0.0 if start <= epoch <= end else min(abs(epoch - start), abs(epoch - end))
        if best is None or gap < best[0]:
            best = (gap, start, end, pos)
    return best[1:] if best else None


def epoch_at(hike: Hike, day: int, clock: str) -> float:
    """UTC epoch of a local clock time ("HH:MM") on a hike day."""
    date = local(hike.day_times(day)[0]).date()
    h, m = (int(x) for x in clock.split(":"))
    return dt.datetime(date.year, date.month, date.day, h, m, tzinfo=TZ).timestamp()


def distance_near(hike: Hike, lon: float, lat: float, days) -> float:
    """Path distance of the walked point closest to lon/lat within the given days."""
    target = hike.enu(lat, lon, 0.0)[:2]
    best = (math.inf, 0.0)
    for day in days:
        d0, d1 = hike.path.track_range(hike.day_index(day))  # day: track number or stem
        for d in np.arange(d0, d1, 10.0):
            gap = float(np.linalg.norm(hike.path.position_at(d)[:2] - target))
            if gap < best[0]:
                best = (gap, float(d))
    return best[1]


_HIKE = None


def poi(pid: str, where, icon: str, size: float = 60.0, caption: str = "", hike=None):
    """A symbol at a (lat, lon) tuple or an ENU point (numpy array)."""
    from earthling.core.pois import Poi

    if isinstance(where, tuple):
        lat, lon = where
    else:
        lon, lat = (hike or _HIKE).lonlat(where)
    return Poi(id=pid, name=pid, lon=float(lon), lat=float(lat), icon=f"builtin:{icon}",
               size_px=size, caption=caption, lift_px=16.0)  # fmt: skip


def show(scene: Scene, pid: str, t_in: float | None, t_out: float | None = None) -> None:
    """Visibility keys: hidden before ``t_in`` (None: from the start), hidden from ``t_out``."""
    # (already there at the start: switched on before the timeline, no pop-in in frame 0)
    key(scene, f"poi.{pid}.visible", -1.0 if t_in is None else 0.0, t_in is None or t_in <= 0.0,
        Interp.STEP)  # fmt: skip
    if t_in is not None and t_in > 0.0:
        key(scene, f"poi.{pid}.visible", t_in, True, Interp.STEP)
    if t_out is not None:
        key(scene, f"poi.{pid}.visible", t_out, False, Interp.STEP)
    scene.store.set(f"poi.{pid}.caption", False)
    scene.store.set(f"poi.{pid}.pin", True)


def smoother_soft(u: float) -> float:
    """Mostly linear with eased ends (the head starts and stops softly)."""
    u = min(max(u, 0.0), 1.0)
    return 0.7 * u + 0.3 * u * u * (3.0 - 2.0 * u)


def smoother(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def key_head(scene: Scene, hike: Hike, times, epochs) -> None:
    """progress.head keys (linear) for the journey times ``epochs`` (UTC) at ``times``."""
    for t, e in zip(times, epochs, strict=True):
        key(scene, "progress.head", round(float(t), 4), hike.head_at_time(float(e)), Interp.LINEAR)


def bearing(a, b) -> float:
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return math.degrees(math.atan2(d[0], d[1]))


def travel_scene(scene: Scene, hike: Hike, stem, *, duration: float = 7.0,
                 t0: float = 1.0, t1: float | None = None, near: float = 6000.0,
                 far: float | None = None, pitch_near: float = -38.0, pitch_far: float = -58.0,
                 side: float = -35.0, a=None, b=None, near_b: float | None = None,
                 pitch_b: float | None = None, clearance: float = 400.0,
                 lead: float = 0.35, offset: float = 0.0, heading_a: float | None = None,
                 heading_b: float | None = None, apply: bool = True,
                 camera: bool = True, keep: float | None = None):  # fmt: skip
    """A simple flight along travel line(s) ``stem``: hold on the start until ``t0``, the
    dotted line grows (in its real time) until ``t1`` while the camera flies along, hold on
    the end until ``duration``. All times are shifted by ``offset`` on the timeline.

    The camera looks across the line (``side`` degrees off the direction of travel; or from
    ``heading_a`` to ``heading_b``), zooms out to ``far`` (default: 70 % of the distance)
    halfway and back in. ``a``/``b``: ENU look points at the start / end (default: the line's
    ends). ``keep`` (metres): the last ``keep`` metres of the drawn line stay in frame (the
    camera aims at their middle and backs off far enough). Returns (start epoch, end epoch,
    shots); with ``apply`` the head keys are set, and the camera keys too unless ``camera`` is
    False."""
    t1 = duration - 1.0 if t1 is None else t1
    stems = [stem] if isinstance(stem, str) else list(stem)  # several lines one after another
    e0, e1 = hike.day_times(stems[0])[0], hike.day_times(stems[-1])[1]
    start = np.asarray(a if a is not None else hike.start(stems[0]), dtype=np.float64)
    end = np.asarray(b if b is not None else hike.end(stems[-1]), dtype=np.float64)
    length = float(np.linalg.norm((end - start)[:2]))
    far = far if far is not None else max(near * 1.5, 0.7 * length)
    near_b = near if near_b is None else near_b
    pitch_b = pitch_near if pitch_b is None else pitch_b
    base = bearing(start, end) + 180.0 + side  # behind and beside the direction of travel
    h_a = base - 10.0 if heading_a is None else heading_a
    h_b = base + 10.0 if heading_b is None else heading_b
    h_b = h_a + ((h_b - h_a + 180.0) % 360.0 - 180.0)  # the short way round

    def u_of(t: float) -> float:
        return smoother((t - t0) / max(t1 - t0, 1e-6))

    step = 1.0 / 30.0
    times = np.arange(0.0, duration + 1e-9, step)
    epochs = [e0 + u_of(t) * (e1 - e0) for t in times]
    if apply:
        key_head(scene, hike, times[::2] + offset, epochs[::2])
    # the look point leads the head a little, ending exactly on b
    heads = []
    for t in times:
        u = u_of(t)
        d = hike.path.distance_for_time(e0 + min(1.0, u + lead * u * (1 - u)) * (e1 - e0))
        heads.append(hike.path.position_at(d))
    heads = gaussian(np.array(heads), 0.35 / step)
    d_line0 = hike.path.distance_for_time(e0)
    if keep is not None:
        tails, spans = [], []
        for t in times:
            d = hike.path.distance_for_time(e0 + u_of(t) * (e1 - e0))
            tail = hike.path.position_at(max(d_line0, d - keep))
            head = hike.path.position_at(d)
            tails.append((tail + head) / 2.0)
            spans.append(float(np.linalg.norm((head - tail)[:2])))
        mids = gaussian(np.array(tails), 0.5 / step)
        spans = gaussian(np.array(spans), 0.5 / step)
    shots = []
    for i, t in enumerate(times):
        u = u_of(t)
        w = smoother(min(1.0, max(0.0, (t - t0) / 1.2)))  # leave a / arrive at b softly
        w_end = smoother(min(1.0, max(0.0, (t - t1 + 1.2) / 1.2)))  # (still during the holds)
        look = heads[i] if keep is None else mids[i]
        target = start + (look - start) * w
        target = target + (end - target) * w_end
        bump = math.sin(math.pi * u) ** 1.5
        dist0 = math.exp(math.log(near) + (math.log(near_b) - math.log(near)) * u)
        dist = math.exp(math.log(dist0) + (math.log(far) - math.log(dist0)) * bump)
        if keep is not None:  # the drawn stretch must fit into the picture
            dist = dist + max(0.0, 1.3 * spans[i] - dist) * w * (1.0 - w_end)
        pitch0 = pitch_near + (pitch_b - pitch_near) * u
        pitch = pitch0 + (pitch_far - pitch0) * bump
        eye = orbit_eye(target, h_a + (h_b - h_a) * u, pitch, dist)
        ground = hike.ground_z(eye)
        if math.isfinite(ground):
            eye[2] = max(eye[2], ground + clearance)
        shots.append((float(t + offset), eye, target))
    if apply:  # less aerial perspective high up (milky from 50+ km)
        for t in np.arange(0.0, duration + 1e-9, 0.5):
            bump = math.sin(math.pi * u_of(t)) ** 1.5
            key(scene, "haze.aerial", round(float(t + offset), 4), 0.6 - 0.3 * bump, Interp.LINEAR)
    if apply and camera:
        key_shots(scene, shots)
    return e0, e1, shots


def night_light(scene: Scene, t_from: float, t_to: float, ambient: float = 3.5,
                exposure: float = 1.4) -> None:  # fmt: skip
    """A readable night (eyes adapted): ambient and exposure rise between t_from and t_to
    (both 0: from the start)."""
    for pid, value, base in (("light.night_ambient", ambient, 1.0), ("post.exposure", exposure, 0.0)):
        if t_to <= 0.0:
            key(scene, pid, 0.0, value)
            continue
        key(scene, pid, 0.0, base)
        key(scene, pid, t_from, base)
        key(scene, pid, t_to, value)


def head_keys_for(scene: Scene, hike: Hike, day, t_from: float, t_to: float, *,
                  realtime: bool, step: float = 1.0 / 15.0, d_end: float | None = None,
                  slow: tuple[float, float, float] | None = None) -> callable:  # fmt: skip
    """Draw walked track ``day`` between timeline times ``t_from`` and ``t_to``: in its real
    timestamps (``realtime``, breaks pause the line) or evenly by distance. ``d_end``: stop at
    this path distance (the rest of the track is not drawn). ``slow`` = (d_a, d_b, factor):
    distance mode only, the head is ``factor`` times slower between the path distances d_a and
    d_b. Returns head_of(t) -> path distance (for the chase camera)."""
    e0, e1 = hike.day_times(day)
    d0, d1 = hike.path.track_range(hike.day_index(day))
    if d_end is not None:
        d1 = min(d1, d_end)
        e1 = hike.path.time_at(d1)
    ds = np.linspace(d0, d1, 2001)
    cost = np.ones_like(ds)
    if slow is not None:
        a, b, factor = slow
        inside = (ds >= a) & (ds <= b)
        cost[inside] = factor
        cost = gaussian(cost, 15.0)  # no sudden change of speed
    cum = np.concatenate([[0.0], np.cumsum((cost[1:] + cost[:-1]) / 2.0)])
    cum /= cum[-1]

    def head_of(t: float) -> float:
        f = smoother_soft(min(max((t - t_from) / (t_to - t_from), 0.0), 1.0))
        if realtime:
            return hike.path.distance_for_time(e0 + f * (e1 - e0))
        return float(np.interp(f, cum, ds))

    for t in np.arange(t_from, t_to + 1e-9, step):
        d = min(max(head_of(t), d0 + 0.01), d1 - 0.01)
        key(scene, "progress.head", round(float(t), 4),
            hike.path.progress_for_distance(d, "time"), Interp.LINEAR)  # fmt: skip
    return head_of


def geo_orbit_eye(hike: Hike, lat: float, lon: float, h: float, heading: float, pitch: float,
                  distance: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(eye, target) in ENU for an orbit around the geodetic point, with heading / pitch in the
    point's own local frame (views from space, far from the frame origin)."""
    from earthling.core.geo import LocalFrame

    target = np.asarray(hike.frame.geodetic_to_enu(lat, lon, h), dtype=np.float64)
    hr, pr = math.radians(heading), math.radians(pitch)
    forward = np.array([math.sin(hr) * math.cos(pr), math.cos(hr) * math.cos(pr), math.sin(pr)])
    offset_ecef = LocalFrame(lat, lon, 0.0).rotation.T @ (-forward * distance)
    eye = target + hike.frame.rotation @ offset_ecef
    return eye, target


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


def clamp_sun(scene: Scene, hike: Hike, head_of, times, latest: str | None = None,
              earliest: str | None = None, day=None) -> None:  # fmt: skip
    """sun.datetime keys from the GPX time at the head, kept between ``earliest`` and
    ``latest`` local clock times ("HH:MM") - daylight for tracks that end after dark.
    ``day``: the track drawn (its end must not read the time of the next track there)."""
    scene.store.set("sun.follow_track", False)
    lo, hi = hike.path.track_range(hike.day_index(day)) if day is not None else (0.0, 1e18)
    for t in times:
        e = hike.path.time_at(min(max(head_of(t), lo + 1.0), hi - 1.0))
        when = local(e)
        if latest is not None:
            h, m = (int(x) for x in latest.split(":"))
            when = min(when, when.replace(hour=h, minute=m, second=0))
        if earliest is not None:
            h, m = (int(x) for x in earliest.split(":"))
            when = max(when, when.replace(hour=h, minute=m, second=0))
        key(scene, "sun.datetime", round(float(t), 4), when, Interp.LINEAR)


def track_point_near(hike: Hike, day, key_or_enu) -> tuple[np.ndarray, float]:
    """(ENU point, path distance) of walked track ``day`` closest to an ENU point."""
    g = hike.day(day)
    p = np.asarray(key_or_enu)
    i = int(np.argmin(np.linalg.norm((g.enu - p)[:, :2], axis=1)))
    return g.enu[i].copy(), float(g.offset_m + g.dist[i])


def track_highest(hike: Hike, day) -> tuple[np.ndarray, float]:
    """(ENU point, path distance) of the highest point of walked track ``day``."""
    g = hike.day(day)
    i = int(np.nanargmax(g.ele))
    return g.enu[i].copy(), float(g.offset_m + g.dist[i])


def reached_at(head_of, distance: float, t_from: float, t_to: float) -> float:
    """Timeline time when head_of(t) first reaches ``distance``."""
    for t in np.arange(t_from, t_to + 1e-9, 1.0 / 60.0):
        if head_of(t) >= distance:
            return round(float(t), 3)
    return t_to


def travel_time_of(epoch: float, e0: float, e1: float, t0: float, t1: float) -> float:
    """Timeline time when a travel_scene line (t0..t1, smoother) reaches ``epoch``."""
    f = min(max((epoch - e0) / max(e1 - e0, 1e-9), 0.0), 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if smoother(mid) < f:
            lo = mid
        else:
            hi = mid
    return round(t0 + lo * (t1 - t0), 3)


def first_reach(hike: Hike, day, point, radius_m: float = 60.0) -> float:
    """Path distance where walked track ``day`` first comes within ``radius_m`` of ``point``."""
    g = hike.day(day)
    gap = np.linalg.norm((g.enu - np.asarray(point))[:, :2], axis=1)
    i = int(np.argmax(gap < radius_m)) if (gap < radius_m).any() else len(gap) - 1
    return float(g.offset_m + g.dist[i])


def heading_of(eye, target) -> float:
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    return math.degrees(math.atan2(d[0], d[1]))


def daylight_keys(scene: Scene, hike: Hike, times, head_of, sigma_s: float = 0.8) -> None:
    """sun.datetime keys from the recorded time at the head (``head_of(t)``: path distance),
    for heads that jump over nights and stops. The clock time is smoothed over ``sigma_s`` on
    the date of the head: from one day's evening to the next morning the sun moves back over
    the sky instead of passing through the night (no dark flash between the days)."""
    times = np.asarray(times, dtype=np.float64)
    whens = [local(hike.path.time_at(head_of(float(t)))) for t in times]
    hours = np.array([w.hour + w.minute / 60.0 + w.second / 3600.0 for w in whens])
    step = float(times[1] - times[0]) if len(times) > 1 else 1.0
    hours = gaussian(hours, sigma_s / step)
    scene.store.set("sun.follow_track", False)
    for t, w, h in zip(times, whens, hours, strict=True):
        clock = dt.datetime(w.year, w.month, w.day) + dt.timedelta(hours=float(h))
        key(scene, "sun.datetime", round(float(t), 4), clock, Interp.LINEAR)
