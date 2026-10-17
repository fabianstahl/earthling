"""Scene helpers for scene scripts: a new scene, keys, POIs, still seconds at both ends,
saving, the drawn head of the journey and the light."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from earthling.core.animation import Interp
from earthling.core.easing import gaussian, smoother, smoother_soft
from earthling.core.pois import Poi
from earthling.core.scene import Scene
from earthling.render.camera_rig import look_orientation
from earthling.scenekit.camera import bearing, key_shots, orbit_eye
from earthling.scenekit.journey import Journey, current, local, track_time_at


def new_scene(hike: Journey, duration: float, fps: float = 60.0) -> Scene:
    """An empty scene of the journey's project with the kit's defaults (1080p, 4 samples, no
    labels or stats, keyed camera); its origin is the journey's frame origin."""
    scene = hike.session.new_scene()
    frame = hike.frame
    scene.origin = (frame.lat, frame.lon)
    scene.animation.duration = duration
    scene.animation.fps = fps
    s = scene.store
    s.set("export.resolution", "1920x1080")
    s.set("export.samples", "4")
    s.set("labels.visible", False)
    s.set("stats.visible", False)
    s.set("camera.mode", "keys")
    s.set("marker.glow", 1.2)
    s.set("marker.size", 10.0)
    return scene


def key(scene: Scene, pid: str, time: float, value, interp: Interp | None = Interp.EASE_IN_OUT):
    """A keyframe (eased by default)."""
    scene.animation.set_key(pid, time, value, interp)


HOLD_S = 1.0  # a still second at both ends of every video


def _ease_warp(x: float, length: float, ramp: float) -> float:
    """Time warp of the camera motion: speed 0 at both ends, ramping up over ``ramp`` seconds
    (the same total duration; no jolt where the still seconds meet the motion)."""
    if length <= 2.0 * ramp:
        return length * smoother(x / max(length, 1e-9))
    if x < ramp:
        u = x / ramp
        return ramp * u * u * (2.0 - u)
    if x > length - ramp:
        u = (length - x) / ramp
        return length - ramp * u * u * (2.0 - u)
    return x


def add_holds(scene: Scene, seconds: float = HOLD_S, ramp: float = 1.0) -> None:
    """Shift the whole animation by ``seconds`` and add a still second at both ends: the camera
    keeps exactly its first / last pose (V3: no turning at all) and its motion starts and ends
    with speed zero (time warp over ``ramp`` seconds)."""
    anim = scene.animation
    old = anim.duration
    cam = anim.curves.get("camera.pose")
    if cam is not None and len(cam.keys) > 1:
        t_first, t_last = cam.keys[0].time, cam.keys[-1].time
        length = t_last - t_first
        ts = np.arange(0.0, length + 1e-9, 1.0 / 60.0)
        poses = [cam.evaluate(t_first + _ease_warp(float(x), length, min(ramp, length / 3)))
                 for x in ts]  # fmt: skip
        cam.keys.clear()
        for x, pose_ in zip(ts, poses, strict=True):
            key(scene, "camera.pose", round(t_first + float(x), 4), tuple(pose_), Interp.LINEAR)
    for curve in anim.curves.values():
        for k in curve.keys:
            if k.time >= 0.0:  # (keys before the timeline stay: "visible from the start")
                k.time = round(k.time + seconds, 4)
    anim.duration = old + 2.0 * seconds
    if cam is None or not cam.keys:
        return
    first, last = cam.keys[0], cam.keys[-1]
    key(scene, "camera.pose", 0.0, tuple(first.value), Interp.LINEAR)
    key(scene, "camera.pose", round(anim.duration, 4), tuple(last.value), Interp.LINEAR)


def save(scene: Scene, name: str, holds: bool = False, hike: Journey | None = None) -> Path:
    """Write ``<project>/scenes/<name>.json`` (set POIs with scene.set_pois before keying their
    properties). With ``holds`` a still second is added at both ends first (see add_holds)."""
    if holds:
        add_holds(scene)
    path = scene.save((hike or current()).session.project.folder / "scenes" / f"{name}.json")
    print(f"wrote {path}")
    return path


def poi(pid: str, where, icon: str, size: float = 60.0, caption: str = "", hike=None) -> Poi:
    """A symbol at ``where``: (lat, lon), a named place of the journey (str) or an ENU point.
    ``icon``: a builtin icon name."""
    journey = hike or current()
    if isinstance(where, str):
        lat, lon = journey.places[where]
    elif isinstance(where, tuple):
        lat, lon = where
    else:
        lon, lat = journey.lonlat(where)
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


def key_head(scene: Scene, hike: Journey, times, epochs) -> None:
    """progress.head keys (linear) for the journey times ``epochs`` (UTC) at ``times``."""
    for t, e in zip(times, epochs, strict=True):
        key(scene, "progress.head", round(float(t), 4), hike.head_at_time(float(e)), Interp.LINEAR)


def head_keys_for(scene: Scene, hike: Journey, day, t_from: float, t_to: float, *,
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


def reached_at(head_of, distance: float, t_from: float, t_to: float) -> float:
    """Timeline time when head_of(t) first reaches ``distance``."""
    for t in np.arange(t_from, t_to + 1e-9, 1.0 / 60.0):
        if head_of(t) >= distance:
            return round(float(t), 3)
    return t_to


def travel_scene(scene: Scene, hike: Journey, stem, *, duration: float = 7.0,
                 t0: float = 1.0, t1: float | None = None, near: float = 6000.0,
                 far: float | None = None, pitch_near: float = -38.0, pitch_far: float = -58.0,
                 side: float = -35.0, a=None, b=None, near_b: float | None = None,
                 pitch_b: float | None = None, clearance: float = 400.0,
                 lead: float = 0.35, offset: float = 0.0, heading_a: float | None = None,
                 heading_b: float | None = None, apply: bool = True,
                 camera: bool = True, keep: float | None = None, settle: float = 1.2):  # fmt: skip
    """A simple flight along travel line(s) ``stem``: hold on the start until ``t0``, the
    dotted line grows (in its real time) until ``t1`` while the camera flies along, hold on
    the end until ``duration``. All times are shifted by ``offset`` on the timeline.

    The camera looks across the line (``side`` degrees off the direction of travel; or from
    ``heading_a`` to ``heading_b``), zooms out to ``far`` (default: 70 % of the distance)
    halfway and back in. ``a``/``b``: ENU look points at the start / end (default: the line's
    ends). ``keep`` (metres): the last ``keep`` metres of the drawn line stay in frame (the
    camera aims at their middle and backs off far enough). Returns (start epoch, end epoch,
    shots); with ``apply`` the head keys are set, and the camera keys too unless ``camera`` is
    False. ``settle``: seconds in which the camera leaves a / settles on b."""
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
    heads = gaussian(np.array(heads), 0.6 / step)
    d_line0 = hike.path.distance_for_time(e0)
    if keep is not None:
        tails, spans = [], []
        for t in times:
            d = hike.path.distance_for_time(e0 + u_of(t) * (e1 - e0))
            tail = hike.path.position_at(max(d_line0, d - keep))
            head = hike.path.position_at(d)
            tails.append((tail + head) / 2.0)
            spans.append(float(np.linalg.norm((head - tail)[:2])))
        mids = gaussian(np.array(tails), 1.0 / step)  # V3: calmer (no sudden moves)
        spans = gaussian(np.array(spans), 1.0 / step)
    shots = []
    for i, t in enumerate(times):
        u = u_of(t)
        w = smoother(min(1.0, max(0.0, (t - t0) / settle)))  # leave a / arrive at b softly
        w_end = smoother(min(1.0, max(0.0, (t - t1 + settle) / settle)))  # (still in the holds)
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
            key(scene, "haze.aerial", round(float(t + offset), 4), 0.4 - 0.22 * bump, Interp.LINEAR)
    if apply and camera:
        key_shots(scene, shots)
    return e0, e1, shots


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


def night_light(scene: Scene, t_from: float, t_to: float, ambient: float = 3.5,
                exposure: float = 1.4) -> None:  # fmt: skip
    """A readable night (eyes adapted): ambient and exposure rise between t_from and t_to
    (both 0: from the start)."""
    for pid, value, base in (
        ("light.night_ambient", ambient, 1.0),
        ("post.exposure", exposure, 0.0),
    ):
        if t_to <= 0.0:
            key(scene, pid, 0.0, value)
            continue
        key(scene, pid, 0.0, base)
        key(scene, pid, t_from, base)
        key(scene, pid, t_to, value)


def clamp_sun(scene: Scene, hike: Journey, head_of, times, latest: str | None = None,
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


def daylight_keys(scene: Scene, hike: Journey, times, head_of, sigma_s: float = 0.8) -> None:
    """sun.datetime keys from the recorded time at the head (``head_of(t)``: path distance),
    for heads that jump over nights and stops. The clock time is smoothed over ``sigma_s`` on
    the date of the head: from one day's evening to the next morning the sun moves back over
    the sky instead of passing through the night (no dark flash between the days)."""
    times = np.asarray(times, dtype=np.float64)
    whens = [local(track_time_at(hike.path, head_of(float(t)))) for t in times]
    hours = np.array([w.hour + w.minute / 60.0 + w.second / 3600.0 for w in whens])
    step = float(times[1] - times[0]) if len(times) > 1 else 1.0
    hours = gaussian(hours, sigma_s / step)
    scene.store.set("sun.follow_track", False)
    clocks = [dt.datetime(w.year, w.month, w.day) + dt.timedelta(hours=float(h))
              for w, h in zip(whens, hours, strict=True)]  # fmt: skip
    for i, (t, clock) in enumerate(zip(times, clocks, strict=True)):
        # (a key before a change of the date steps: interpolating would run through the night)
        nxt = clocks[min(i + 1, len(clocks) - 1)]
        interp = Interp.STEP if nxt.date() != clock.date() else Interp.LINEAR
        key(scene, "sun.datetime", round(float(t), 4), clock, interp)


# --- generators (also used by the Animation menu of the application) -------------------------
def head_distance_of(scene: Scene, path) -> Callable[[float], float]:
    """``t -> path distance`` of the drawn head, from the scene's progress keys."""
    curves = [scene.animation.curves.get(pid) for pid in ("progress.head", "progress.mode")]
    head_c, mode_c = (c if c is not None and c.keys else None for c in curves)

    def head_of(t: float) -> float:
        head = head_c.evaluate(t) if head_c else scene.store["progress.head"]
        mode = mode_c.evaluate(t) if mode_c else scene.store["progress.mode"]
        return path.distance_for(head, mode)

    return head_of


def clear_keys(scene: Scene, pid: str, t0: float, t1: float) -> None:
    """Remove the keys of ``pid`` between t0 and t1 (inclusive)."""
    curve = scene.animation.curves.get(pid)
    if curve is not None:
        curve.keys = [k for k in curve.keys if not t0 - 1e-6 <= k.time <= t1 + 1e-6]


def draw_keys(scene: Scene, hike: Journey, d0: float, d1: float, t0: float, t1: float, *,
              realtime: bool, soft: bool = True, step: float = 1.0 / 15.0):  # fmt: skip
    """progress.head keys: the head runs from path distance ``d0`` to ``d1`` between the
    timeline times t0 and t1, in the recorded time (breaks and nights pause the line) or at an
    even speed; ``soft``: it starts and stops gently. Returns head_of(t) -> path distance."""
    path = hike.path
    mode = scene.store["progress.mode"]
    e0, e1 = path.time_at(d0), path.time_at(d1)
    timed = realtime and e0 is not None and e1 is not None and e1 > e0

    def head_of(t: float) -> float:
        f = min(max((t - t0) / max(t1 - t0, 1e-9), 0.0), 1.0)
        f = smoother_soft(f) if soft else f
        if timed:
            return min(max(path.distance_for_time(e0 + f * (e1 - e0)), d0), d1)
        return d0 + f * (d1 - d0)

    clear_keys(scene, "progress.head", t0, t1)
    for t in np.arange(t0, t1 + 1e-9, step):
        key(scene, "progress.head", round(float(t), 4),
            path.progress_for_distance(head_of(float(t)), mode), Interp.LINEAR)  # fmt: skip
    return head_of


def sun_keys(scene: Scene, hike: Journey, times, head_of, *, earliest: float | None = None,
             latest: float | None = None, smooth_nights: bool = False,
             sigma_s: float = 0.8) -> None:  # fmt: skip
    """sun.datetime keys from the recorded time at the head (``head_of(t)``: path distance):
    the light of the hike. ``earliest`` / ``latest``: limits of the clock time in hours (e.g.
    7.5) for tracks that start before dawn or end after dark; ``smooth_nights``: where the head
    jumps from one evening to the next morning, the sun moves back over the sky (smoothed over
    ``sigma_s``) instead of through the night."""
    times = np.asarray(times, dtype=np.float64)
    whens = [local(track_time_at(hike.path, head_of(float(t)))) for t in times]
    hours = np.array([w.hour + w.minute / 60.0 + w.second / 3600.0 for w in whens])
    if smooth_nights and len(times) > 1:
        hours = gaussian(hours, sigma_s / float(times[1] - times[0]))
    if earliest is not None:
        hours = np.maximum(hours, earliest)
    if latest is not None:
        hours = np.minimum(hours, latest)
    scene.store.set("sun.follow_track", False)
    clear_keys(scene, "sun.datetime", float(times[0]), float(times[-1]))
    clocks = [dt.datetime(w.year, w.month, w.day) + dt.timedelta(hours=float(h))
              for w, h in zip(whens, hours, strict=True)]  # fmt: skip
    for i, (t, clock) in enumerate(zip(times, clocks, strict=True)):
        # (a key before a change of the date steps: interpolating would run through the night)
        nxt = clocks[min(i + 1, len(clocks) - 1)]
        interp = Interp.STEP if nxt.date() != clock.date() else Interp.LINEAR
        key(scene, "sun.datetime", round(float(t), 4), clock, interp)


# --- the frame origin --------------------------------------------------------------------------
def _camera_axes(heading: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, right, up at roll 0) of a camera in its ENU frame (float64; roll r turns the up
    axis to up * cos r + right * sin r, as Camera.rotation_matrix does)."""
    from earthling.render.camera import forward_vector

    f = forward_vector(heading, pitch)
    right = np.cross(f, [0.0, 0.0, 1.0])
    n = float(np.linalg.norm(right))
    right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return f, right, np.cross(right, f)


def reframe_pose(pose, old, new) -> tuple:
    """A camera pose (ENU of LocalFrame ``old``) as seen in LocalFrame ``new``: the same place
    and view, so changing a scene's origin moves no camera."""
    x, y, z, heading, pitch, roll = pose
    eye = np.asarray(new.ecef_to_enu(old.enu_to_ecef(np.array([x, y, z]))), dtype=np.float64)
    f, right, up0 = _camera_axes(heading, pitch)
    a = math.radians(roll)
    turn = old.rotation @ new.rotation.T  # (ENU directions: old -> ECEF -> new)
    forward, up = f @ turn, (up0 * math.cos(a) + right * math.sin(a)) @ turn
    h, p, _ = look_orientation(np.zeros(3), forward)
    _, right2, up2 = _camera_axes(h, p)
    r = math.degrees(math.atan2(float(np.dot(up, right2)), float(np.dot(up, up2))))
    return (float(eye[0]), float(eye[1]), float(eye[2]), h, p, r)


def move_origin(scene: Scene, old, new) -> None:
    """Re-express the scene's ENU values (camera keys and pose, the look-at target) from
    LocalFrame ``old`` in LocalFrame ``new`` (e.g. before setting ``scene.origin``)."""

    def point(v) -> tuple:
        return tuple(float(c) for c in new.ecef_to_enu(old.enu_to_ecef(np.asarray(v, float))))

    for pid, convert in (("camera.pose", lambda v: reframe_pose(v, old, new)),
                         ("camera.target", point)):  # fmt: skip
        scene.store.set(pid, convert(scene.store[pid]), notify=False)
        curve = scene.animation.curves.get(pid)
        for k in curve.keys if curve is not None else ():
            k.value = convert(k.value)
