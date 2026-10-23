"""Showcase 4b: into the storm, low through the canyon (~5 s, for the trailer).

Dusk on the North Kaibab trail: the camera flies low along the trail from Phantom Ranch up
Bright Angel Canyon, below the rims, towards the North Rim, where a monsoon thunderstorm
builds - the clouds close in, the canyon gets dark and lightning strikes ahead.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import Interp, key, key_shots, load_hike, new_scene, save, smoother

DURATION = 5.0
FROM_F, TO_F = 0.38, 0.52  # the flight: these fractions of the hike's time (north of Phantom)
CLEARANCE = 450.0  # metres above the trail (still well below the rims, ~1300 m higher)
AHEAD = 2500.0  # the camera looks this far ahead along its line ...
LOOK_UP = -120.0  # ... and a little down: the trail ahead, the storm above the rims


def gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    """Low-pass along the first axis (gaussian of ``sigma`` samples, the ends held)."""
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, "valid"), 0, padded)


def main() -> None:
    hike = load_hike("s04_grand_canyon")
    scene = new_scene(hike, DURATION)
    s = scene.store
    path = hike.path
    e0, e1 = hike.day_times(1)
    # the whole trail drawn ahead of the camera, no hiker, no stats
    s.set("progress.mode", "time")
    s.set("progress.head", 1.0)
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.85, 0.2))
    s.set("tracks.width", 4.5)
    s.set("tracks.casing", 2.5)
    s.set("tracks.emissive", 0.6)
    s.set("tracks.glow", 1.6)
    s.set("marker.visible", False)
    s.set("stats.visible", False)
    s.set("labels.visible", False)

    # --- the flight line: the trail smoothed (no zigzags), a little above it ----------------
    d0 = path.distance_for_time(e0 + FROM_F * (e1 - e0))
    d1 = path.distance_for_time(e0 + TO_F * (e1 - e0)) + AHEAD
    ds = np.arange(d0, d1, 10.0)
    line = np.array([path.position_at(float(d)) for d in ds])
    line = gaussian(line, 25.0)  # (250 m)
    ground = np.array([hike.ground_z(p) for p in line])
    line[:, 2] = gaussian(np.maximum(ground, line[:, 2]), 30.0) + CLEARANCE
    # (raised where the smoothed line would come too close to a wall)
    for i, p in enumerate(line):
        near = hike.max_ground_z(p - (40.0, 0, 0), p + (40.0, 0, 0), samples=5, radius=60.0)
        line[i, 2] = max(line[i, 2], near + 90.0)
    line[:, 2] = gaussian(line[:, 2], 20.0)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    travel = along[-1] - AHEAD
    shots = []
    for t in np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0):
        # an even glide that eases out at the end (the cut comes at full speed)
        u = t / DURATION
        a = travel * (0.85 * u + 0.15 * smoother(u))
        eye = np.array([np.interp(a, along, line[:, k]) for k in range(3)])
        target = np.array([np.interp(a + AHEAD, along, line[:, k]) for k in range(3)])
        target[2] += LOOK_UP
        shots.append((round(float(t), 4), eye, target))
    key_shots(scene, shots)
    print(f"flight {travel:.0f} m, lowest clearance "
          f"{min(e[2] - hike.ground_z(e) for _, e, _ in shots):.0f} m")  # fmt: skip
    s.set("view.fov", 62.0)

    # --- dusk and the storm closing in ---------------------------------------------------------
    s.set("sun.follow_track", False)
    key(scene, "sun.datetime", 0.0, dt.datetime(2026, 8, 20, 18, 40), Interp.LINEAR)
    key(scene, "sun.datetime", DURATION, dt.datetime(2026, 8, 20, 19, 0))
    s.set("haze.aerial", 0.6)
    s.set("weather.enabled", True)
    s.set("weather.layer1.enabled", False)
    s.set("weather.cloud_shadows", 0.9)
    p = "weather.layer2."
    for pid, value in (("enabled", True), ("base", 3400.0), ("thickness", 2800.0),
                       ("scale", 5000.0), ("wind_speed", 10.0), ("wind_dir", 200.0),
                       ("density", 0.9), ("detail", 0.7), ("brightness", 0.8)):  # fmt: skip
        s.set(p + pid, value)
    key(scene, p + "coverage", 0.0, 0.45)
    key(scene, p + "coverage", DURATION, 0.75)
    key(scene, "post.exposure", 0.0, 0.5)
    key(scene, "post.exposure", DURATION, -0.1)
    s.set("lightning.rate", 30.0)
    s.set("lightning.radius_km", 9.0)
    s.set("lightning.intensity", 1.5)
    s.set("lightning.branching", 0.55)
    s.set("lightning.seed", 36)  # (strikes at 1.0, 2.2 and 3.9 s)
    save(scene, "s04_grand_canyon_storm")


if __name__ == "__main__":
    main()
