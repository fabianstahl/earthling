"""Showcase 4b: into the storm, low through the canyon (~5 s, for the trailer).

Dusk on the North Kaibab trail: the camera flies in one straight, even line over the trail
from Phantom Ranch up Bright Angel Canyon, below the rims, towards the North Rim, where a
monsoon thunderstorm builds - the clouds close in, the canyon gets dark and lightning strikes
ahead.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import Interp, key, key_shots, load_hike, new_scene, save

DURATION = 5.0
FROM_F, TO_F = 0.38, 0.46  # the flight starts here (north of Phantom) towards there ...
TRAVEL = 2000.0  # ... and flies this far, in a straight line
CLEARANCE = 300.0  # metres above the highest ground under the line
AHEAD = 2500.0  # the camera looks this far ahead ...
LOOK_UP = -350.0  # ... and down: the trail ahead, the storm above the rims


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

    # --- the flight: one straight, even line high above the trail (no turns, no speed-ups) ---
    a = path.position_at(path.distance_for_time(e0 + FROM_F * (e1 - e0)))
    b = path.position_at(path.distance_for_time(e0 + TO_F * (e1 - e0)))
    b = a + (b - a) * (TRAVEL / float(np.linalg.norm((b - a)[:2])))
    ahead = (b - a)[:2] / np.linalg.norm((b - a)[:2])
    # one height for the whole line: clear of everything below it and the trail ahead in view
    under = max(hike.max_ground_z(a, b, samples=60, radius=80.0), a[2], b[2])
    height = under + CLEARANCE
    shots = []
    for t in np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0):
        eye = a + (b - a) * (t / DURATION)
        eye[2] = height
        target = eye + np.array([ahead[0] * AHEAD, ahead[1] * AHEAD, LOOK_UP])
        shots.append((round(float(t), 4), eye, target))
    key_shots(scene, shots)
    print(f"flight {np.linalg.norm((b - a)[:2]):.0f} m at {height:.0f} m "
          f"({height - under:.0f} m above the highest ground under it)")  # fmt: skip
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
