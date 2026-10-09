"""Showcase 9: five days by canoe through the Boundary Waters (~32 s).

The first morning: the camera skims low over a lake through the morning mist towards the canoe,
then cranes straight up to 4 km while five days of paddling and portages draw (blue on the
lakes, white dots over the portages, split by the water mask), a tent at every night camp.
Autumn, late September; the country border (Canada / USA) glows across the lakes.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import (
    Interp,
    blend_shots,
    key,
    key_shots,
    load_hike,
    new_scene,
    orbit_eye,
    poi,
    save,
    show,
    smoother,
)

DURATION = 32.0
PADDLE = (6.0, 28.0)


def main() -> None:
    hike = load_hike("s09_boundary_waters")
    path = hike.path
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "distance")
    p = "trackgroup.canoe."
    for pid, value in (("color_mode", "single"), ("color", (1.0, 1.0, 1.0)), ("dots", True),
                       ("dash", 9.0), ("water_color", (0.15, 0.7, 1.0)), ("water_dash", 0.0),
                       ("marker", "canoe"), ("marker_size", 44.0)):  # fmt: skip
        s.set(p + pid, value)
    s.set("tracks.width", 4.5)
    s.set("tracks.casing", 2.0)
    s.set("water.enabled", True)
    s.set("water.wind", 2.0)
    s.set("water.waves", 0.6)
    key(scene, "haze.aerial", 0.0, 0.8)
    key(scene, "haze.aerial", 8.0, 0.8)
    key(scene, "haze.aerial", 16.0, 0.3)
    s.set("borders.overlay", True)
    s.set("borders.regions", False)
    s.set("borders.color", (1.0, 0.85, 0.35))
    s.set("borders.width", 2.0)
    s.set("borders.glow_strength", 0.8)
    s.set("stats.visible", True)
    s.set("stats.show_ascent", False)
    s.set("stats.show_elevation", False)
    s.set("stats.profile", False)
    s.set("stats.day_style", "day")
    # the morning: low sun, mist lying on the lakes, burning off
    key(scene, "sun.datetime", 0.0, dt.datetime(2026, 9, 19, 8, 5), Interp.LINEAR)
    key(scene, "sun.datetime", PADDLE[0], dt.datetime(2026, 9, 19, 8, 30), Interp.LINEAR)
    key(scene, "sun.datetime", DURATION, dt.datetime(2026, 9, 19, 11, 0), Interp.LINEAR)
    s.set("weather.enabled", True)
    fog = "weather.layer1."
    for pid, value in (("enabled", True), ("base", 420.0), ("thickness", 60.0),
                       ("coverage", 0.55), ("scale", 700.0), ("wind_speed", 1.0),
                       ("detail", 0.8), ("brightness", 1.2)):  # fmt: skip
        s.set(fog + pid, value)
    key(scene, fog + "density", 0.0, 0.7)
    key(scene, fog + "density", 3.0, 0.5)
    key(scene, fog + "density", 9.0, 0.0)  # (burnt off before the crane rises above it)

    # the paddling: by distance, the night camps (day ends) as tents
    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    key(scene, "progress.head", PADDLE[0], 0.002, Interp.LINEAR)
    key(scene, "progress.head", PADDLE[1], 1.0, Interp.EASE_OUT)
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    start = hike.start(1)
    surface = start[2]
    # A: low over the lake, 1.5 km out on the first day's route, gliding back towards the
    # put-in through the mist (the route's first kilometres are on Moose Lake)
    out = path.position_at(1500.0)
    a_from = out + (out - start) * 0.3
    a_from[2] = surface + 14.0
    a_to = out + (start - out) * 0.35
    a_to[2] = surface + 40.0

    def skim(t: float):
        u = smoother(min(t / 6.0, 1.0))
        return a_from + (a_to - a_from) * u, start + np.array([0.0, 0.0, 2.0])

    # B: the crane - straight up above the put-in, looking down onto the lake maze
    # the whole route in the frame at the end: the centre of its bounds, high enough for them
    every = np.vstack([g.enu for g in path.tracks])
    lo, hi = every.min(axis=0), every.max(axis=0)
    centre = (lo + hi) / 2.0
    span = max(hi[0] - lo[0], (hi[1] - lo[1]) * 16.0 / 9.0)
    top = 0.62 * span / np.tan(np.radians(s["view.fov"]) / 2.0) / (16.0 / 9.0)

    def crane(t: float):
        u = smoother(min(max((t - 6.0) / 12.0, 0.0), 1.0))
        look = start + (centre - start) * u
        height = 30.0 + (top - 30.0) * u ** 1.4
        eye = orbit_eye(look, 20.0 - 20.0 * u, -12.0 - 76.0 * u, height)
        return eye, look

    shots = blend_shots([(t, *skim(t)) for t in times], [(t, *crane(t)) for t in times], 5.0, 8.0)
    key_shots(scene, shots)

    pois = []
    for day in range(1, 5):
        pois.append(poi(f"camp{day}", hike.end(day), "tent", 34.0))
    pois.append(poi("putin", start, "flag", 36.0))
    scene.set_pois(pois)
    show(scene, "putin", None)
    curve = scene.animation.curves["progress.head"]
    for day in range(1, 5):  # the tent pops up when the head reaches the camp
        f = path.track_range(day - 1)[1] / path.total_m
        t = next((t for t in times if curve.evaluate(t) >= f), PADDLE[1])
        show(scene, f"camp{day}", round(float(t), 3))
    save(scene, "s09_boundary_waters")


if __name__ == "__main__":
    main()
