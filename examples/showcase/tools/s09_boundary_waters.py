"""Showcase 9: five days by canoe through the Boundary Waters (~32 s).

Late afternoon: the camera skims low over a lake through a light haze towards the canoe, then
rises with it, looking into the low sun so the lakes glint, while five days of paddling and
portages draw (blue on the lakes, white dots over the portages, split by the water mask), a
tent at every night camp; at the end the whole route. Autumn, late September; the country
border (Canada / USA) glows across the lakes.
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
    map_shots,
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
    s.set("water.wind", 4.0)  # (a breeze: a broader glint on the lakes)
    s.set("water.waves", 0.6)
    # brighter from above (looking into the sun low over the lake, the skim is bright already)
    key(scene, "post.exposure", 0.0, 0.0)
    key(scene, "post.exposure", 4.5, 0.0)
    key(scene, "post.exposure", 8.0, 0.4)
    s.set("light.ambient", 1.2)
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
    # late afternoon: the sun low in the south-west, the camera looking towards it (glint)
    key(scene, "sun.datetime", 0.0, dt.datetime(2026, 9, 19, 16, 0), Interp.LINEAR)
    key(scene, "sun.datetime", DURATION, dt.datetime(2026, 9, 19, 17, 30), Interp.LINEAR)
    s.set("weather.enabled", True)
    fog = "weather.layer1."
    for pid, value in (("enabled", True), ("base", 420.0), ("thickness", 60.0),
                       ("coverage", 0.55), ("scale", 700.0), ("wind_speed", 1.0),
                       ("detail", 0.8), ("brightness", 1.2)):  # fmt: skip
        s.set(fog + pid, value)
    key(scene, fog + "density", 0.0, 0.35)
    key(scene, fog + "density", 3.0, 0.25)
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

    # B: rising with the canoe, looking south-west into the sun: the canoe and the drawn
    # route always in the picture (the camera zooms out as far as needed for them)
    def distance_of(t: float) -> float:
        u = smoother(min(max((t - 5.0) / 10.0, 0.0), 1.0))
        return float(np.exp(np.log(600.0) + (np.log(9_000.0) - np.log(600.0)) * u))

    def pitch_of(t: float) -> float:
        return -30.0 - 22.0 * smoother(min(max((t - 5.0) / 10.0, 0.0), 1.0))

    def head_of(t: float) -> float:
        return path.total_m * float(scene.animation.curves["progress.head"].evaluate(t))

    follow = map_shots(hike, times, head_of, distance_of=distance_of, pitch_of=pitch_of,
                       heading_of=lambda t: 215.0, sigma_s=1.2, fit=0.3, clearance=60.0)  # fmt: skip
    # C: the whole route at the end, from the same direction
    every = np.vstack([g.enu for g in path.tracks])
    lo, hi = every.min(axis=0), every.max(axis=0)
    centre = (lo + hi) / 2.0
    span = max(hi[0] - lo[0], (hi[1] - lo[1]) * 16.0 / 9.0)
    top = 0.75 * span / np.tan(np.radians(s["view.fov"]) / 2.0) / (16.0 / 9.0)
    overview = [(t, orbit_eye(centre, 215.0, -60.0, top), centre) for t in times]
    shots = blend_shots([(t, *skim(t)) for t in times], follow, 4.5, 7.5)
    shots = blend_shots(shots, overview, PADDLE[1] - 2.0, PADDLE[1] + 2.0)
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
