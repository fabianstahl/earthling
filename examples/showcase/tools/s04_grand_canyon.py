"""Showcase 4: Grand Canyon rim to rim (~34 s).

1. (0-7 s) Establishing shot over the canyon: the texture layers crossfade from the imagery to
   the elevation colours and the contour map and back (a geology lesson), while the vertical
   exaggeration settles from 1.6 to 1.
2. (7-21 s) Down into the canyon: a low flight along the South Kaibab trail to the Colorado and
   up the North Kaibab trail, the sun following the GPX time (05:00-14:40).
3. (21-34 s) A locked-off camera on the South Rim: only the light moves. The afternoon turns into
   dusk while a monsoon thunderstorm builds over the North Rim - lightning in daylight.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import (
    Interp,
    blend_shots,
    follow_shots,
    head_keys_for,
    key,
    key_shots,
    load_hike,
    new_scene,
    orbit_eye,
    poi,
    save,
    show,
    still_shots,
)

DURATION = 34.0
HIKE = (8.0, 20.5)  # the hike is drawn between these times
STATIC = 21.0  # the locked-off time-lapse starts


def main() -> None:
    hike = load_hike("s04_grand_canyon")
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "time")
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.85, 0.2))
    s.set("tracks.width", 4.5)
    s.set("tracks.casing", 2.5)
    s.set("haze.aerial", 0.9)
    s.set("trackgroup.tracks.marker", "hiker")
    s.set("trackgroup.tracks.marker_size", 40.0)
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.min_prominence", 150.0)
    s.set("labels.max_count", 6)
    s.set("labels.max_distance", 20.0)
    s.set("contours.interval", 100.0)
    s.set("layer_elevation.min", 700.0)
    s.set("layer_elevation.max", 2700.0)
    s.set("layer_elevation.ramp", "turbo")

    path = hike.path
    e0, e1 = hike.day_times(1)
    centre = (hike.start(1) + hike.end(1)) / 2.0
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)

    # --- 1: the establishing shot with the layers -------------------------------------------
    opening = orbit_eye(centre, 5.0, -38.0, 30_000.0)
    approach = orbit_eye(centre, 15.0, -30.0, 16_000.0)
    shots = [(t, opening + (approach - opening) * min(1.0, t / 7.0) ** 1.2, centre.copy())
             for t in times]  # fmt: skip
    for t, value, interp in ((0.0, 1.6, Interp.EASE_IN_OUT), (7.0, 1.0, Interp.LINEAR)):
        key(scene, "terrain.exaggeration", t, value, interp)
    key(scene, "layers.a", 0.0, "satellite", Interp.STEP)
    key(scene, "layers.b", 0.0, "elevation", Interp.STEP)
    key(scene, "layers.b", 3.2, "contours", Interp.STEP)  # (switched while slot B is hidden)
    for t, mix in ((0.6, 0.0), (1.8, 1.0), (2.6, 1.0), (3.0, 0.0), (3.4, 0.0), (4.4, 1.0),
                   (5.4, 1.0), (6.6, 0.0)):  # fmt: skip
        key(scene, "layers.mix", t, mix)
    key(scene, "contours.overlay", 0.0, False, Interp.STEP)

    # --- 2: the hike, low through the canyon ------------------------------------------------
    s.set("sun.follow_track", True)
    head_of = head_keys_for(scene, hike, 1, *HIKE, realtime=True)
    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    chase = follow_shots(hike, times, lambda t: head_of(min(max(t, HIKE[0]), HIKE[1])),
                         distance=2600.0, pitch=-16.0, side=-35.0, window=1500.0, sigma_s=1.2,
                         heading_sigma_s=3.0, clearance=250.0)  # fmt: skip
    shots = blend_shots(shots, chase, 6.0, 9.0)

    # --- 3: locked off on the South Rim, the storm over the North Rim -------------------------
    # at the edge of the rim north of Yaki Point, down into the canyon and across to the North Rim
    rim_eye = hike.enu(36.0575, -112.0815, 2300.0)
    across = hike.enu(36.13, -112.068, 1450.0)
    static = still_shots(times, rim_eye, across)
    shots = blend_shots(shots, static, STATIC - 1.5, STATIC)
    key_shots(scene, shots)

    # the sun: morning light for the opening (the hike starts before sunrise), from the moment
    # the head reaches that time the GPX time, then a time-lapse to dusk
    switch = HIKE[0] + 2.0
    morning = dt.datetime.fromtimestamp(
        path.time_at(head_of(switch)), dt.timezone(dt.timedelta(hours=-7))
    ).replace(tzinfo=None)
    key(scene, "sun.datetime", 0.0, morning, Interp.STEP)
    key(scene, "sun.follow_track", 0.0, False, Interp.STEP)
    key(scene, "sun.follow_track", switch, True, Interp.STEP)
    key(scene, "sun.follow_track", STATIC, False, Interp.STEP)
    finish = dt.datetime.fromtimestamp(e1, dt.timezone(dt.timedelta(hours=-7))).replace(tzinfo=None)
    key(scene, "sun.datetime", STATIC, finish, Interp.LINEAR)
    # (more of the time-lapse in the golden hour than in the afternoon)
    for t, clock in ((STATIC + 3.0, (16, 30)), (28.0, (18, 25)), (31.5, (19, 5)),
                     (DURATION - 0.5, (19, 35))):  # fmt: skip
        key(scene, "sun.datetime", t, dt.datetime(2026, 8, 20, *clock), Interp.LINEAR)
    key(scene, "haze.aerial", STATIC - 1.0, 0.9)
    key(scene, "haze.aerial", STATIC + 1.0, 0.55)
    # monsoon clouds build up over the afternoon, lightning from 26 s, the rim gets dark
    s.set("weather.enabled", True)
    s.set("weather.layer1.enabled", False)
    p = "weather.layer2."
    for pid, value in (("enabled", True), ("base", 3600.0), ("thickness", 2500.0),
                       ("scale", 5000.0), ("wind_speed", 8.0), ("wind_dir", 200.0),
                       ("density", 0.8), ("detail", 0.7)):  # fmt: skip
        s.set(p + pid, value)
    key(scene, p + "density", 0.0, 0.0)  # no clouds before the time-lapse
    key(scene, p + "density", STATIC - 1.5, 0.0)
    key(scene, p + "density", STATIC + 0.5, 0.8)
    key(scene, p + "coverage", 0.0, 0.08)
    key(scene, p + "coverage", STATIC, 0.12)
    key(scene, p + "coverage", 29.0, 0.5)
    s.set("lightning.radius_km", 22.0)
    s.set("lightning.intensity", 2.0)
    s.set("lightning.branching", 0.5)
    key(scene, "lightning.rate", 0.0, 0.0, Interp.STEP)
    key(scene, "lightning.rate", 25.5, 0.0, Interp.LINEAR)
    key(scene, "lightning.rate", 28.0, 14.0, Interp.LINEAR)
    key(scene, "post.exposure", STATIC, 0.0)
    key(scene, "post.exposure", DURATION, 0.6)
    s.set("lightning.seed", 7)

    # stats during the hike (an inverted profile: first down, then up)
    s.set("stats.visible", True)
    s.set("stats.show_day", False)
    s.set("stats.show_time", True)
    key(scene, "stats.opacity", 0.0, 0.0)
    key(scene, "stats.opacity", HIKE[0] - 0.5, 0.0)
    key(scene, "stats.opacity", HIKE[0] + 0.5, 1.0)
    key(scene, "stats.opacity", STATIC + 1.0, 1.0)
    key(scene, "stats.opacity", STATIC + 2.5, 0.0)

    phantom = path.position_at(path.distance_for_time(e0 + 0.36 * (e1 - e0)))
    scene.set_pois([poi("south", hike.start(1), "flag", 40.0),
                    poi("phantom", phantom, "hotel", 40.0),
                    poi("north", hike.end(1), "finish", 44.0)])  # fmt: skip
    show(scene, "south", HIKE[0] - 0.8)
    show(scene, "phantom", HIKE[0] + 3.0)
    show(scene, "north", HIKE[1] - 0.3)
    save(scene, "s04_grand_canyon")


if __name__ == "__main__":
    main()
