"""Showcase 10: Moscow - Beijing by train in winter, seven days in ~46 s.

A satellite's view travelling with the train, high above Siberia: the day-night line sweeps
over the land again and again, the cities light up behind it. It starts as a border map (the
countries glowing) that turns into the snowy winter imagery; the clock in the overlay shows the
local time where the train is and jumps at every time zone.
"""

from __future__ import annotations

import math

import numpy as np
from sc_common import (
    Interp,
    gaussian,
    geo_orbit_eye,
    key,
    key_shots,
    load_hike,
    new_scene,
    poi,
    save,
    show,
)

ORIGIN = (52.0, 85.0)
DURATION = 46.0
RIDE = (1.5, 44.0)


def main() -> None:
    hike = load_hike("s10_trans_siberian", ORIGIN)
    path = hike.path
    scene = new_scene(hike, DURATION)
    scene.origin = ORIGIN
    s = scene.store
    s.set("progress.mode", "time")
    s.set("sun.follow_track", True)
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.55, 0.15))
    s.set("tracks.width", 4.0)
    s.set("tracks.casing", 2.0)
    s.set("tracks.emissive", 0.9)
    s.set("trackgroup.tracks.marker", "train")
    s.set("trackgroup.tracks.marker_size", 44.0)
    s.set("night.lights", 3.0)
    s.set("haze.aerial", 0.25)  # (the land shows through)
    s.set("light.night_ambient", 6.0)
    s.set("terrain.exaggeration", 3.0)  # (relief seen from 2000 km)
    # winter: snow on the land (the imagery is a summer mosaic)
    s.set("snow.cover", 0.35)  # (a light cover: the land stays readable)
    s.set("snow.line", -100.0)
    s.set("snow.transition", 300.0)
    s.set("snow.max_slope", 20.0)
    # the clock where the train is
    for pid, value in (("stats.visible", True), ("stats.show_time", True),
                       ("stats.timezone", "position"), ("stats.show_utc_offset", True),
                       ("stats.show_ascent", False), ("stats.show_elevation", False),
                       ("stats.profile", False), ("stats.date_format", "long")):  # fmt: skip
        s.set(pid, value)
    s.set("labels.visible", True)
    # countries and the large cities (Natural Earth): where on the continent the train is
    s.set("labels.peaks", False)
    s.set("labels.passes", False)
    s.set("labels.places", False)
    s.set("labels.cities", True)
    s.set("labels.countries", True)
    s.set("labels.min_population", 1_000_000.0)
    s.set("labels.max_count", 12)
    s.set("labels.max_distance", 4000.0)
    s.set("labels.size", 17.0)
    # the border map first, then the imagery
    s.set("layers.a", "borders")
    s.set("layers.b", "satellite")
    s.set("borders.overlay", True)
    s.set("borders.regions", False)
    s.set("borders.width", 2.2)
    s.set("borders.color", (0.35, 0.85, 1.0))  # (cyan: the line of the train is orange)
    s.set("borders.glow_strength", 1.2)
    s.set("layer_borders.land", (0.2, 0.22, 0.26))
    s.set("layer_borders.sea", (0.05, 0.08, 0.14))
    key(scene, "layers.mix", 0.0, 0.0)
    key(scene, "layers.mix", 3.0, 0.0)
    key(scene, "layers.mix", 7.0, 1.0)
    key(scene, "borders.opacity", 0.0, 1.0)
    key(scene, "borders.opacity", 7.0, 0.4)

    # the ride in the GPX time: a day every ~6 s (the nights included)
    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    key(scene, "progress.head", RIDE[0], 0.0, Interp.LINEAR)
    key(scene, "progress.head", RIDE[1], 1.0, Interp.LINEAR)
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    t = path.time[np.isfinite(path.time)]
    e0, e1 = t[0], t[-1]

    def head_of(tt: float) -> float:
        u = min(max((tt - RIDE[0]) / (RIDE[1] - RIDE[0]), 0.0), 1.0)
        return path.distance_for_time(e0 + u * (e1 - e0))

    # the camera: 1300 km behind and above the train, the look point a little ahead of it
    heads = np.array([path.position_at(head_of(tt)) for tt in times])
    ahead = np.array([path.position_at(head_of(tt) + 400e3) for tt in times])
    look = gaussian(heads * 0.6 + ahead * 0.4, 2.5 * 30)
    shots = []
    for tt, p in zip(times, look, strict=True):
        lat, lon, _ = hike.frame.enu_to_geodetic(p)
        u = min(max((tt - RIDE[0]) / (RIDE[1] - RIDE[0]), 0.0), 1.0)
        dist = 2600e3 - 900e3 * math.sin(math.pi * min(tt / 8.0, 1.0) / 2.0)  # descending first
        eye, target = geo_orbit_eye(hike, float(lat), float(lon), 0.0, 100.0 - 20.0 * u, -52.0,
                                    dist)  # fmt: skip
        shots.append((float(tt), eye, target))
    key_shots(scene, shots)

    stations = [("moscow", path.tracks[0].enu[0], "flag"),
                ("beijing", path.tracks[-1].enu[-1], "finish")]
    scene.set_pois([poi(pid, where, icon, 40.0) for pid, where, icon in stations])
    show(scene, "moscow", None)
    show(scene, "beijing", RIDE[1] - 0.3)
    save(scene, "s10_trans_siberian")


if __name__ == "__main__":
    main()
