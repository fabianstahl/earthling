"""Showcase 8: the Lukla flight and the trek to Everest Base Camp (~34 s).

1. (0-9 s) First person: the camera sits just behind the plane (follow mode) and banks into
   the curves of the flight up the valleys to Lukla.
2. (9-26 s) Seven days up the Khumbu valley, drawn by distance (the nights left out), the
   great peaks labelled as they come into view.
3. (26-34 s) Base Camp: snow starts to fall and settles above the snow line; the camera, above
   Kala Patthar, turns from the tents up to Everest as the snow clouds clear.
"""

from __future__ import annotations

import numpy as np
from sc_common import (
    Interp,
    blend_shots,
    daylight_keys,
    follow_shots,
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

DURATION = 34.0
FLIGHT = (0.5, 9.0)
TREK = (10.0, 26.0)
SUMMIT = (27.9881, 86.9250)


def main() -> None:
    hike = load_hike("s08_everest")
    path = hike.path
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "time")
    s.set("sun.follow_track", True)
    s.set("trackgroup.flight.marker", "plane")
    s.set("trackgroup.flight.marker_size", 56.0)
    s.set("trackgroup.flight.dots", True)
    s.set("trackgroup.flight.dash", 12.0)
    s.set("trackgroup.flight.color_mode", "single")
    s.set("trackgroup.flight.color", (1.0, 0.86, 0.3))
    s.set("trackgroup.trek.color_mode", "single")
    s.set("trackgroup.trek.color", (1.0, 0.42, 0.12))
    s.set("trackgroup.trek.marker", "hiker")
    s.set("trackgroup.trek.marker_size", 40.0)
    s.set("tracks.width", 4.5)
    s.set("tracks.casing", 2.0)
    s.set("haze.aerial", 0.8)
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.passes", False)
    s.set("labels.min_prominence", 600.0)
    s.set("labels.max_count", 7)
    s.set("labels.max_distance", 45.0)
    s.set("labels.min_elevation", 5500.0)
    # stats during the trek: day, date, elevation and the profile
    s.set("stats.visible", True)
    s.set("stats.scope", "day")  # (the day's walk: not the flight's 138 km)
    key(scene, "stats.opacity", 0.0, 0.0)
    key(scene, "stats.opacity", TREK[0] - 0.5, 0.0)
    key(scene, "stats.opacity", TREK[0] + 0.5, 1.0)

    # --- the flight: first person, follow mode with banking ----------------------------------
    flight = 0  # (the first track in time)
    d0, d1 = path.track_range(flight)
    key(scene, "progress.head", 0.0, path.progress_for_distance(d0 + 1.0, "time"), Interp.STEP)
    for t in np.arange(FLIGHT[0], FLIGHT[1] + 1e-9, 1.0 / 15.0):
        u = smoother((t - FLIGHT[0]) / (FLIGHT[1] - FLIGHT[0]))
        key(scene, "progress.head", round(float(t), 4),
            path.progress_for_distance(d0 + 1.0 + u * (d1 - d0 - 2.0), "time"), Interp.LINEAR)  # fmt: skip
    # (wide and calm: far enough behind that the valley and the plane's turns read from outside)
    for pid, value in (("follow.distance", 1100.0), ("follow.height", 260.0), ("follow.lag", 1.5),
                       ("follow.look_ahead", 3000.0), ("follow.bank", 0.2),
                       ("camera.transition", 2.5)):  # fmt: skip
        s.set(pid, value)
    key(scene, "camera.mode", 0.0, "follow", Interp.STEP)
    key(scene, "camera.mode", FLIGHT[1] + 0.3, "keys", Interp.STEP)
    s.set("view.fov", 60.0)
    key(scene, "view.fov", FLIGHT[1], 60.0)
    key(scene, "view.fov", FLIGHT[1] + 2.0, 48.0)

    # --- the trek: by distance over seven days --------------------------------------------
    t0, t1 = path.track_range(1)[0], path.track_range(len(path.tracks) - 1)[1]

    def head_of(t: float) -> float:
        u = min(max((t - TREK[0]) / (TREK[1] - TREK[0]), 0.0), 1.0)
        u = 0.75 * u + 0.25 * u * u * (3 - 2 * u)
        return t0 + 1.0 + u * (t1 - t0 - 2.0)

    for t in np.arange(TREK[0], TREK[1] + 1e-9, 1.0 / 15.0):
        key(scene, "progress.head", round(float(t), 4),
            path.progress_for_distance(head_of(t), "time"), Interp.LINEAR)  # fmt: skip
    # the sun from the recorded time at the head, switching straight from one day's end to the
    # next morning (the head jumps over the nights: "follow the GPX time" showed a night flash)
    def sun_head(t: float) -> float:
        if t >= TREK[0]:
            return head_of(t)
        u = smoother(min(max((t - FLIGHT[0]) / (FLIGHT[1] - FLIGHT[0]), 0.0), 1.0))
        return d0 + 1.0 + (d1 - d0 - 2.0) * u

    daylight_keys(scene, hike, np.arange(0.0, DURATION + 1e-9, 1.0 / 15.0), sun_head)
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    chase = follow_shots(hike, times, lambda t: head_of(max(t, TREK[0])), distance=6000.0,
                         pitch=-22.0, side=40.0, window=2500.0, sigma_s=1.5,
                         heading_sigma_s=4.0, clearance=500.0)  # fmt: skip

    # --- Base Camp: from the tents up to the summit ---------------------------------------
    ebc = hike.end(len(path.tracks))
    summit = hike.enu(*SUMMIT, 8848.0)
    # above Kala Patthar: from the camp itself Nuptse hides the summit (and its label)
    eye = hike.enu(27.9958, 86.8285, 5644.0)
    # well clear of the ground: an eye close to it was pushed up by the renderer's ground
    # clamp whenever finer terrain tiles arrived, and the picture bounced
    eye[2] = max(eye[2] + 450.0, hike.ground_z(eye) + 450.0)

    def finale(t: float):
        u = smoother(min(max((t - 27.0) / 6.0, 0.0), 1.0))
        look = ebc + (summit - ebc) * u
        return eye.copy(), look

    final = [(t, *finale(t)) for t in times]
    shots = blend_shots(chase, final, TREK[1] - 1.0, TREK[1] + 1.5)
    # the keyed camera only matters from the end of the flight (follow mode before)
    start_view = orbit_eye(path.position_at(d1), 30.0, -20.0, 6000.0)
    shots = [(t, start_view, path.position_at(d1)) if t < FLIGHT[1] else (t, e, g)
             for t, e, g in shots]  # fmt: skip
    key_shots(scene, shots)

    # --- snow at Base Camp: snowfall, fresh snow, the clouds below the summit ---------------
    # a broken deck high above the camp: the summit shows above it and through the gaps
    for pid, value in (("rain.enabled", True), ("rain.type", "snow"), ("rain.cloud_base", 7100.0),
                       ("rain.cloud_thickness", 600.0), ("rain.coverage", 0.5),
                       ("rain.darkness", 0.3), ("rain.visibility_km", 25.0), ("rain.wind", 3.0),
                       ("snow.line", 4900.0), ("snow.transition", 250.0)):  # fmt: skip
        s.set(pid, value)
    key(scene, "rain.intensity", 0.0, 0.0)
    key(scene, "rain.intensity", 25.0, 0.0)
    key(scene, "rain.intensity", 28.0, 0.6)
    key(scene, "rain.intensity", 30.5, 0.5)
    key(scene, "rain.intensity", 33.0, 0.05)  # (the snowfall stops: the summit appears)
    key(scene, "rain.cloud_density", 0.0, 0.0)
    key(scene, "rain.cloud_density", 25.5, 0.0)
    key(scene, "rain.cloud_density", 28.0, 0.7)
    key(scene, "rain.cloud_density", 30.0, 0.7)
    key(scene, "rain.cloud_density", 32.5, 0.0)
    key(scene, "snow.cover", 0.0, 0.0)
    key(scene, "snow.cover", 25.0, 0.0)
    key(scene, "snow.cover", 33.0, 0.6)

    lukla = path.position_at(d1)
    namche = hike.enu(27.8048, 86.7120, 3440.0)
    tengboche = hike.enu(27.8361, 86.7638, 3870.0)
    scene.set_pois([poi("lukla", lukla, "airport", 44.0), poi("namche", namche, "hotel", 38.0),
                    poi("tengboche", tengboche, "attraction", 38.0),
                    poi("ebc", ebc, "tent", 44.0)])  # fmt: skip
    show(scene, "lukla", FLIGHT[1] - 0.5)
    show(scene, "namche", TREK[0] + 2.5)
    show(scene, "tengboche", TREK[0] + 6.0)
    show(scene, "ebc", TREK[1] - 0.3)
    save(scene, "s08_everest")


if __name__ == "__main__":
    main()
