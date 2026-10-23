"""Showcase 11: Concordia, the snow clouds part and K2 appears (~6 s, for the trailer).

The Baltoro trek (five days from Askole) has arrived at Concordia. It is snowing, a cloud deck
hides the mountains; the snowfall stops, the clouds dissolve and K2 stands at the end of the
Godwin-Austen glacier, 15 km away. The camera barely moves: a slow push and a slight tilt up.
"""

from __future__ import annotations

import datetime as dt

from sc_common import Interp, key, key_shots, load_hike, new_scene, poi, save, show, smoother

DURATION = 6.0
CLEAR = (2.2, 4.4)  # the clouds dissolve
K2 = (35.88165, 76.51347, 8611.0)


def main() -> None:
    hike = load_hike("s11_k2")
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "time")
    s.set("progress.head", 1.0)  # the trek has arrived
    s.set("trackgroup.trek.color_mode", "single")
    s.set("trackgroup.trek.color", (1.0, 0.42, 0.12))
    s.set("trackgroup.trek.marker", "hiker")
    s.set("trackgroup.trek.marker_size", 40.0)
    s.set("tracks.width", 4.5)
    s.set("tracks.casing", 2.0)
    s.set("haze.aerial", 0.8)
    s.set("view.fov", 30.0)  # (a long lens: K2 fills the end of the glacier)
    # the last day's stats, as in the Everest scene
    s.set("stats.visible", True)
    s.set("stats.scope", "day")
    # the great peaks, once the clouds are gone
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.passes", False)
    s.set("labels.huts", False)
    s.set("labels.min_elevation", 7000.0)
    s.set("labels.min_prominence", 800.0)
    s.set("labels.max_count", 3)
    s.set("labels.max_distance", 45.0)
    key(scene, "labels.opacity", 0.0, 0.0)
    key(scene, "labels.opacity", CLEAR[1] - 0.8, 0.0, Interp.EASE_IN_OUT)
    key(scene, "labels.opacity", CLEAR[1] + 0.2, 1.0)

    # afternoon light from the west on the south face
    s.set("sun.follow_track", False)
    key(scene, "sun.datetime", 0.0, dt.datetime(2026, 7, 14, 16, 20), Interp.LINEAR)
    key(scene, "sun.datetime", DURATION, dt.datetime(2026, 7, 14, 16, 26))

    # snowfall from a deck that hides the upper mountains; it stops and the deck dissolves
    for pid, value in (("rain.enabled", True), ("rain.type", "snow"), ("rain.cloud_base", 6000.0),
                       ("rain.cloud_thickness", 1600.0), ("rain.coverage", 0.9),
                       ("rain.darkness", 0.35), ("rain.wind", 4.0), ("snow.line", 4400.0),
                       ("snow.transition", 250.0), ("snow.cover", 0.4)):  # fmt: skip
        s.set(pid, value)
    key(scene, "rain.intensity", 0.0, 0.6)
    key(scene, "rain.intensity", CLEAR[0], 0.5)
    key(scene, "rain.intensity", CLEAR[1], 0.03)
    key(scene, "rain.cloud_density", 0.0, 0.9)
    key(scene, "rain.cloud_density", CLEAR[0], 0.85)
    key(scene, "rain.cloud_density", CLEAR[1], 0.0)
    key(scene, "rain.visibility_km", 0.0, 7.0)
    key(scene, "rain.visibility_km", CLEAR[0], 8.0)
    key(scene, "rain.visibility_km", CLEAR[1], 45.0)

    # --- the camera: south of the camp, a slow push up the glacier, a slight tilt up ---------
    summit = hike.enu(*K2)
    start_eye = hike.enu(35.7330, 76.5120, 4600.0)
    start_eye[2] = max(start_eye[2], hike.ground_z(start_eye)) + 450.0
    end_eye = start_eye + (summit - start_eye) * 0.012  # (~190 m closer)
    low = hike.enu(35.8780, 76.5134, 6800.0)  # K2's lower slopes: the summit high in the frame
    shots = []
    t = 0.0
    while t <= DURATION + 1e-9:
        u = smoother(t / DURATION)
        target = low + (summit - low) * (0.35 + 0.35 * u)
        shots.append((round(t, 4), start_eye + (end_eye - start_eye) * u, target))
        t += 1.0 / 30.0
    key_shots(scene, shots)

    scene.set_pois([poi("concordia", hike.end(5), "tent", 44.0)])  # (the trek's end)
    show(scene, "concordia", 0.0)
    save(scene, "s11_k2")


if __name__ == "__main__":
    main()
