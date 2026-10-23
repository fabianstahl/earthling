"""Showcase 8b: sunrise over Everest (~7 s, the trailer's finale).

Dawn in October, seen from the west (above the Gokyo valley): Everest and Lhotse stand dark
against the glowing sky, the sun breaks out behind Lhotse's shoulder during the shot, a sea of
clouds fills every valley below. The camera glides east towards the South Col.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import Interp, key, key_shots, load_hike, new_scene, save, smoother

DURATION = 7.0


def main() -> None:
    hike = load_hike("s08_everest")
    scene = new_scene(hike, DURATION)
    s = scene.store
    # the whole trek drawn, glowing; the flight line hidden
    s.set("progress.mode", "distance")
    s.set("progress.head", 1.0)
    s.set("trackgroup.flight.visible", False)
    s.set("trackgroup.summit.visible", False)
    s.set("trackgroup.trek.color_mode", "single")
    s.set("trackgroup.trek.color", (1.0, 0.46, 0.14))
    s.set("trackgroup.trek.marker", "dot")
    s.set("tracks.width", 5.0)
    s.set("tracks.emissive", 1.0)
    s.set("tracks.glow", 3.0)
    s.set("tracks.casing", 2.0)
    s.set("marker.visible", False)
    s.set("stats.visible", False)
    # dawn: the sun just above the horizon in the east-south-east (Nepal, UTC+5:45)
    s.set("sun.follow_track", False)
    key(scene, "sun.datetime", 0.0, dt.datetime(2026, 10, 19, 6, 12), Interp.LINEAR)
    key(scene, "sun.datetime", DURATION, dt.datetime(2026, 10, 19, 6, 22), Interp.LINEAR)
    s.set("sun.intensity", 2.2)
    s.set("sun.color", (1.0, 0.82, 0.62))
    s.set("light.ambient", 0.22)  # the faces towards the camera in shadow: silhouettes
    s.set("post.exposure", -0.1)
    s.set("sky.rayleigh", 1.8)  # denser air: a redder low sun, a deeper dawn sky
    s.set("sky.illuminance", 13.0)
    s.set("sky.haze", 2.4)  # (a warm glow around the low sun)
    s.set("sky.sun_disc", 2.5)
    s.set("haze.aerial", 1.1)
    s.set("glow.intensity", 1.4)
    # a sea of clouds up to ~5000 m: the lower valleys vanish, Base Camp stays above it
    s.set("weather.enabled", True)
    s.set("weather.cloud_shadows", 0.6)
    for layer, (on, base, thick, cover, dens, scale, wind) in enumerate((
        (True, 4600.0, 800.0, 0.98, 1.5, 2600.0, 4.0),
        (False, 7000.0, 600.0, 0.3, 0.3, 6000.0, 10.0),
        (True, 9800.0, 900.0, 0.22, 0.12, 12000.0, 25.0),
    ), start=1):  # fmt: skip
        p = f"weather.layer{layer}."
        s.set(p + "enabled", on)
        s.set(p + "base", base)
        s.set(p + "thickness", thick)
        s.set(p + "coverage", cover)
        s.set(p + "density", dens)
        s.set(p + "scale", scale)
        s.set(p + "wind_speed", wind)
        s.set(p + "wind_dir", 80.0)
        s.set(p + "variation", 0.35)
        s.set(p + "brightness", 1.15)
    # the great peaks only
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.passes", False)
    s.set("labels.huts", False)
    s.set("labels.min_elevation", 5000.0)
    s.set("labels.min_prominence", 2000.0)  # (the giants only: Everest, Makalu)
    s.set("labels.max_count", 3)
    s.set("labels.max_distance", 60.0)
    s.set("labels.size", 19.0)
    key(scene, "labels.opacity", 0.0, 0.0)
    key(scene, "labels.opacity", 1.6, 0.0, Interp.EASE_IN_OUT)
    key(scene, "labels.opacity", 3.0, 1.0)
    s.set("view.fov", 50.0)

    # --- the camera: from the south-west, rising out of the cloud tops towards the summit ----
    south_col = hike.enu(27.9730, 86.9300, 7906.0)
    start_eye = hike.enu(27.9880, 86.7500, 7000.0)
    end_eye = hike.enu(27.9840, 86.7900, 7450.0)
    shots = []
    for t in np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0):
        u = smoother(t / DURATION)
        eye = start_eye + (end_eye - start_eye) * u
        shots.append((round(float(t), 4), eye, south_col.copy()))
    key_shots(scene, shots)
    save(scene, "s08_everest_sunrise")


if __name__ == "__main__":
    main()
