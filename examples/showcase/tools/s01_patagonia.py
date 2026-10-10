"""Showcase 1: from space into Torres del Paine (one continuous dive, ~16 s).

The globe over the South Atlantic, a vertical dive with a slow spin through the atmosphere,
flattening into a glide towards the granite towers while the first day of the W trek draws.
Patagonian wind: cloud layers racing in gusts, their shadows sweeping over the lakes.
Southern summer, late morning: the sun stands in the north.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from sc_common import (
    Interp,
    geo_orbit_eye,
    key,
    load_hike,
    new_scene,
    poi,
    pose,
    save,
    show,
    smoother,
)

ORIGIN = (-50.96, -72.92)  # the scene's frame: at the trek (level there)
DURATION = 16.0


def lerp(a, b, f):
    return a + (b - a) * f


def main() -> None:
    hike = load_hike("s01_patagonia", ORIGIN)
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("sun.datetime", dt.datetime(2026, 1, 15, 10, 45))
    s.set("progress.mode", "distance")
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.45, 0.12))
    s.set("tracks.width", 5.0)
    s.set("tracks.casing", 2.5)
    s.set("tracks.casing_opacity", 0.6)
    s.set("haze.aerial", 0.8)
    s.set("shadows.enabled", True)
    # Patagonian wind: two cloud layers racing from the west, in gusts
    s.set("weather.enabled", True)
    s.set("weather.cloud_shadows", 0.9)
    s.set("weather.gusts", 0.6)
    s.set("weather.gust_period", 6.0)
    s.set("weather.layer1.enabled", False)
    for layer, base, thick, cover, dens, scale, wind in (
        (2, 3100.0, 1100.0, 0.36, 0.35, 3500.0, 35.0),
        (3, 6500.0, 1500.0, 0.2, 0.18, 9000.0, 55.0),
    ):
        p = f"weather.layer{layer}."
        s.set(p + "enabled", True)
        s.set(p + "base", base)
        s.set(p + "thickness", thick)
        s.set(p + "coverage", cover)
        s.set(p + "density", dens)
        s.set(p + "scale", scale)
        s.set(p + "wind_speed", wind)
        s.set(p + "wind_dir", 75.0)  # the westerlies blow towards the east
        s.set(p + "variation", 0.85)  # (bases and tops at many heights)
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.passes", False)
    s.set("labels.min_prominence", 600.0)
    s.set("labels.max_count", 5)
    s.set("labels.max_distance", 25.0)

    # --- the dive: geodetic look-at point, distance (log), heading spin, pitch ----------------
    start = hike.start(1)
    mirador = hike.end(1)
    lon_s, lat_s = hike.lonlat(start)
    lon_m, lat_m = hike.lonlat(mirador)
    look_lat, look_lon = (-50.952, -72.935)  # the Ascencio valley below the towers
    far_lat, far_lon = (-44.0, -62.0)  # South Atlantic coast of Patagonia, seen from space

    def state(t: float):
        u = min(max(t / 13.0, 0.0), 1.0)  # the dive lasts 13 s, then a slow glide
        e = smoother(u)
        # distance: 14 000 km down to 4 km (log), faster in the middle
        dist = math.exp(lerp(math.log(14_000e3), math.log(8_500.0), e))
        g = smoother(min(1.0, u * 1.4))  # the look point arrives before the camera
        lat = lerp(far_lat, look_lat, g)
        lon = lerp(far_lon, look_lon, g)
        heading = lerp(10.0, 292.0, e)  # a slow spin, ending looking west-north-west
        pitch = lerp(-62.0, -22.0, smoother(min(1.0, max(0.0, (u - 0.45) / 0.55))))
        if t > 13.0:  # glide on: closer, the towers rising
            v = smoother((t - 13.0) / 3.0)
            dist = lerp(8_500.0, 7_000.0, v)
            heading = lerp(292.0, 286.0, v)
            pitch = lerp(-22.0, -16.0, v)
        return lat, lon, heading, pitch, dist

    for t in np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0):
        lat, lon, heading, pitch, dist = state(float(t))
        h = hike.data.height_at(lon, lat) or 0.0
        eye, target = geo_orbit_eye(hike, lat, lon, h, heading, pitch, dist)
        ground = hike.ground_z(eye)
        if math.isfinite(ground):
            eye[2] = max(eye[2], ground + 400.0)
        scene.animation.set_key("camera.pose", round(float(t), 4), pose(eye, target),
                                Interp.LINEAR)  # fmt: skip

    # the aerial perspective is milky from far up: less of it in the first half
    for t, v in ((0.0, 0.35), (7.0, 0.4), (11.0, 0.8)):
        key(scene, "haze.aerial", t, v)
    # the clouds fade in slowly during the descent (from about 150 km up to 10 km)
    for layer, density in ((2, 0.35), (3, 0.18)):
        key(scene, f"weather.layer{layer}.density", 0.0, 0.0, Interp.LINEAR)
        key(scene, f"weather.layer{layer}.density", 6.5, 0.0, Interp.EASE_IN_OUT)
        key(scene, f"weather.layer{layer}.density", 10.0, density, Interp.LINEAR)
    # the track draws from the hotel towards the towers while the camera arrives
    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    key(scene, "progress.head", 10.5, 0.0, Interp.LINEAR)
    key(scene, "progress.head", 15.3, 1.0, Interp.EASE_OUT)
    scene.set_pois([poi("hotel", (lat_s, lon_s), "hotel", 44.0),
                    poi("mirador", (lat_m, lon_m), "peak", 44.0)])  # fmt: skip
    show(scene, "hotel", 10.5)
    show(scene, "mirador", 15.0)
    save(scene, "s01_patagonia")


if __name__ == "__main__":
    main()
