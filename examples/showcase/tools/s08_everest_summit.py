"""Showcase 8c: the summit push, above a thunderstorm (~4.5 s, the trailer's finale).

Blue hour before sunrise: a storm deck fills the valleys below 5900 m, lightning flashes inside
it, the South Col route glows its way from the Western Cwm up the Lhotse Face and the
south-east ridge towards the summit as the first light touches it. The camera rises in one arc
from over the Western Cwm; just before the line arrives it pulls away, fast, out to the globe
(Everest on the dawn line).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import Interp, draw_keys, key, key_shots, load_hike, new_scene, save, smoother

from earthling.render.atmosphere import PLANET_RADIUS

DURATION = 4.5
ARRIVE = 4.35  # the head reaches the summit (the trailer's final hit)
SUMMIT = (27.98806, 86.92521)
# the camera around the summit: (compass bearing from the summit, distance m, height m)
START = (275.0, 7500.0, 7300.0)  # over the Khumbu Icefall, looking up the Western Cwm
END = (215.0, 3800.0, 9300.0)  # above the Nuptse ridge, the south-east ridge in view
LOOK_FROM = (27.9720, 86.9180, 6900.0)  # the Lhotse Face at the head of the Cwm
LOOK_TO = (27.98806, 86.92521, 8350.0)  # just under the summit: it stands high in the frame
ZOOM = 3.35  # the pull out to the globe starts ...
GLOBE_M = 9.0e6  # ... and ends this far from the summit (the whole Earth in view)
TILT = 18.0  # the globe seen this many degrees off the vertical, from the arc's side


def around(hike, bearing: float, distance: float, height: float) -> np.ndarray:
    lat, lon = SUMMIT
    b = np.radians(bearing)
    return hike.enu(lat + distance * np.cos(b) / 111_320.0,
                    lon + distance * np.sin(b) / (111_320.0 * np.cos(np.radians(lat))),
                    height)  # fmt: skip


def main() -> None:
    hike = load_hike("s08_everest")
    scene = new_scene(hike, DURATION)
    s = scene.store
    # the trek drawn (dim), the summit push drawn up during the shot (bright, with its head)
    s.set("progress.mode", "distance")
    s.set("trackgroup.flight.visible", False)
    for group, colour in (("trek", (1.0, 0.42, 0.12)), ("summit", (1.0, 0.68, 0.26))):
        s.set(f"trackgroup.{group}.color_mode", "single")
        s.set(f"trackgroup.{group}.color", colour)
    s.set("trackgroup.trek.marker", "dot")
    s.set("trackgroup.summit.marker", "dot")
    s.set("tracks.width", 5.0)
    s.set("tracks.emissive", 1.0)
    s.set("tracks.glow", 3.5)
    s.set("tracks.casing", 2.0)
    s.set("marker.visible", True)
    s.set("marker.size", 14.0)
    s.set("stats.visible", False)
    d0, d1 = hike.path.track_range(hike.day_index("summit_push"))
    draw_keys(scene, hike, d0 + 0.35 * (d1 - d0), d1 - 1.0, 0.0, ARRIVE, realtime=False)

    # blue hour, then (faster and faster) the first light on the summit and the cloud tops
    s.set("sun.follow_track", False)
    key(scene, "sun.datetime", 0.0, dt.datetime(2026, 10, 24, 5, 52), Interp.LINEAR)
    key(scene, "sun.datetime", 2.4, dt.datetime(2026, 10, 24, 5, 56), Interp.EASE_IN)
    key(scene, "sun.datetime", DURATION, dt.datetime(2026, 10, 24, 6, 5))
    s.set("sun.intensity", 2.4)
    s.set("sun.color", (1.0, 0.72, 0.5))
    s.set("light.ambient", 0.35)
    s.set("post.exposure", 0.4)
    s.set("sky.rayleigh", 1.6)
    s.set("sky.haze", 2.0)
    s.set("haze.aerial", 0.9)
    s.set("glow.intensity", 1.5)
    s.set("view.fov", 55.0)
    s.set("night.lights", 1.5)  # (the night side of the globe at the end)

    # the storm: the rain clouds as a deck below 5800 m (the lightning starts at their base, so
    # it flashes inside them; no rain, no rain haze) and a few towers rising out of it
    s.set("weather.enabled", True)
    s.set("weather.cloud_shadows", 0.4)
    s.set("weather.layer1.enabled", False)
    s.set("weather.layer3.enabled", False)
    p = "weather.layer2."
    for pid, value in (("enabled", True), ("base", 5200.0), ("thickness", 2600.0),
                       ("coverage", 0.3), ("density", 0.6), ("scale", 3000.0),
                       ("wind_speed", 14.0), ("wind_dir", 80.0), ("variation", 0.5),
                       ("brightness", 0.9)):  # fmt: skip
        s.set(p + pid, value)
    for pid, value in (("enabled", True), ("intensity", 0.0), ("visibility_km", 50.0),
                       ("cloud_base", 4200.0), ("cloud_thickness", 1600.0), ("coverage", 0.97),
                       ("cloud_density", 1.4), ("darkness", 0.35), ("wetness", 0.0)):  # fmt: skip
        s.set("rain." + pid, value)
    s.set("lightning.rate", 50.0)
    s.set("lightning.radius_km", 18.0)
    s.set("lightning.intensity", 1.2)  # (at blue hour a flash lights the whole scene)
    s.set("lightning.branching", 0.5)
    s.set("lightning.seed", 19)  # (strikes at 1.4, 2.7 and 3.6 s)
    # (the cloud layers cover the whole Earth: they dissolve as the camera pulls away)
    for pid, value in (("rain.cloud_density", 1.4), ("weather.layer2.density", 0.6),
                       ("lightning.rate", 50.0)):  # fmt: skip
        key(scene, pid, ZOOM, value)
        key(scene, pid, ZOOM + 0.5, 0.0)
    # (and the blue-hour exposure and the heavy dawn air would wash out the day side)
    for pid, value, end in (("post.exposure", 0.4, -0.7), ("sky.haze", 2.0, 1.0),
                            ("sky.rayleigh", 1.6, 1.0), ("haze.aerial", 0.9, 0.5)):  # fmt: skip
        key(scene, pid, ZOOM + 0.2, value)
        key(scene, pid, DURATION, end)

    # the great peaks: Everest's label as the line arrives
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.passes", False)
    s.set("labels.huts", False)
    s.set("labels.min_elevation", 5000.0)
    s.set("labels.min_prominence", 2000.0)
    s.set("labels.max_count", 3)
    s.set("labels.max_distance", 60.0)
    s.set("labels.size", 20.0)
    key(scene, "labels.opacity", 0.0, 0.0)
    key(scene, "labels.opacity", 2.4, 0.0, Interp.EASE_IN_OUT)
    key(scene, "labels.opacity", 3.6, 1.0)

    # --- the camera: one rising arc around the summit pyramid ------------------------------
    look_from, look_to = hike.enu(*LOOK_FROM), hike.enu(*LOOK_TO)
    summit = hike.enu(*SUMMIT, 8849.0)
    up = summit + (0.0, 0.0, PLANET_RADIUS)  # (the local vertical at the summit)
    up /= np.linalg.norm(up)
    shots = []
    for t in np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0):
        u = smoother(t / DURATION)
        bearing, distance, height = (a + (b - a) * u for a, b in zip(START, END, strict=True))
        eye = around(hike, bearing, distance, height)
        target = look_from + (look_to - look_from) * smoother(min(1.0, t / (0.85 * DURATION)))
        if t > ZOOM:
            # the pull-out: the distance grows exponentially, the view turns down onto the
            # summit; it starts gently from the arc's own motion
            g = smoother((t - ZOOM) / (DURATION - ZOOM))
            arm = eye - summit
            d0 = float(np.linalg.norm(arm))
            side = arm - np.dot(arm, up) * up
            side /= np.linalg.norm(side)
            tilt = np.radians(TILT)
            final = up * np.cos(tilt) + side * np.sin(tilt)
            direction = arm / d0 + (final - arm / d0) * g
            direction /= np.linalg.norm(direction)
            eye = summit + direction * d0 * (GLOBE_M / d0) ** g
            target = target + (summit - target) * g
        shots.append((round(float(t), 4), eye, target))
    key_shots(scene, shots)
    clearance = min(eye[2] - hike.ground_z(eye) for t, eye, _ in shots if t <= ZOOM)
    print(f"lowest clearance {clearance:.0f} m")
    save(scene, "s08_everest_summit")


if __name__ == "__main__":
    main()
