"""Showcase 5: Kilimanjaro summit night (~26 s).

From Barafu Camp (4670 m) at midnight to Uhuru Peak (5895 m) at sunrise, in the GPX's own time.
The camera rises at night through the cloud deck into the starry sky, follows the glowing line
up the switchbacks while the sky turns red, and circles the summit as the sun rises over the
sea of clouds, Mawenzi to the east.
"""

from __future__ import annotations

import math

import numpy as np
from sc_common import (
    Interp,
    blend_shots,
    follow_shots,
    gaussian,
    head_keys_for,
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

DURATION = 26.0
WALK = (3.0, 20.0)  # the summit night is drawn between these times


def main() -> None:
    hike = load_hike("s05_kilimanjaro")
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "time")
    s.set("sun.follow_track", True)
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.62, 0.2))
    s.set("tracks.emissive", 1.0)
    s.set("tracks.glow", 2.2)
    s.set("tracks.width", 4.0)
    s.set("tracks.casing", 1.5)
    s.set("progress.head_boost", 1.5)
    s.set("marker.size", 12.0)
    s.set("marker.glow", 2.2)
    s.set("sky.stars", 2.5)
    s.set("haze.aerial", 0.7)
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.min_prominence", 300.0)
    s.set("labels.max_count", 5)
    # the sea of clouds, about 1000 m thick, far below the summit
    s.set("weather.enabled", True)
    p = "weather.layer1."
    for pid, value in (("enabled", True), ("base", 2500.0), ("thickness", 1100.0),
                       ("coverage", 0.93), ("density", 1.0), ("scale", 5000.0),
                       ("wind_speed", 4.0), ("wind_dir", 250.0), ("detail", 0.6),
                       ("brightness", 1.1)):  # fmt: skip
        s.set(p + pid, value)
    # the clouds stay dim in the night (the raised exposure would make them glow)
    key(scene, p + "brightness", 0.0, 0.25)
    key(scene, p + "brightness", 14.5, 0.25)
    key(scene, p + "brightness", 19.0, 1.1)
    # a readable night (eyes adapted): more night ambient and exposure until the dawn
    for t, ambient, exposure in ((0.0, 4.0, 1.6), (14.5, 4.0, 1.6), (19.0, 1.0, 0.2),
                                 (DURATION, 1.0, 0.0)):  # fmt: skip
        key(scene, "light.night_ambient", t, ambient)
        key(scene, "post.exposure", t, exposure)

    head_of = head_keys_for(scene, hike, 1, *WALK, realtime=True)
    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    summit = hike.end(1)
    barafu = hike.start(1)

    # A: up through the clouds (below the deck, south-east of the camp, looking at the summit)
    def rise(t: float):
        u = smoother(min(t / 4.5, 1.0))
        eye_low = orbit_eye(barafu, 315.0, 6.0, 9000.0)
        eye_low[2] = 3900.0  # just above the cloud deck (below it the night is black)
        eye_high = orbit_eye(barafu, 320.0, -14.0, 3500.0)
        eye = eye_low + (eye_high - eye_low) * u
        look = barafu + (summit - barafu) * (0.5 - 0.4 * u) + np.array([0.0, 0.0, 300.0 * (1 - u)])
        return eye, look

    rising = [(t, *rise(t)) for t in times]
    # B: along the switchbacks, from the side
    chase = follow_shots(hike, times, lambda t: head_of(min(max(t, WALK[0]), WALK[1])),
                         distance=1900.0, pitch=-14.0, side=55.0, window=500.0, sigma_s=1.0,
                         heading_sigma_s=3.0, clearance=150.0)  # fmt: skip
    shots = blend_shots(rising, chase, 3.5, 6.0)

    # C: around the summit at sunrise (from the west, the sun and Mawenzi beyond the crater)
    def circle(t: float):
        u = smoother(min(max((t - 19.0) / (DURATION - 19.0), 0.0), 1.0))
        heading = 40.0 + 55.0 * u  # looking north-east .. east
        eye = orbit_eye(summit + np.array([0.0, 0.0, 150.0]), heading, -12.0 + 4.0 * u,
                        2600.0 - 500.0 * u)  # fmt: skip
        return eye, summit + np.array([0.0, 0.0, 150.0])

    orbit = [(t, *circle(t)) for t in times]
    shots = blend_shots(shots, orbit, 18.0, 21.0)
    # keep the eye clear of the crater rim
    for i, (t, eye, look) in enumerate(shots):
        ground = hike.ground_z(eye)
        if math.isfinite(ground) and eye[2] < ground + 120.0:
            shots[i] = (t, eye + np.array([0.0, 0.0, ground + 120.0 - eye[2]]), look)
    eyes = gaussian(np.array([e for _, e, _ in shots]), 6.0)
    shots = [(t, e, g) for (t, _, g), e in zip(shots, eyes, strict=True)]
    key_shots(scene, shots)

    stella = hike.enu(-3.0782, 37.3626, 5739.0)
    # the summit flag floats above the peak's label (name and elevation), without a pin line
    # through the text
    top = poi("uhuru", summit, "summit", 48.0)
    top.lift_px = 92.0
    scene.set_pois([poi("barafu", barafu, "tent", 40.0), poi("stella", stella, "flag", 36.0),
                    top])  # fmt: skip
    scene.store.set("poi.uhuru.pin", False)
    show(scene, "barafu", None)
    t_stella = next((t for t in times if head_of(t) >= hike.path.total_m * 0.86), WALK[1])
    show(scene, "stella", round(float(t_stella), 3))
    show(scene, "uhuru", WALK[1] - 1.0)
    save(scene, "s05_kilimanjaro")


if __name__ == "__main__":
    main()
