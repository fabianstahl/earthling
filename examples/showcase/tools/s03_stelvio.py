"""Showcase 3: the Stelvio pass by road bike (~26 s).

From Prato (920 m) up the 48 hairpins of the east ramp to the pass (2757 m). The camera starts
as a map looking straight down, turning with the general direction of the climb (so the
hairpins swing left and right), then tilts into 3D and shows the wall of hairpins below the
pass. Morning fog in the valley thins out as the rider climbs out of it. The line is coloured
by the gradient; the stats show speed, gradient, heart rate and a hairpin counter.
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
)

DURATION = 26.0
T0, T1 = 1.0, 23.0  # the ride is drawn between these times


def hairpins(hike, window: float = 60.0) -> list[float]:
    """Path distances of the hairpin turns: the direction turns by more than 140 degrees
    within +- ``window`` metres."""
    path = hike.path
    ds = np.arange(0.0, path.total_m, 10.0)
    pts = np.array([path.position_at(d) for d in ds])
    heading = np.unwrap(np.arctan2(np.gradient(pts[:, 0]), np.gradient(pts[:, 1])))
    heading = gaussian(heading, 2.0)
    k = int(window / 10.0)
    turn = np.abs(heading[2 * k :] - heading[: -2 * k])
    out, last = [], -1e9
    for i in np.flatnonzero(turn > math.radians(115.0)):
        d = float(ds[i + k])
        if d - last > 150.0:  # one turn, not several samples of it
            out.append(d)
            last = d
    return out


def main() -> None:
    hike = load_hike("s03_stelvio")
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "time")
    s.set("sun.follow_track", True)
    s.set("trackgroup.tracks.marker", "bike")
    s.set("trackgroup.tracks.marker_size", 42.0)
    s.set("tracks.color_by", "gradient")
    s.set("tracks.color_min", -3.0)  # (turbo: easy green, yellow, steep red)
    s.set("tracks.color_max", 12.0)
    s.set("tracks.color_ramp", "turbo")
    s.set("tracks.width", 5.0)
    s.set("tracks.casing", 2.0)
    s.set("progress.head_boost", 0.3)
    s.set("haze.aerial", 0.8)
    for pid in ("stats.visible", "stats.show_speed", "stats.show_gradient",
                "stats.show_heart_rate", "stats.show_time"):  # fmt: skip
        s.set(pid, True)
    s.set("stats.show_day", False)
    s.set("stats.date_format", "long")
    s.set("tracks.elevation", "dem")
    s.set("labels.visible", True)
    s.set("labels.places", False)
    s.set("labels.min_prominence", 500.0)
    s.set("labels.max_count", 6)
    # morning fog in the valley: it sinks and thins out while the rider climbs
    s.set("weather.enabled", True)
    p = "weather.layer1."
    for pid, value in (("enabled", True), ("thickness", 450.0), ("coverage", 0.75),
                       ("scale", 1800.0), ("wind_speed", 1.5), ("detail", 0.7)):  # fmt: skip
        s.set(p + pid, value)
    key(scene, p + "base", 0.0, 1000.0)
    key(scene, p + "base", 16.0, 700.0)
    key(scene, p + "density", 0.0, 0.9)
    key(scene, p + "density", 9.0, 0.6)
    key(scene, p + "density", 18.0, 0.0)

    head_of = head_keys_for(scene, hike, 1, T0, T1, realtime=True)
    total = hike.path.total_m
    # hairpin counter (the extra line of the stats, switching at every turn)
    turns = hairpins(hike)
    print(f"{len(turns)} hairpins found")
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    heads = np.array([head_of(t) for t in times])
    key(scene, "stats.text", -1.0, "", Interp.STEP)
    for n, d in enumerate(turns, 1):
        t = float(times[np.argmax(heads >= d)]) if (heads >= d).any() else DURATION
        key(scene, "stats.text", round(t, 3), f"Hairpin {n} / {len(turns)}", Interp.STEP)

    # camera: a turning map, then 3D
    top = follow_shots(hike, times, head_of, distance=2200.0, pitch=-82.0, side=0.0,
                       window=1500.0, sigma_s=1.2, heading_sigma_s=4.0, clearance=300.0)  # fmt: skip
    tilt = follow_shots(hike, times, head_of, distance=3200.0, pitch=-28.0, side=35.0,
                        window=900.0, sigma_s=1.5, heading_sigma_s=5.0, clearance=250.0)  # fmt: skip
    shots = blend_shots(top, tilt, 9.0, 14.0)
    # the end: above the pass, looking down the wall of hairpins
    pass_top = hike.end(1)
    below = hike.path.position_at(total * 0.88)
    look = pass_top * 0.35 + below * 0.65
    final_eye = orbit_eye(look, 252.0, -27.0, 2400.0)  # from the east, up the wall
    final_eye[2] = max(final_eye[2], hike.ground_z(final_eye) + 350.0)
    end = [(t, final_eye.copy(), look.copy()) for t in times]
    shots = blend_shots(shots, end, 19.5, 25.0)
    key_shots(scene, shots)

    start = hike.start(1)
    scene.set_pois([poi("start", start, "flag", 40.0), poi("pass", pass_top, "pass", 46.0)])
    show(scene, "start", None)
    show(scene, "pass", T1 - 0.3)
    save(scene, "s03_stelvio")


if __name__ == "__main__":
    main()
