"""Showcase 6: eight days around Iceland - four seasons in a day (~44 s).

The ring road by camper (dotted, a car at the head) with short hikes in between (solid, a hiker),
each hike given its own few seconds while the drives run by distance. A high chase camera that
drops down to every hike and climbs back up. The weather changes every few seconds: sun, a
squall with a rainbow at Skógafoss, fog in the canyon, sun on the glacier lagoon, rain in the
east fjords, a clear day at Dettifoss, low cloud in the north, drizzle back to Reykjavík.
"""

from __future__ import annotations

import numpy as np
from sc_common import (
    Interp,
    follow_shots,
    gaussian,
    key,
    key_shots,
    load_hike,
    new_scene,
    poi,
    save,
    show,
)

DURATION = 44.0
START, END = 1.0, 42.5
HIKE_S = 2.6  # seconds per hike


def main() -> None:
    hike = load_hike("s06_iceland")
    path = hike.path
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("progress.mode", "distance")
    s.set("sun.follow_track", True)
    for group, color, dots, marker in (("drive", (1.0, 0.86, 0.3), True, "car"),
                                       ("hike", (1.0, 0.4, 0.12), False, "hiker")):  # fmt: skip
        p = f"trackgroup.{group}."
        s.set(p + "color_mode", "single")
        s.set(p + "color", color)
        s.set(p + "dots", dots)
        s.set(p + "dash", 11.0 if dots else 0.0)
        s.set(p + "width", 1.0)
        s.set(p + "marker", marker)
        s.set(p + "marker_size", 40.0)
    s.set("tracks.width", 4.5)
    s.set("tracks.casing", 2.0)
    s.set("haze.aerial", 0.7)
    s.set("water.enabled", True)
    s.set("water.wind", 9.0)
    s.set("stats.visible", True)
    s.set("stats.show_ascent", False)
    s.set("stats.show_elevation", False)
    s.set("stats.profile", False)
    s.set("labels.visible", True)
    s.set("labels.peaks", False)
    s.set("labels.passes", False)
    s.set("labels.max_count", 6)
    s.set("labels.max_distance", 60.0)

    # --- the timeline: drives by distance, every hike a fixed few seconds ----------------
    tracks = path.tracks
    hikes = [g for g in tracks if g.track.group == "hike"]
    drive_m = sum(g.length_m for g in tracks if g.track.group == "drive")
    drive_s = (END - START) - HIKE_S * len(hikes)
    marks = [(START, 0.0)]  # (timeline time, global distance)
    for g in tracks:
        span = HIKE_S if g.track.group == "hike" else drive_s * g.length_m / drive_m
        marks.append((marks[-1][0] + span, g.offset_m + g.length_m))
    mt, md = np.array(marks).T

    def head_of(t: float) -> float:
        return float(np.interp(t, mt, md))

    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    for t in times[::2]:
        key(scene, "progress.head", round(float(t), 4),
            path.progress_for_distance(min(head_of(t), path.total_m - 0.5)), Interp.LINEAR)  # fmt: skip

    # --- the camera: high over the drives, down to the hikes ----------------------------
    on_hike = np.array([1.0 if path.track_at(head_of(t)).track.group == "hike" else 0.0
                        for t in times])  # fmt: skip
    near = gaussian(on_hike, 0.8 * 30)  # (eases the descent and the climb)
    shots = []
    # (below the clouds: the weather is seen from inside it, not as a white blanket from above)
    high = follow_shots(hike, times, head_of, distance=8_000.0, pitch=-11.0, side=-25.0,
                        window=12_000.0, sigma_s=1.8, heading_sigma_s=4.0, clearance=700.0)  # fmt: skip
    low = follow_shots(hike, times, head_of, distance=2500.0, pitch=-30.0, side=-30.0,
                       window=600.0, sigma_s=0.6, heading_sigma_s=2.0, clearance=300.0)  # fmt: skip
    for (t, e_hi, g_hi), (_, e_lo, g_lo), w in zip(high, low, near, strict=True):
        w = float(min(1.0, w * 1.6))
        # (log of the distance: no rush through the middle heights)
        f = w ** 0.5
        shots.append((t, e_hi + (e_lo - e_hi) * f, g_hi + (g_lo - g_hi) * w))
    # keep the camera below the cloud deck (2700 m), but above the ground
    eyes = []
    for _t, eye, _look in shots:
        lat, lon, h = hike.frame.enu_to_geodetic(eye)
        ground = hike.data.height_at(float(lon), float(lat)) or 0.0
        cap = max(2300.0, ground + 250.0)
        eyes.append(np.asarray(hike.frame.geodetic_to_enu(lat, lon, min(float(h), cap))))
    eyes = gaussian(np.array(eyes), 20.0)
    shots = [(t, e, g) for (t, _, g), e in zip(shots, eyes, strict=True)]
    key_shots(scene, shots)

    # --- the weather of every day ---------------------------------------------------------
    day_start = {}
    for g in tracks:
        day = int(g.track.source.stem.split("day")[1].split("_")[0].split(".")[0])
        t0 = float(np.interp(g.offset_m, md, mt))
        day_start.setdefault(day, t0)
    s.set("weather.enabled", True)
    s.set("weather.gusts", 0.5)
    for i, (base, thick, scale, wind) in enumerate(((150.0, 250.0, 4000.0, 6.0),
                                                     (2700.0, 900.0, 4000.0, 14.0)), start=1):  # fmt: skip
        p = f"weather.layer{i}."
        for pid, value in (("enabled", True), ("base", base), ("thickness", thick),
                           ("scale", scale), ("wind_speed", wind), ("wind_dir", 60.0)):  # fmt: skip
            s.set(p + pid, value)
    for pid, value in (("rain.enabled", True), ("rain.cloud_base", 2400.0), ("rain.wind", 4.0),
                       ("rain.visibility_km", 8.0), ("rain.coverage", 0.75)):  # fmt: skip
        s.set(pid, value)
    # (day: fog density, cloud coverage, rain intensity, rainbow)
    weather = {1: (0.0, 0.25, 0.0, 0.0), 2: (0.0, 0.55, 0.7, 2.0), 3: (0.55, 0.45, 0.0, 0.0),
               4: (0.0, 0.15, 0.0, 0.0), 5: (0.2, 0.7, 0.8, 0.0), 6: (0.0, 0.2, 0.0, 0.0),
               7: (0.35, 0.65, 0.0, 0.0), 8: (0.15, 0.55, 0.4, 0.0)}  # fmt: skip
    for day, (fog, cover, rain, bow) in weather.items():
        t = day_start.get(day)
        if t is None:
            continue
        for pid, value in (("weather.layer1.density", fog * 1.4),
                           ("weather.layer1.coverage", 0.45 if fog else 0.0),
                           ("weather.layer2.coverage", cover), ("weather.layer2.density", 0.6),
                           ("rain.intensity", rain), ("rain.cloud_density", 1.0 if rain else 0.0),
                           ("rain.rainbow", bow)):  # fmt: skip
            key(scene, pid, max(t - 0.6, 0.0), value)
            key(scene, pid, t + 2.5, value)
    # the squall clears for the rainbow on day 2 (the sun out, the rain still falling ahead)
    t2 = day_start.get(2)
    if t2 is not None:
        key(scene, "rain.coverage", t2 + 1.0, 0.75)
        key(scene, "rain.coverage", t2 + 3.0, 0.35)
        key(scene, "rain.coverage", t2 + 5.0, 0.75)

    pois = []
    for g in hikes:
        pois.append(poi(f"hike_{g.track.source.stem}", g.enu[0], "attraction", 34.0))
    pois.append(poi("reykjavik", tracks[0].enu[0], "flag", 40.0))
    scene.set_pois(pois)
    show(scene, "reykjavik", None)
    for g in hikes:
        show(scene, f"hike_{g.track.source.stem}", round(float(np.interp(g.offset_m, md, mt)) - 1.0, 3))
    save(scene, "s06_iceland")


if __name__ == "__main__":
    main()
