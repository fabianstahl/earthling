"""Showcase 6: eight days around Iceland - four seasons in a day (~54 s).

The ring road by camper (dotted, a car at the head) with short hikes in between (solid, a hiker),
each hike given its own few seconds while the drives run by distance; at every change between
driving and hiking the head pauses briefly. A calm map camera, north up, with a mini-map: high
over the drives (the island's shape readable), closer at the hikes, the head and the next stop
always in the picture. The cloud changes day by day: fair weather, a cloudy day at Skógafoss,
low cloud in the canyon, sun on the glacier lagoon, cloud in the east fjords, a clear day at
Dettifoss, low cloud in the north, cloud again back to Reykjavík. Photos of the sights (public
domain, photos/CREDITS.md) appear when the line gets there; a still second at both ends.
"""

from __future__ import annotations

import numpy as np
from sc_common import (
    SHOWCASE,
    Interp,
    daylight_keys,
    gaussian,
    key,
    key_shots,
    load_hike,
    map_shots,
    new_scene,
    poi,
    save,
    show,
)

from earthling.core.photos import Photo

# (file in s06_iceland/photos, lat, lon): where they were taken, in the order of the trip
PHOTOS = [("gullfoss", 64.3271, -20.1199), ("seljalandsfoss", 63.6156, -19.9886),
          ("skogafoss", 63.5321, -19.5114), ("fjadrargljufur", 63.7714, -18.1717), ("jokulsarlon", 64.0784, -16.2306),
          ("dettifoss", 65.8146, -16.3849), ("godafoss", 65.6828, -17.5502)]  # fmt: skip
# alternating places on the screen: the stats panel is bottom left, the mini-map top right
SLOTS = [(0.19, 0.27, -2.5), (0.81, 0.71, 2.0)]  # (x, y, rotation)

DURATION = 52.0
START, END = 1.0, 50.5
HIKE_S = 2.6  # seconds per hike
PAUSE_S = 0.6  # the head waits at every change between driving and hiking


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
    s.set("haze.aerial", 0.35)  # (from 100 km the island turned blue)
    s.set("water.enabled", True)
    s.set("water.wind", 9.0)
    s.set("water.uniform", 0.85)  # (seams between the imagery tiles at sea)
    s.set("stats.visible", True)
    s.set("stats.show_ascent", False)
    s.set("stats.show_elevation", False)
    s.set("stats.profile", False)
    s.set("labels.visible", True)
    s.set("labels.peaks", False)
    s.set("labels.passes", False)
    s.set("labels.max_count", 6)
    s.set("labels.max_distance", 300.0)
    s.set("minimap.visible", True)

    # --- the timeline: drives by distance, every hike a fixed few seconds ----------------
    tracks = path.tracks
    hikes = [g for g in tracks if g.track.group == "hike"]
    drive_m = sum(g.length_m for g in tracks if g.track.group == "drive")
    drive_s = (END - START) - HIKE_S * len(hikes) - PAUSE_S * (len(tracks) - 1)
    marks = [(START, 0.0)]  # (timeline time, global distance)
    for i, g in enumerate(tracks):
        if i > 0:  # a short pause between the tracks (mode changes, nights)
            marks.append((marks[-1][0] + PAUSE_S, marks[-1][1]))
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

    daylight_keys(scene, hike, times[::2], lambda t: min(head_of(t), path.total_m - 0.5))

    # --- the camera: a calm map view, north up; high over the drives, closer at the hikes --
    on_hike = np.array([1.0 if path.track_at(head_of(t)).track.group == "hike" else 0.0
                        for t in times])  # fmt: skip
    near = np.clip(gaussian(on_hike, 0.7 * 30) * 1.4, 0.0, 1.0)  # (eases the zoom)
    hike_times = [float(np.interp(g.offset_m, md, mt)) for g in hikes]
    hike_starts = [g.enu[0] for g in hikes]

    def distance_of(t: float) -> float:
        w = float(np.interp(t, times, near))
        return float(np.exp(np.log(120_000.0) + (np.log(14_000.0) - np.log(120_000.0)) * w))

    def pitch_of(t: float) -> float:
        return -60.0 + 8.0 * float(np.interp(t, times, near))

    def next_stop(t: float):
        # the next hike's start is in the picture 3 s before the head gets there
        return [q for t0, q in zip(hike_times, hike_starts, strict=True) if t <= t0 <= t + 3.0]

    shots = map_shots(hike, times, head_of, distance_of=distance_of, pitch_of=pitch_of,
                      sigma_s=1.6, fit=0.3, points_of=next_stop)  # fmt: skip
    key_shots(scene, shots)

    # --- the weather of every day ---------------------------------------------------------
    day_start = {}
    for g in tracks:
        day = int(g.track.source.stem.split("day")[1].split("_")[0].split(".")[0])
        t0 = float(np.interp(g.offset_m, md, mt))
        day_start.setdefault(day, t0)
    # seen from high above, the weather is the cloud: fields of cumulus at two heights,
    # changing from day to day (a fog layer and rain haze turned the map grey and speckled)
    s.set("weather.enabled", True)
    s.set("weather.gusts", 0.5)
    s.set("weather.cloud_shadows", 0.7)
    for i, (base, thick, scale, wind) in enumerate(((1200.0, 900.0, 7000.0, 8.0),
                                                     (3000.0, 1400.0, 12000.0, 14.0)), start=1):  # fmt: skip
        p = f"weather.layer{i}."
        for pid, value in (("enabled", True), ("base", base), ("thickness", thick),
                           ("scale", scale), ("wind_speed", wind), ("wind_dir", 60.0),
                           ("variation", 0.85), ("density", 0.25)):  # fmt: skip
            s.set(p + pid, value)
    # (day: low cloud coverage, high cloud coverage) - sun, squall, fog in the canyon, glacier
    # in the sun, rain in the east, clear at Dettifoss, low cloud in the north, drizzle
    weather = {1: (0.05, 0.1), 2: (0.2, 0.25), 3: (0.25, 0.05), 4: (0.02, 0.05),
               5: (0.2, 0.25), 6: (0.02, 0.02), 7: (0.25, 0.1), 8: (0.15, 0.2)}  # fmt: skip
    for day, (low, high_cover) in weather.items():
        t = day_start.get(day)
        if t is None:
            continue
        for pid, value in (("weather.layer1.coverage", low),
                           ("weather.layer2.coverage", high_cover)):  # fmt: skip
            key(scene, pid, max(t - 0.6, 0.0), value)
            key(scene, pid, t + 2.5, value)

    pois = []
    for g in hikes:  # four waterfalls and the canyon of Fjaðrárgljúfur (day 3)
        icon = "viewpoint" if "day3" in g.track.source.stem else "waterfall"
        pois.append(poi(f"hike_{g.track.source.stem}", g.enu[0], icon, 34.0))
    pois.append(poi("reykjavik", tracks[0].enu[0], "flag", 40.0))
    scene.set_pois(pois)
    show(scene, "reykjavik", None)
    for g in hikes:
        show(
            scene,
            f"hike_{g.track.source.stem}",
            round(float(np.interp(g.offset_m, md, mt)) - 1.0, 3),
        )
    # --- photos of the sights: shown when the line reaches the place -----------------------
    folder = SHOWCASE / "s06_iceland" / "photos"
    photos = [Photo(name, name.capitalize(), f"../photos/{name}.jpg", lat, lon)
              for name, lat, lon in PHOTOS]  # fmt: skip
    scene.set_photos(photos)
    busy = [-1.0] * len(SLOTS)  # (until when a place on the screen is taken)
    for photo in photos:
        assert (folder / f"{photo.id}.jpg").exists(), photo.id
        d = float(path.nearest_distances(np.asarray(hike.enu(photo.lat, photo.lon))[None, :2])[0])
        t = float(np.interp(d, md, mt))
        slot = min(range(len(SLOTS)), key=lambda i: (busy[i] > t, busy[i]))
        busy[slot] = t + 3.0 + 2 * 0.6
        x, y, rotation = SLOTS[slot]
        p = f"photo.{photo.id}."
        for pid, value in (("appear", "reached"), ("hold", 3.0), ("x", x), ("y", y),
                           ("height", 0.3), ("rotation", rotation), ("border", 7.0),
                           ("zoom", 1.05)):  # fmt: skip
            s.set(p + pid, value)
        print(f"  photo {photo.id:15s} at {t:5.1f} s, place {slot}")
    save(scene, "s06_iceland", holds=True)


if __name__ == "__main__":
    main()
