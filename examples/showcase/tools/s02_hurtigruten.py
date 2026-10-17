"""Showcase 2: the Hurtigruten northbound - six days along the coast of Norway (~34 s).

A calm map view, north up, with a mini-map of the voyage: three chapters closer (~9 km), the
long legs in between from high up (~130 km):
1. Bergen in the evening drizzle: scattered low cloud over the fjords.
2. Into the Lofoten wall across the Vestfjord in the evening sun.
3. Midnight off Finnmark: the sun stays above the horizon in the north, the sea glitters.
The ship's track draws in its real time; the clock shows that the sun never sets.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import (
    Interp,
    gaussian,
    key,
    key_shots,
    load_hike,
    local,
    map_shots,
    new_scene,
    poi,
    save,
    show,
    track_point_near,
)

ORIGIN = (66.0, 13.0)
DURATION = 34.0
PORTS = {"bergen": (60.399, 5.318), "bodo": (67.288, 14.392), "stamsund": (68.124, 13.844),
         "svolvaer": (68.233, 14.568), "tromso": (69.647, 18.96), "skjervoy": (70.033, 20.98),
         "oksfjord": (70.239, 22.35), "honningsvag": (70.98, 25.97),
         "kirkenes": (69.727, 30.045)}  # fmt: skip


def main() -> None:
    hike = load_hike("s02_hurtigruten", ORIGIN)
    path = hike.path
    scene = new_scene(hike, DURATION)
    scene.origin = ORIGIN
    s = scene.store
    s.set("progress.mode", "distance")
    s.set("sun.follow_track", True)
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.82, 0.25))
    s.set("tracks.width", 3.5)
    s.set("tracks.casing", 2.0)
    s.set("trackgroup.tracks.marker", "ship")
    s.set("trackgroup.tracks.marker_size", 46.0)
    s.set("water.enabled", True)
    s.set("water.wind", 6.0)
    s.set("water.uniform", 0.85)  # (the imagery's sea showed tile seams from high up)
    s.set("haze.aerial", 0.7)
    for pid, value in (("stats.visible", True), ("stats.show_time", True),
                       ("stats.show_ascent", False), ("stats.show_elevation", False),
                       ("stats.profile", False), ("stats.show_speed", True),
                       ("stats.speed_unit", "knots")):  # fmt: skip
        s.set(pid, value)
    s.set("labels.visible", True)
    s.set("labels.peaks", False)
    s.set("labels.passes", False)
    s.set("labels.max_count", 5)
    s.set("labels.max_distance", 80.0)

    d = {name: track_point_near(hike, 1, hike.enu(lat, lon, 0.0))[1]
         for name, (lat, lon) in PORTS.items()}  # fmt: skip
    total = path.total_m
    # (timeline time, path distance, camera: 0 close beside the ship .. 1 high above)
    marks = [
        (0.0, d["bergen"] + 1500.0, 0.0), (4.5, d["bergen"] + 9000.0, 0.0),
        (6.5, d["bergen"] + 60e3, 1.0), (11.0, d["bodo"] - 40e3, 1.0),
        (12.5, d["bodo"] + 10e3, 0.2), (17.0, d["svolvaer"] - 15e3, 0.2),
        (19.0, d["tromso"], 1.0), (21.0, d["skjervoy"] + 15e3, 0.05),
        (27.5, d["oksfjord"] - 15e3, 0.05), (29.5, d["honningsvag"], 1.0),
        (33.0, d["kirkenes"], 1.0),
    ]  # fmt: skip
    mt, md, mh = (np.array(v, dtype=float) for v in zip(*marks, strict=True))
    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    dist_curve = gaussian(np.interp(times, mt, md), 0.6 * 30)
    high = np.clip(gaussian(np.interp(times, mt, mh), 0.8 * 30), 0.0, 1.0)

    def head_of(t: float) -> float:
        return float(np.interp(t, times, dist_curve))

    for t in times[::2]:
        key(scene, "progress.head", round(float(t), 4),
            path.progress_for_distance(min(head_of(t), total - 1.0)), Interp.LINEAR)  # fmt: skip
    # the sun from the recorded time at the head, smoothed: the head runs past the port stops
    # (3.5 h in Trondheim) in a few frames, and the light jumped with it
    s.set("sun.follow_track", False)
    stamps = np.array([path.time_at(min(head_of(t), total - 1.0)) for t in times])
    stamps = gaussian(stamps, 0.8 * 30)
    from earthling.core.sun import solar_position

    for t, stamp in zip(times[::2], stamps[::2], strict=True):
        key(scene, "sun.datetime", round(float(t), 4), local(float(stamp)), Interp.LINEAR)
    # the light summer night: in June the sky stays bright with the sun a few degrees below
    # the horizon (civil twilight), the light model darkens too early - eyes adapted
    for t, stamp in zip(times[::15], stamps[::15], strict=True):
        lon, lat = hike.lonlat(path.position_at(min(head_of(t), total - 1.0)))
        elev = solar_position(dt.datetime.fromtimestamp(float(stamp), dt.UTC), lat, lon).elevation
        boost = float(np.clip((3.0 - elev) / 9.0, 0.0, 1.0))
        key(scene, "post.exposure", round(float(t), 4), 3.6 * boost, Interp.LINEAR)
        key(scene, "light.night_ambient", round(float(t), 4), 1.0 + 9.0 * boost, Interp.LINEAR)

    # a calm map camera, north up: no turns, only the zoom changes between the chapters
    # (close: ~9 km away, the ship and the coast; high: ~130 km, the coastline like on a map)
    def distance_of(t: float) -> float:
        h = float(np.interp(t, times, high))
        return float(np.exp(np.log(9_000.0) + (np.log(130_000.0) - np.log(9_000.0)) * h))

    def pitch_of(t: float) -> float:
        return -52.0 - 12.0 * float(np.interp(t, times, high))

    shots = map_shots(hike, times, head_of, distance_of=distance_of, pitch_of=pitch_of,
                      heading_of=lambda t: 10.0, sigma_s=1.2, fit=0.25)  # fmt: skip
    key_shots(scene, shots)
    # less haze from high up (milky from 70 km): the coast stays readable
    for t, h in zip(times[::15], high[::15], strict=True):
        key(scene, "haze.aerial", round(float(t), 4), 0.5 - 0.3 * float(h), Interp.LINEAR)
    # the whole voyage on a mini-map: where along the coast the ship is
    s.set("minimap.visible", True)

    # weather: drizzle and low cloud around Bergen, clearing northwards
    s.set("weather.enabled", True)
    s.set("weather.layer1.enabled", False)
    p = "weather.layer2."
    for pid, value in (("enabled", True), ("base", 550.0), ("thickness", 500.0),
                       ("scale", 3000.0), ("wind_speed", 8.0), ("wind_dir", 60.0)):  # fmt: skip
        s.set(p + pid, value)
    # (only around Bergen, scattered and thin: the camera stays above them, no crossing)
    key(scene, p + "density", 0.0, 0.35)
    key(scene, p + "density", 3.0, 0.35)
    key(scene, p + "density", 5.5, 0.0)
    key(scene, p + "coverage", 0.0, 0.45)
    key(scene, p + "coverage", 5.5, 0.3)
    for pid, value in (("rain.enabled", True), ("rain.cloud_base", 550.0), ("rain.coverage", 0.0),
                       ("rain.visibility_km", 12.0), ("rain.wind", 4.0)):  # fmt: skip
        s.set(pid, value)
    key(scene, "rain.intensity", 0.0, 0.3)
    key(scene, "rain.intensity", 4.5, 0.25)
    key(scene, "rain.intensity", 6.5, 0.0)
    key(scene, "rain.wetness", 0.0, 0.0)  # (the sea is wet anyway)

    pois = [poi(name, hike.enu(lat, lon, 0.0), "ferry", 34.0) for name, (lat, lon) in PORTS.items()]
    scene.set_pois(pois)
    curve = scene.animation.curves["progress.head"]
    for name in PORTS:
        f = d[name] / total
        t = next((t for t in times if curve.evaluate(t) >= f - 1e-4), None)
        show(scene, name, None if t is None or t <= 0.1 else round(float(t), 3))
    save(scene, "s02_hurtigruten")


if __name__ == "__main__":
    main()
