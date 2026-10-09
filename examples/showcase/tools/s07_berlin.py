"""Showcase 7: the Berlin marathon as a ghost race, in portrait (1080 x 1920, ~30 s).

This year's run (2025, 3:41:20) against last year's (2024, 3:47:05), shifted to the same start
(progress by clock): two heads on the course, the gap in the overlay. A helicopter view from
above, slightly ahead of the leader, district names and kilometre flags; an overcast autumn
morning turning to drizzle. The finish: low behind the Brandenburg Gate.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sc_common import (
    Interp,
    blend_shots,
    gaussian,
    key,
    key_shots,
    load_hike,
    new_scene,
    orbit_eye,
    poi,
    save,
    show,
)

DURATION = 30.0
RACE = (1.0, 26.5)  # the clock runs from the start (09:15) to the slower finish
SHIFT_H = (dt.date(2025, 9, 21) - dt.date(2024, 9, 29)).days * 24.0  # 2024 -> 2025


def main() -> None:
    hike = load_hike("s07_berlin")
    path = hike.path
    path.time_offsets = {"run2025": 0.0, "run2024": SHIFT_H * 3600.0}
    path.lead_group = "run2025"
    scene = new_scene(hike, DURATION)
    s = scene.store
    s.set("export.resolution", "1080x1920")
    s.set("view.fov", 62.0)
    s.set("progress.mode", "clock")
    s.set("trackgroup.run2024.time_offset", SHIFT_H)
    for group, color, glow in (("run2025", (1.0, 0.45, 0.1), 1.2),
                               ("run2024", (0.62, 0.7, 0.8), 0.15)):
        p = f"trackgroup.{group}."
        s.set(p + "color_mode", "single")
        s.set(p + "color", color)
        s.set(p + "glow", glow)
    s.set("trackgroup.run2024.opacity", 0.55)  # the ghost: wide and faint below this year's
    s.set("trackgroup.run2024.width", 2.2)
    s.set("trackgroup.run2025.marker", "runner")
    s.set("trackgroup.run2025.marker_size", 52.0)
    s.set("marker.size", 13.0)  # (the 2024 ghost: a glowing dot)
    s.set("tracks.width", 5.0)
    s.set("tracks.casing", 2.0)
    s.set("progress.head_fade", 600.0)
    s.set("haze.aerial", 0.9)
    s.set("shadows.enabled", False)  # overcast: no sun shadows
    s.set("sun.datetime", dt.datetime(2025, 9, 21, 10, 30))
    s.set("light.ambient", 0.9)
    s.set("sun.intensity", 0.35)
    # labels: the districts and places along the course
    s.set("labels.visible", True)
    s.set("labels.peaks", False)
    s.set("labels.passes", False)
    s.set("labels.places", True)
    s.set("labels.max_count", 10)
    s.set("labels.max_distance", 8.0)
    s.set("labels.elevations", False)
    s.set("labels.size", 20.0)
    # overcast autumn morning, drizzle in the second half
    s.set("weather.enabled", True)
    s.set("weather.layer1.enabled", False)
    p = "weather.layer2."
    for pid, value in (("enabled", True), ("base", 4200.0), ("thickness", 700.0),
                       ("coverage", 0.85), ("density", 0.5), ("scale", 4000.0),
                       ("wind_speed", 6.0), ("wind_dir", 80.0), ("brightness", 1.0)):  # fmt: skip
        s.set(p + pid, value)
    s.set("weather.cloud_shadows", 0.3)
    s.set("rain.enabled", True)
    s.set("rain.cloud_base", 4200.0)
    s.set("rain.coverage", 0.0)  # (the drizzle falls from the stratus layer)
    s.set("rain.visibility_km", 9.0)
    s.set("rain.wind", 2.0)
    key(scene, "rain.intensity", 0.0, 0.0)
    key(scene, "rain.intensity", 13.0, 0.0)
    key(scene, "rain.intensity", 17.0, 0.35)
    key(scene, "rain.wetness", 0.0, 0.0)
    key(scene, "rain.wetness", 13.0, 0.0)
    key(scene, "rain.wetness", 19.0, 0.6)
    # stats: pace, heart rate and the gap to last year's run
    for pid, value in (("stats.visible", True), ("stats.position", "top_left"),
                       ("stats.show_day", False), ("stats.show_time", True),
                       ("stats.show_ascent", False), ("stats.show_elevation", False),
                       ("stats.profile", False), ("stats.show_speed", True),
                       ("stats.speed_unit", "pace"), ("stats.show_heart_rate", True),
                       ("stats.show_gap", True), ("stats.scale", 1.15)):  # fmt: skip
        s.set(pid, value)
    key(scene, "stats.text", -1.0, "2025 vs 2024", Interp.STEP)

    # the clock: linear between the start and the slower finish
    key(scene, "progress.head", 0.0, 0.0, Interp.STEP)
    key(scene, "progress.head", RACE[0], 0.0, Interp.LINEAR)
    key(scene, "progress.head", RACE[1], 1.0, Interp.LINEAR)

    def head_of(t: float) -> float:
        u = min(max((t - RACE[0]) / (RACE[1] - RACE[0]), 0.0), 1.0)
        return path.distance_for(u, "clock")

    times = np.arange(0.0, DURATION + 1e-9, 1.0 / 30.0)
    # both heads in the picture: the camera aims between them and backs off with the gap
    ghost = next(g for g in path.tracks if g.track.group == "run2024")

    def ghost_at(t: float):
        u = min(max((t - RACE[0]) / (RACE[1] - RACE[0]), 0.0), 1.0)
        d = path.head_on(ghost, path.clock_at(u))
        return path.position_at(ghost.offset_m + (d or 0.0))

    # the race runs up the portrait frame: the ghost below, the leader above, the camera
    # aiming between them and backing off with the gap
    lead_pos = np.array([path.position_at(head_of(t)) for t in times])
    ghost_pos = np.array([ghost_at(t) for t in times])
    mid = gaussian((lead_pos + ghost_pos) / 2.0, 30.0)
    ahead = [path.position_at(head_of(t) + 400.0) for t in times]
    behind = [path.position_at(max(head_of(t) - 400.0, 0.0)) for t in times]
    course = np.array(ahead) - np.array(behind)
    gap = np.linalg.norm((lead_pos - ghost_pos)[:, :2], axis=1)
    heading = gaussian(np.unwrap(np.arctan2(course[:, 0], course[:, 1])), 60.0)
    # (far enough that 42 km in 25 s do not race through the picture)
    dist = gaussian(np.maximum(3200.0, 2.2 * gap), 45.0)
    heli = []
    for t, c, h, d in zip(times, mid, heading, dist, strict=True):
        eye = orbit_eye(c, float(np.degrees(h)), -60.0, float(d))
        heli.append((float(t), eye, c))
    # the finish: low, east of the Brandenburg Gate, looking west along Unter den Linden
    gate = hike.enu(52.51627, 13.37770, 40.0)
    finish_eye = orbit_eye(gate, 270.0, -18.0, 800.0)
    finish = [(t, finish_eye, gate) for t in times]
    shots = blend_shots(heli, finish, RACE[1] - 2.5, RACE[1] + 1.5)
    key_shots(scene, shots)

    # kilometre flags every 5 km on this year's course
    lead = path.lead()
    pois = []
    for km in range(5, 45, 5):
        i = int(np.searchsorted(lead.ground, km * 1000.0))
        if i >= len(lead.enu):
            break
        pois.append(poi(f"km{km}", lead.enu[i], "flag", 34.0, caption=f"{km} km"))
    pois.append(poi("finish", lead.enu[-1], "finish", 46.0))
    scene.set_pois(pois)
    for p_ in pois:
        show(scene, p_.id, None)
        s.set(f"poi.{p_.id}.caption", True)
        s.set(f"poi.{p_.id}.caption_size", 18.0)
    s.set("poi.finish.caption", False)
    save(scene, "s07_berlin")


if __name__ == "__main__":
    main()
