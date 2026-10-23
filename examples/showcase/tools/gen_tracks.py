"""Generate the synthetic GPX tracks of the ten showcase scenes.

    python -m uv run python examples/showcase/tools/gen_tracks.py [s01 s02 ...]

Every track follows real paths, roads or waterways (see gen_common) and has plausible times;
the cyclist and the runners also have a heart rate. The files are committed, so the scenes do
not depend on the routing services.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_common import (  # noqa: E402
    SHOWCASE,
    bike_times,
    cumulative,
    great_circle,
    heart_rate,
    heights,
    local,
    osm_ways,
    place,
    resample,
    route,
    run_times,
    schedule_times,
    smooth,
    smooth_line,
    thin_by_time,
    tobler_times,
    trail,
    water_route,
    write_gpx,
)


def join_legs(legs):
    """One line through consecutive legs: (points, index of every stop)."""
    pts, stops_at = [legs[0]], [0, len(legs[0]) - 1]
    for leg in legs[1:]:
        pts.append(leg[1:])
        stops_at.append(stops_at[-1] + len(leg) - 1)
    return np.vstack(pts), stops_at


def hike(path, name, stops, start, factor=1.0, breaks=(), every_s=10.0, profile="foot",
         seed=0, spacing=10.0):  # fmt: skip
    if profile == "direct":  # paths the routers do not know: straight between the waypoints
        pts = resample(smooth_line(resample([place(p) for p in stops], spacing), 7), spacing)
    else:
        pts = resample(route(profile, stops), spacing)
    ele = smooth(heights(pts, 13), 5)
    t = tobler_times(pts, ele, start, factor, breaks, seed)
    write_gpx(path, name, *thin_by_time(pts, ele, t, every_s))
    return pts, ele, t


# --- 1 Torres del Paine: the first day of the W trek -------------------------------------------
def s01() -> None:
    out = SHOWCASE / "s01_patagonia" / "gpx"
    hike(out / "w_trek_day1.gpx", "W trek day 1: Las Torres - Mirador Base Torres",
         [(-50.97805, -72.87198), "Refugio Chileno, Torres del Paine", "Mirador Base Torres"],
         local(2026, 1, 15, 7, 30, -3), factor=1.05, breaks=[(0.55, 20), (0.999, 1)])  # fmt: skip


# --- 2 Hurtigruten: Bergen - Kirkenes in six days -------------------------------------------------
PORTS = [  # (quay, arrival "day HH:MM", departure) local time (CEST), day 0 = 20 June
    (
        (60.399, 5.318),  # Bergen
        None,
        "0 20:30",
    ),
    (
        (61.599, 5.032),  # Florø
        "1 02:15",
        "1 02:45",
    ),
    (
        (61.935, 5.113),  # Måløy
        "1 04:30",
        "1 04:45",
    ),
    (
        (62.47, 6.149),  # Ålesund
        "1 09:30",
        "1 12:00",
    ),
    (
        (62.736, 7.158),  # Molde
        "1 17:15",
        "1 18:30",
    ),
    (
        (63.111, 7.735),  # Kristiansund
        "1 22:00",
        "1 22:30",
    ),
    (
        (63.44, 10.41),  # Trondheim
        "2 06:30",
        "2 10:00",
    ),
    (
        (64.862, 11.238),  # Rørvik
        "2 16:45",
        "2 17:15",
    ),
    (
        (65.474, 12.209),  # Brønnøysund
        "2 20:45",
        "2 21:00",
    ),
    (
        (66.02, 12.628),  # Sandnessjøen
        "3 00:45",
        "3 01:15",
    ),
    (
        (66.868, 13.705),  # Ørnes
        "3 09:00",
        "3 09:15",
    ),
    (
        (67.288, 14.392),  # Bodø
        "3 12:30",
        "3 15:00",
    ),
    (
        (68.124, 13.844),  # Stamsund
        "3 19:00",
        "3 19:30",
    ),
    (
        (68.233, 14.568),  # Svolvær
        "3 21:00",
        "3 22:00",
    ),
    (
        (68.565, 14.91),  # Stokmarknes
        "4 01:00",
        "4 01:15",
    ),
    (
        (68.697, 15.413),  # Sortland
        "4 02:45",
        "4 03:00",
    ),
    (
        (68.799, 16.543),  # Harstad
        "4 06:45",
        "4 08:00",
    ),
    (
        (69.23, 17.98),  # Finnsnes
        "4 11:00",
        "4 11:30",
    ),
    (
        (69.647, 18.96),  # Tromsø
        "4 14:30",
        "4 18:30",
    ),
    (
        (70.033, 20.98),  # Skjervøy
        "4 22:30",
        "4 22:45",
    ),
    (
        (70.239, 22.35),  # Øksfjord
        "5 02:30",
        "5 02:45",
    ),
    (
        (70.665, 23.682),  # Hammerfest
        "5 05:15",
        "5 06:00",
    ),
    (
        (70.98, 25.97),  # Honningsvåg
        "5 11:00",
        "5 14:45",
    ),
    (
        (70.945, 27.35),  # Kjøllefjord
        "5 17:15",
        "5 17:30",
    ),
    (
        (71.035, 27.85),  # Mehamn
        "5 19:15",
        "5 19:30",
    ),
    (
        (70.857, 29.09),  # Berlevåg
        "5 21:45",
        "5 22:00",
    ),
    (
        (70.635, 29.72),  # Båtsfjord
        "5 23:30",
        "5 23:59",
    ),
    (
        (70.372, 31.108),  # Vardø
        "6 03:15",
        "6 03:45",
    ),
    (
        (70.073, 29.75),  # Vadsø
        "6 06:45",
        "6 07:15",
    ),
    (
        (69.727, 30.045),  # Kirkenes
        "6 09:00",
        None,
    ),
]


def s02() -> None:
    import datetime as dt

    def when(text):
        if text is None:
            return None
        day, clock = text.split()
        h, m = (int(v) for v in clock.split(":"))
        return local(2026, 6, 20, h, m, 2) + dt.timedelta(days=int(day))

    def sea_leg(a, b):
        for margin in (0.35, 0.8, 1.5):  # e.g. out of the Trondheimsfjord: far to the west
            try:
                return resample(water_route(a, b, zoom=11, margin_deg=margin, shore_m=400.0), 400.0)
            except RuntimeError:
                continue
        raise RuntimeError(f"no sea route {a} - {b}")

    pts, stops_at = join_legs([sea_leg(qa, qb) for (qa, _, _), (qb, _, _)
                               in zip(PORTS[:-1], PORTS[1:], strict=True)])  # fmt: skip
    times = [(when(a) or when(d), when(d) or when(a)) for _, a, d in PORTS]
    t = schedule_times(pts, stops_at, times)
    ele = np.zeros(len(pts))
    write_gpx(SHOWCASE / "s02_hurtigruten" / "gpx" / "hurtigruten_northbound.gpx",
              "Hurtigruten northbound: Bergen - Kirkenes", *thin_by_time(pts, ele, t, 120.0))  # fmt: skip


# --- 3 Stelvio by road bike ---------------------------------------------------------------------
def s03() -> None:
    pts = resample(route("bike", ["Prato allo Stelvio", (46.5287, 10.4531)], snap_ends=False),
                   10.0)  # fmt: skip
    ele = smooth(smooth(heights(pts, 13), 101), 101)  # a road climbs evenly (the DEM does not)
    t = bike_times(pts, ele, local(2026, 7, 12, 7, 0, 2), power_w=245.0, mass=79.0, seed=3)
    hr = heart_rate(t, pts, ele, base=133.0, gain=2300.0, seed=3, speed_gain=1.2)
    write_gpx(SHOWCASE / "s03_stelvio" / "gpx" / "stelvio_climb.gpx",
              "Stelvio: Prato - Passo dello Stelvio", *thin_by_time(pts, ele, t, 5.0, hr))  # fmt: skip


# --- 4 Grand Canyon rim to rim -----------------------------------------------------------------
def s04() -> None:
    # (the routers do not connect the corridor trails: they come straight from OpenStreetMap)
    south = trail("South Kaibab", (36.03, -112.12, 36.12, -112.06), "South Kaibab Trailhead",
                  "Phantom Ranch")  # fmt: skip
    north = trail("North Kaibab", (36.09, -112.12, 36.23, -112.02), "Phantom Ranch",
                  "North Kaibab Trailhead")  # fmt: skip
    pts = resample(np.vstack([south, north[1:]]), 10.0)
    ele = smooth(heights(pts, 13), 5)
    t = tobler_times(pts, ele, local(2026, 8, 20, 5, 0, -7), factor=1.2,
                     breaks=[(0.12, 10), (0.36, 45), (0.65, 25), (0.88, 10)])  # fmt: skip
    write_gpx(SHOWCASE / "s04_grand_canyon" / "gpx" / "rim_to_rim.gpx",
              "Grand Canyon rim to rim: South Kaibab - North Kaibab",
              *thin_by_time(pts, ele, t, 15.0))  # fmt: skip


# --- 5 Kilimanjaro summit night ----------------------------------------------------------------
def s05() -> None:
    # the summit path as mapped in OpenStreetMap: Barafu Camp up the ridge, the switchbacks of
    # the scree slope to Stella Point, then along the crater rim to Uhuru Peak
    pts = osm_ways([627821153, 627821152, 628020731, 5116848, 25908806, 628020733])
    pts = resample(smooth_line(resample(pts, 5.0), 3), 10.0)
    ele = smooth(heights(pts, 13), 5)
    t = tobler_times(pts, ele, local(2026, 2, 14, 0, 5, 3), factor=0.42,
                     breaks=[(0.25, 10), (0.5, 12), (0.8, 15), (0.999, 25)])  # fmt: skip
    write_gpx(SHOWCASE / "s05_kilimanjaro" / "gpx" / "summit_night.gpx",
              "Kilimanjaro summit night: Barafu Camp - Uhuru Peak", *thin_by_time(pts, ele, t, 10.0))  # fmt: skip


# --- 6 Iceland ring road -------------------------------------------------------------------------
ICELAND_DAYS = [  # (day, drive stops, hikes [(stops, start clock)]), September 2026
    (1, ["Reykjavík", "Þingvellir", "Geysir, Iceland", "Gullfoss", "Seljalandsfoss"],
     [([(63.6156, -19.9886), (63.6203, -19.9884), (63.6156, -19.9886)], "17:10")]),
    (2, ["Seljalandsfoss", "Skógafoss"],
     [([(63.5321, -19.5114), (63.5300, -19.5110), (63.5340, -19.5040), (63.5390, -19.4985),
        (63.5450, -19.4935), (63.5390, -19.4990), (63.5340, -19.5045), (63.5300, -19.5115),
        (63.5321, -19.5114)], "09:30")]),
    (2, ["Skógafoss", "Vík í Mýrdal"], []),
    (3, ["Vík í Mýrdal", "Fjaðrárgljúfur"],
     [([(63.7714, -18.1717), (63.7790, -18.1550), (63.7714, -18.1717)], "11:00")]),
    (3, ["Fjaðrárgljúfur", "Skaftafell Visitor Centre"], []),
    (4, ["Skaftafell Visitor Centre"],
     [([(64.0163, -16.9663), "Svartifoss", (64.0163, -16.9663)], "08:30")]),
    (4, ["Skaftafell Visitor Centre", "Jökulsárlón", "Höfn, Iceland"], []),
    (5, ["Höfn, Iceland", "Djúpivogur", "Egilsstaðir"], []),
    (6, ["Egilsstaðir", "Dettifoss"],
     [([(65.8146, -16.3849), (65.8100, -16.3860), (65.8049, -16.3869), (65.8100, -16.3880),
        (65.8146, -16.3849)], "12:30")]),
    (6, ["Dettifoss", "Reykjahlíð"], []),
    (7, ["Reykjahlíð", "Goðafoss", "Akureyri"], []),
    (8, ["Akureyri", "Varmahlíð", "Borgarnes", "Reykjavík"], []),
]  # fmt: skip


def s06() -> None:
    import datetime as dt

    out = SHOWCASE / "s06_iceland" / "gpx"
    clock = local(2026, 9, 14, 8, 30, 0)
    day_start = {d: local(2026, 9, 13 + d, 8, 30, 0) for d in range(1, 9)}
    n = 0
    for day, stops, hikes in ICELAND_DAYS:
        clock = max(clock, day_start[day])
        if len(stops) > 1:
            pts = resample(route("car", stops), 100.0)
            s = cumulative(pts)
            hours = s[-1] / 1000.0 / 68.0 + 0.4 * (len(stops) - 2)  # sightseeing stops
            t = clock.timestamp() + s / s[-1] * hours * 3600.0
            n += 1
            write_gpx(out / "drive" / f"drive_{n:02d}_day{day}.gpx",
                      f"Drive day {day}: {stops[0]} - {stops[-1]}", pts, np.zeros(len(pts)), t)  # fmt: skip
            clock = dt.datetime.fromtimestamp(t[-1], clock.tzinfo)
        for k, (hstops, at) in enumerate(hikes, 1):
            h, m = (int(v) for v in at.split(":"))
            start = max(clock, local(2026, 9, 13 + day, h, m, 0))
            _, _, t = hike(out / "hike" / f"hike_day{day}_{k}.gpx", f"Hike day {day}", hstops,
                           start, factor=1.0, breaks=[(0.5, 8)], seed=day,
                           profile="foot" if day in (1, 4) else "direct")  # fmt: skip
            clock = dt.datetime.fromtimestamp(t[-1], clock.tzinfo) + dt.timedelta(minutes=10)


# --- 7 Berlin marathon: two runs on the same course --------------------------------------------
BERLIN_COURSE = [
    (52.51393, 13.36345), (52.51452, 13.35009), "Alt-Moabit, Berlin", (52.52490, 13.37220),
    (52.52928, 13.39878), "Rosenthaler Platz, Berlin", (52.52760, 13.41940),
    "Strausberger Platz, Berlin", (52.50810, 13.43560), (52.49880, 13.41850),
    "Hermannplatz, Berlin", (52.48940, 13.39550), "Rathaus Schöneberg", (52.47700, 13.33100),
    (52.45600, 13.32100), (52.46400, 13.29500), (52.48400, 13.30100), (52.49760, 13.30100), (52.50130, 13.31820), "Wittenbergplatz, Berlin",
    (52.50150, 13.37530), (52.50980, 13.37670), (52.51130, 13.39050), (52.51650, 13.39050),
    (52.51630, 13.37770),
]  # fmt: skip


def s07() -> None:
    pts = resample(route("foot", BERLIN_COURSE), 8.0)
    ele = smooth(heights(pts, 14), 21)
    print(f"  course {cumulative(pts)[-1] / 1000:.2f} km")
    for name, start, finish, fade, seed in (
        ("run_2025", local(2025, 9, 21, 9, 15, 2), 3 * 3600 + 41 * 60 + 20, 0.05, 1),
        ("run_2024", local(2024, 9, 29, 9, 15, 2), 3 * 3600 + 47 * 60 + 5, 0.10, 2),
    ):
        t = run_times(pts, ele, start, finish, fade, seed)
        hr = heart_rate(t, pts, ele, base=134.0, gain=900.0, seed=seed)
        write_gpx(SHOWCASE / "s07_berlin" / "gpx" / name / f"{name}.gpx",
                  f"Berlin marathon {name[-4:]}", *thin_by_time(pts, ele, t, 5.0, hr))  # fmt: skip


# --- 8 Everest: the Lukla flight and the trek to Base Camp ------------------------------------
LUKLA, PHAKDING, NAMCHE = (27.6868, 86.7298), (27.7398, 86.7129), (27.8048, 86.7120)
EVEREST_VIEW, TENGBOCHE = (27.8167, 86.7236), (27.8361, 86.7638)
DINGBOCHE, LOBUCHE, GORAK_SHEP = (27.8925, 86.8313), (27.9483, 86.8103), (27.9806, 86.8290)
EBC = (28.0026, 86.8528)  # the tents on the Khumbu glacier (Nepal side)
EVEREST_DAYS = [  # (stops, start "HH:MM", factor, breaks, profile), October 2026, UTC+5:45
    ([LUKLA, PHAKDING], "10:30", 0.9, [(0.5, 20)], "foot"),
    ([PHAKDING, NAMCHE], "08:00", 0.75, [(0.4, 30), (0.8, 15)], "foot"),
    ([NAMCHE, EVEREST_VIEW, NAMCHE], "08:30", 0.7, [(0.5, 60)], "foot"),
    ([NAMCHE, TENGBOCHE], "08:00", 0.75, [(0.5, 40)], "foot"),
    ([TENGBOCHE, DINGBOCHE], "08:00", 0.7, [(0.5, 40)], "foot"),
    ([DINGBOCHE, LOBUCHE], "08:00", 0.6, [(0.5, 45)], "foot"),
    (
        [LOBUCHE, GORAK_SHEP, (27.9905, 86.8370), (27.9980, 86.8460), EBC],
        "07:30",
        0.55,
        [(0.45, 50)],
        "direct",
    ),
]


def s08() -> None:
    out = SHOWCASE / "s08_everest" / "gpx"
    # the flight: Kathmandu - Lukla up the valleys, over the passes at ~4 km
    ktm, lukla = place("Tribhuvan International Airport"), LUKLA
    way = [ktm, (27.72, 85.55), (27.66, 85.95), (27.60, 86.30), (27.63, 86.55), (27.66, 86.68),
           lukla]  # fmt: skip
    pts = np.vstack([great_circle(a, b, 500.0)[(1 if i else 0):]
                     for i, (a, b) in enumerate(zip(way[:-1], way[1:], strict=True))])  # fmt: skip
    pts = resample(np.array([*pts]), 200.0)
    s = cumulative(pts)
    f = s / s[-1]
    ground = heights(pts, 11)
    cruise = 4300.0
    ele = np.where(f < 0.25, 1338 + (cruise - 1338) * np.sin(np.pi / 2 * f / 0.25), cruise)
    ele = np.where(f > 0.8, 2860 + (cruise - 2860) * np.sin(np.pi / 2 * (1 - f) / 0.2), ele)
    ele = np.maximum(ele, ground + 300.0 * np.sin(np.pi * f) ** 0.3)
    ele[-1] = 2860.0
    t = local(2026, 10, 12, 6, 30, 5.75).timestamp() + f * 35 * 60
    write_gpx(out / "flight" / "flight_ktm_lukla.gpx", "Flight Kathmandu - Lukla", pts, ele, t)
    for day, (stops, at, factor, breaks, profile) in enumerate(EVEREST_DAYS, 1):
        h, m = (int(v) for v in at.split(":"))
        hike(out / "trek" / f"day{day}.gpx", f"EBC trek day {day}", stops,
             local(2026, 10, 11 + day, h, m, 5.75), factor=factor, breaks=breaks, seed=day,
             profile=profile)  # fmt: skip


def s08_summit() -> None:
    # the South Col route as mapped in OpenStreetMap: from Base Camp (the end of the trek) over
    # the glacier into the Khumbu Icefall, up the Western Cwm and the Lhotse Face to the South
    # Col and along the south-east ridge to the summit; four camps on the way
    route_line = osm_ways([1380301536, 518618562])
    base_camp = np.array([[28.0026, 86.8528]])
    pts = np.vstack([base_camp, route_line])
    pts = resample(smooth_line(resample(pts, 5.0), 3), 10.0)
    ele = smooth(heights(pts, 13), 5)
    t = tobler_times(pts, ele, local(2026, 10, 21, 3, 0, 5.75), factor=0.35,
                     breaks=[(0.3, 900), (0.55, 900), (0.75, 900), (0.88, 600)])  # fmt: skip
    write_gpx(SHOWCASE / "s08_everest" / "gpx" / "summit" / "summit_push.gpx",
              "Everest summit push: Base Camp - South Col - summit",
              *thin_by_time(pts, ele, t, 10.0))  # fmt: skip


# --- 9 Boundary Waters canoe trip ------------------------------------------------------------
CANOE_DAYS = [  # (from, to, start)
    ((47.9875, -91.4993), (48.0397, -91.4560), "09:00"),   # Moose Lake landing - Sucker Lake
    ((48.0397, -91.4560), (48.0700, -91.3560), "08:30"),   # - Knife Lake (west)
    ((48.0700, -91.3560), (48.0393, -91.1820), "08:30"),   # - Kekekabic Lake area
    ((48.0393, -91.1820), (47.9925, -91.3100), "08:30"),   # - Ensign / Vera lakes
    ((47.9925, -91.3100), (47.9875, -91.4993), "08:00"),   # back to Moose Lake
]  # fmt: skip


def s09() -> None:
    from gen_common import _terrarium  # noqa: F401  (heights below)

    out = SHOWCASE / "s09_boundary_waters" / "gpx"
    from earthling.data.cache import TileCache
    from earthling.data.water import WaterData

    water = WaterData(TileCache(Path.home() / "earthling-cache"))
    for day, (a, b, at) in enumerate(CANOE_DAYS, 1):
        pts = water_route(a, b, land_cost=25.0, zoom=14, margin_deg=0.12, shore_m=60.0)
        pts = resample(pts, 15.0)
        wet = water.is_water(pts[:, 1], pts[:, 0]) > 0.5
        wet = smooth(wet.astype(float), 3) > 0.5
        s = cumulative(pts)
        speed = np.where(wet[1:], 4.6, 2.2) / 3.6  # paddling / carrying the canoe
        t = np.concatenate([[0.0], np.cumsum(np.diff(s) / speed)])
        h, m = (int(v) for v in at.split(":"))
        t = local(2026, 9, 18 + day, h, m, -5).timestamp() + t
        t[len(t) // 2 :] += 45 * 60  # lunch
        ele = heights(pts, 13)
        write_gpx(out / f"canoe_day{day}.gpx", f"Canoe day {day}",
                  *thin_by_time(pts, ele, t, 15.0))  # fmt: skip


# --- 10 Trans-Siberian / Trans-Mongolian: Moscow - Beijing ---------------------------------------
STATIONS = [  # (query, arrival, departure) in UTC "day HH:MM", day 0 = 12 January 2026
    ("Yaroslavsky railway station, Moscow", None, "0 20:45"),
    ("Kirov, Russia", "1 05:20", "1 05:40"),
    ("Perm, Russia", "1 13:30", "1 13:55"),
    ("Yekaterinburg, Russia", "1 20:10", "1 20:40"),
    ("Tyumen, Russia", "2 01:30", "2 01:50"),
    ("Omsk, Russia", "2 09:50", "2 10:10"),
    ("Novosibirsk, Russia", "2 17:40", "2 18:15"),
    ("Krasnoyarsk, Russia", "3 07:30", "3 07:55"),
    ("Irkutsk, Russia", "4 03:40", "4 04:10"),
    ("Ulan-Ude, Russia", "4 11:30", "4 12:00"),
    ((50.3914, 106.1158), "4 17:00", "4 19:30"),  # Naushki (the border station)
    ((50.2314, 106.2078), "4 21:00", "4 22:30"),  # Sükhbaatar (the town, not the province)
    ("Ulaanbaatar railway station", "5 05:20", "5 05:50"),
    ("Sainshand, Mongolia", "5 14:30", "5 14:45"),
    ("Zamyn-Üüd", "5 19:30", "5 21:00"),
    ("Erenhot, China", "5 21:30", "6 00:30"),
    ("Datong, China", "6 08:10", "6 08:25"),
    ("Beijing railway station", "6 13:05", None),
]


def s10() -> None:
    import datetime as dt

    def when(text):
        if text is None:
            return None
        day, clock = text.split()
        h, m = (int(v) for v in clock.split(":"))
        return dt.datetime(2026, 1, 12 + int(day), h, m, tzinfo=dt.UTC)

    pts, stops_at = join_legs(
        [
            resample(route("car", [qa, qb]), 2000.0)
            for (qa, _, _), (qb, _, _) in zip(STATIONS[:-1], STATIONS[1:], strict=True)
        ]
    )
    times = [(when(a) or when(d), when(d) or when(a)) for _, a, d in STATIONS]
    t = schedule_times(pts, stops_at, times)
    write_gpx(SHOWCASE / "s10_trans_siberian" / "gpx" / "moscow_beijing.gpx",
              "Trans-Siberian / Trans-Mongolian: Moscow - Beijing", pts, np.zeros(len(pts)), t)  # fmt: skip


SCENES = {"s01": s01, "s02": s02, "s03": s03, "s04": s04, "s05": s05, "s06": s06, "s07": s07,
          "s08": s08, "s08_summit": s08_summit, "s09": s09, "s10": s10}  # fmt: skip

if __name__ == "__main__":
    for key in sys.argv[1:] or list(SCENES):
        print(f"--- {key}")
        SCENES[key]()
