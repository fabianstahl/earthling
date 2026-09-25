"""Generate the synthetic demo hike in examples/alps_demo/gpx.

Two days on the Tour du Mont Blanc (Chamonix -> Col de Balme -> Trient -> Fenetre d'Arpette ->
Champex), crossing from France into Switzerland. Waypoints are real places; the path in between
is interpolated with some wiggle, elevation is interpolated between waypoint elevations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DAYS = {
    "day01_chamonix_trient": (
        datetime(2026, 7, 1, 6, 30),
        [
            (45.9237, 6.8694, 1035),  # Chamonix
            (45.9530, 6.8960, 1100),
            (45.9786, 6.9270, 1252),  # Argentiere
            (46.0010, 6.9490, 1453),  # Le Tour
            (46.0275, 6.9594, 2191),  # Col de Balme (border)
            (46.0450, 6.9830, 1850),
            (46.0567, 7.0033, 1279),  # Trient
        ],
    ),
    "day02_trient_champex": (
        datetime(2026, 7, 2, 6, 0),
        [
            (46.0567, 7.0033, 1279),  # Trient
            (46.0470, 7.0250, 1580),
            (46.0340, 7.0520, 2665),  # Fenetre d'Arpette
            (46.0330, 7.0800, 2000),
            (46.0300, 7.1160, 1466),  # Champex
        ],
    ),
}


def densify(waypoints, spacing_m=25.0, rng=None):
    rng = rng or np.random.default_rng(1)
    lat, lon, ele = [], [], []
    for (a_lat, a_lon, a_ele), (b_lat, b_lon, b_ele) in zip(waypoints, waypoints[1:], strict=False):
        d = np.hypot((b_lat - a_lat) * 111_000, (b_lon - a_lon) * 78_000)
        n = max(2, int(d / spacing_m))
        t = np.linspace(0, 1, n, endpoint=False)
        wiggle = np.sin(t * np.pi) * np.sin(t * np.pi * 6) * 0.0012
        lat.extend(a_lat + (b_lat - a_lat) * t + wiggle)
        lon.extend(a_lon + (b_lon - a_lon) * t - wiggle)
        ele.extend(a_ele + (b_ele - a_ele) * t + rng.normal(0, 1.5, n))
    lat.append(waypoints[-1][0])
    lon.append(waypoints[-1][1])
    ele.append(waypoints[-1][2])
    return np.array(lat), np.array(lon), np.array(ele)


def write(path: Path, name: str, start: datetime, lat, lon, ele):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="earthling demo" xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>{name}</name><trkseg>",
    ]
    t = start
    for i in range(len(lat)):
        lines.append(
            f'    <trkpt lat="{lat[i]:.6f}" lon="{lon[i]:.6f}"><ele>{ele[i]:.1f}</ele>'
            f"<time>{t.isoformat()}Z</time></trkpt>"
        )
        t += timedelta(seconds=20)
    lines += ["  </trkseg></trk>", "</gpx>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    out = Path(__file__).resolve().parents[1] / "examples" / "alps_demo" / "gpx"
    out.mkdir(parents=True, exist_ok=True)
    for name, (start, waypoints) in DAYS.items():
        write(out / f"{name}.gpx", name, start, *densify(waypoints))


if __name__ == "__main__":
    main()
