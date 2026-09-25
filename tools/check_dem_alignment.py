"""Measure the alignment of a DEM source against the Copernicus GLO-30 baseline.

Bakes the same heightmap tiles from both sources and finds the sub-sample shift that maximises
the correlation of the slope images. A good source shows shifts of about 0 at every zoom.

    python tools/check_dem_alignment.py ign_rgealti 6.87 45.92 10 12 14
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from earthling.core.geo import lonlat_to_tile
from earthling.data.cache import TileCache
from earthling.data.dem import DEM_SOURCES, DemBaker, download_sources, sample_spacing_m


def slope(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a.astype(np.float64))
    return np.hypot(gx, gy)


def shifted(a: np.ndarray, s: float, axis: int) -> np.ndarray:
    k = int(np.floor(s))
    f = s - k
    return (1 - f) * np.roll(a, -k, axis) + f * np.roll(a, -(k + 1), axis)


def best_shift(a: np.ndarray, b: np.ndarray, axis: int) -> tuple[float, float]:
    m = slice(10, -10)
    return max(
        (np.corrcoef(a[m, m].ravel(), shifted(b, s, axis)[m, m].ravel())[0, 1], s)
        for s in np.arange(-2.5, 2.51, 0.0625)
    )


def main() -> int:
    source_id, lon, lat, *zooms = sys.argv[1:]
    lon, lat = float(lon), float(lat)
    cache = TileCache(Path.home() / "earthling-cache")
    reference = DEM_SOURCES["copernicus_glo30"]
    download_sources(reference, reference.files_for_bounds((lon, lat, lon, lat)), cache)
    ref = DemBaker(reference, cache)
    test = DemBaker(DEM_SOURCES[source_id], cache)
    for z in map(int, zooms or (10, 12, 14)):
        tx, ty = lonlat_to_tile(lon, lat, z)
        key = (z, int(tx), int(ty))
        a, b = slope(test.bake_tile(*key)), slope(ref.bake_tile(*key))
        (r_row, s_row), (r_col, s_col) = best_shift(a, b, 0), best_shift(a, b, 1)
        spacing = sample_spacing_m(z, lat)
        print(
            f"z{z:<2} ({spacing:5.1f} m samples): south {s_row:+.2f} ({s_row * spacing:+6.1f} m, "
            f"r={r_row:.2f})  east {s_col:+.2f} ({s_col * spacing:+6.1f} m, r={r_col:.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
