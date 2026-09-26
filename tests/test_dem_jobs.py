import shapely

from earthling.core.geo import lonlat_to_tile
from earthling.data.jobs import source_area


class FineSource:
    native_resolution_m = 2.0
    coverage_area = None


class CoarseSource:
    native_resolution_m = 30.0
    coverage_area = None


def tiles_around(lon, lat, zooms):
    out = []
    for z in zooms:
        tx, ty = lonlat_to_tile(lon, lat, z)
        out.append((z, int(tx), int(ty)))
    return out


def test_fine_sources_are_only_fetched_for_detailed_tiles():
    area = shapely.box(5.0, 45.0, 9.0, 47.0)
    coarse_tiles = tiles_around(7.0, 46.0, [7, 9, 11])  # ~1.2 km .. 53 m per sample
    fine_tiles = tiles_around(7.5, 46.5, [13])  # ~13 m per sample
    # 2 m: only below the z13 tile (spacing <= 32 m), not the huge low-zoom tiles
    fine = source_area(FineSource(), coarse_tiles + fine_tiles, area)
    assert not fine.is_empty and fine.area < 0.02
    assert fine.intersects(shapely.Point(7.5, 46.5))
    assert source_area(FineSource(), coarse_tiles, area).is_empty
    # 30 m sources are worth it down to ~480 m samples (z9 and finer)
    coarse = source_area(CoarseSource(), coarse_tiles, area)
    assert coarse.area > 0.2
