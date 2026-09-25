import numpy as np
import pytest
import shapely
from shapely.geometry import LineString
from shapely.strtree import STRtree

from earthling.core.geo import lonlat_to_tile, tile_bounds_lonlat
from earthling.data.borders import (
    MAX_DISTANCE_UV,
    BorderData,
    clamped_distance_field,
    point_segment_distances,
)
from earthling.data.cache import TileCache


def test_point_segment_distances():
    pts = np.array([[0.0, 1.0], [2.0, 0.0], [-1.0, 0.0], [0.5, 0.0]])
    seg = np.array([[0.0, 0.0, 1.0, 0.0]])
    assert point_segment_distances(pts, seg) == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_bucketed_field_matches_brute_force():
    rng = np.random.default_rng(3)
    seg = rng.random((40, 4))
    size = 64
    field = clamped_distance_field(seg, size, 0.1)
    c = (np.arange(size) + 0.5) / size
    gu, gv = np.meshgrid(c, c)
    brute = point_segment_distances(np.column_stack([gu.ravel(), gv.ravel()]), seg)
    assert field == pytest.approx(np.minimum(brute, 0.1).reshape(size, size), abs=1e-9)


def make_border_data(tmp_path, line):
    data = BorderData(TileCache(tmp_path), datasets=("countries",))
    lines = np.array([line], dtype=object)
    data._trees = {"countries": (STRtree(lines), lines)}
    data._ready.set()
    return data


def test_distance_field_of_a_meridian_through_the_tile(tmp_path):
    z = 12
    tx, ty = lonlat_to_tile(7.0, 46.0, z)
    x, y = int(tx), int(ty)
    west, south, east, north = tile_bounds_lonlat(z, x, y)
    mid = (west + east) / 2
    data = make_border_data(tmp_path, LineString([(mid, south - 1), (mid, north + 1)]))
    field = data.distance_field(z, x, y)
    assert field.shape == (256, 256, 2)
    col = field[128, :, 0]
    assert col.min() < 0.01  # the line runs through the centre column
    assert col[0] == pytest.approx(MAX_DISTANCE_UV)  # clamped far away
    assert field[:, 128, 0].max() < 0.01  # every row crosses the meridian
    assert (field[..., 1] == MAX_DISTANCE_UV).all()  # no regional data
    # far away tile: nothing
    assert data.distance_field(z, x + 20, y) is None


def test_loading_real_file_format(tmp_path):
    import json

    data = BorderData(TileCache(tmp_path), datasets=("countries",))
    from earthling.data.borders import dataset_path

    path = dataset_path(data.cache, "countries")
    path.parent.mkdir(parents=True)
    features = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[6, 45], [7, 46]]}},
        {"type": "Feature", "geometry": None},  # Natural Earth has such entries
        {"type": "Feature", "geometry": {"type": "MultiLineString",
                                          "coordinates": [[[8, 45], [8, 46]], [[9, 45], [9, 46]]]}},
    ]  # fmt: skip
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    data.allow_download = False
    assert data.wait_ready(10)
    lines = data.lines_in("countries", (5, 44, 10, 47))
    assert len(lines) == 3 and all(shapely.get_type_id(g) == 1 for g in lines)
