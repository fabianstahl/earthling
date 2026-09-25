from pathlib import Path

import numpy as np
import pytest
import shapely

from earthling.core.aoi import compute_aoi, format_bytes, plan_report, plan_tiles
from earthling.core.config import AreaSection, Zone
from earthling.core.geo import tile_bounds_lonlat
from earthling.core.gpx import Segment, Track, load_gpx_folder

AREA = AreaSection(
    border_km=5.0,
    zones=(
        Zone(within_km=1.0, imagery_zoom=15, dem_zoom=12),
        Zone(within_km=5.0, imagery_zoom=12, dem_zoom=10),
    ),
)


def straight_track():
    lat = np.linspace(46.0, 46.2, 50)
    lon = np.full(50, 7.0)
    seg = Segment(
        lat, lon, np.zeros(50), np.full(50, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    )
    return Track("line", Path("x.gpx"), [seg])


def test_aoi_buffer_width_is_metric():
    aoi = compute_aoi([straight_track()], AREA)
    west, _, east, _ = aoi.aoi.bounds
    # 5 km on each side at 46 deg latitude: 1 deg lon ~ 77.3 km
    assert (east - west) * 77_300 == pytest.approx(10_000, rel=0.02)
    assert aoi.zones[0].within(aoi.zones[1].buffer(1e-9))


def test_plan_pyramid_properties():
    aoi = compute_aoi([straight_track()], AREA)
    plan = plan_tiles(aoi, AREA)
    assert plan.zooms("imagery")[0] == 5 and plan.zooms("imagery")[-1] == 15
    assert plan.zooms("dem")[-1] == 12
    # every tile has its parent in the plan
    for z in plan.zooms("imagery")[1:]:
        parents = {tuple(t) for t in plan.levels["imagery"][z - 1]}
        assert all((x // 2, y // 2) in parents for x, y in plan.levels["imagery"][z])
    # finest tiles only near the track (within zone 0), coarse ones cover the whole AOI
    for x, y in plan.levels["imagery"][15]:
        assert shapely.box(*tile_bounds_lonlat(15, int(x), int(y))).intersects(aoi.zones[0])
    covered = shapely.union_all(
        [
            shapely.box(*tile_bounds_lonlat(12, int(x), int(y)))
            for x, y in plan.levels["imagery"][12]
        ]
    )
    assert covered.contains(aoi.aoi)
    # corridor: far fewer z15 tiles than the bounding box would need
    assert plan.count("imagery", 15) < 1000
    assert "estimated size" in plan_report(plan)


def test_demo_project_plan():
    tracks, _ = load_gpx_folder(Path(__file__).parents[1] / "examples" / "alps_demo" / "gpx")
    aoi = compute_aoi(tracks, AREA)
    assert aoi is not None and plan_tiles(aoi, AREA).count("dem") > 0


def test_no_tracks():
    assert compute_aoi([], AREA) is None


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(3 * 1024**3) == "3.0 GB"


def test_session_outlines():
    from earthling.core.config import Project
    from earthling.core.session import Session

    session = Session(Project.load(Path(__file__).parents[1] / "examples" / "alps_demo"))
    lines = session.outline_lines()
    assert len(lines) == 3  # two inner zones + AOI
    assert session.plan.count("imagery") > 0
    # outline is centred around the frame origin
    assert abs(lines[-1][0][:, 0].mean()) < 20_000
