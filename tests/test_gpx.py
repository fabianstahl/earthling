import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from earthling.core.gpx import ascent_descent, haversine_m, load_gpx_file, load_gpx_folder

FIXTURES = Path(__file__).parent / "fixtures"


def test_haversine_one_degree_latitude():
    assert haversine_m(45.0, 6.0, 46.0, 6.0) == pytest.approx(111_195, rel=1e-3)


def test_ascent_descent_hysteresis():
    ele = np.array([100, 102, 99, 101, 150, 149, 151, 120, np.nan])
    ascent, descent = ascent_descent(ele)
    assert ascent == pytest.approx(50)
    assert descent == pytest.approx(30)


def test_load_file_tracks_segments_and_times():
    tracks = load_gpx_file(FIXTURES / "two_tracks.gpx")
    assert [t.name for t in tracks] == ["Second", "First"]
    first = tracks[1]
    assert len(first.segments) == 2
    assert np.isnan(first.segments[1].ele).all()
    stats = first.stats
    # times converted from +02:00 to naive UTC
    assert stats.start_time == datetime(2026, 7, 1, 6, 0)
    assert stats.duration_s == 90 * 60
    assert stats.ascent_m == pytest.approx(200)
    assert stats.descent_m == pytest.approx(100)
    # 3 x 0.01 deg + 1 x 0.01 deg in second segment; the gap between segments is not counted
    assert stats.distance_m == pytest.approx(4 * 1112, rel=1e-2)
    assert first.bbox() == pytest.approx((6.0, 45.0, 6.0, 45.11))


def test_load_folder_sorts_by_time_and_reports_errors(tmp_path):
    shutil.copy(FIXTURES / "two_tracks.gpx", tmp_path / "a.gpx")
    shutil.copy(FIXTURES / "broken.gpx.txt", tmp_path / "b.gpx")
    tracks, errors = load_gpx_folder(tmp_path)
    assert [t.name for t in tracks] == ["First", "Second"]
    assert len(errors) == 1 and errors[0].startswith("b.gpx")


def test_missing_folder():
    tracks, errors = load_gpx_folder(Path("does/not/exist"))
    assert tracks == [] and errors


def test_demo_project_loads():
    folder = Path(__file__).parents[1] / "examples" / "alps_demo" / "gpx"
    tracks, errors = load_gpx_folder(folder)
    assert not errors and len(tracks) == 2
    assert 10_000 < tracks[0].stats.distance_m < 40_000
