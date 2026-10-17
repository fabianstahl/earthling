from datetime import UTC, datetime

import numpy as np
import pytest

from earthling.core.config import Config, ConfigError, DistanceMark
from earthling.core.geo import LocalFrame
from earthling.render.tracks import calibrate_ground


def _loop(frame):
    """A 4 km square loop (1 m steps) starting and ending at the origin, 1 m/s from t=1000."""
    side = np.arange(0.0, 1000.0, 1.0)
    xs = np.concatenate([side, np.full(1000, 1000.0), 1000.0 - side, np.zeros(1000), [0.0]])
    ys = np.concatenate([np.zeros(1000), side, np.full(1000, 1000.0), 1000.0 - side, [0.0]])
    enu = np.column_stack([xs, ys, np.zeros(len(xs))])
    ground = np.arange(len(xs), dtype=np.float64)
    return enu, ground, 1000.0 + ground


def test_marks_by_position_time_and_end():
    frame = LocalFrame(50.0, 8.0, 0.0)
    enu, ground, time = _loop(frame)

    def mark(x, y, km):
        lat, lon, _ = frame.enu_to_geodetic(np.array([x, y, 0.0]))
        return DistanceMark(km=km, lat=float(lat), lon=float(lon))

    t_2500 = datetime.fromtimestamp(1000.0 + 2500.0, tz=UTC)
    marks = [mark(1000.0, 0.0, 0.9), DistanceMark(km=2.4, time=t_2500), mark(0.0, 5.0, 3.6)]
    out = calibrate_ground(ground, enu, time, marks, 3.7, frame)
    assert out[0] == 0.0
    assert out[1000] == pytest.approx(900.0)
    assert out[500] == pytest.approx(450.0)
    assert out[2500] == pytest.approx(2400.0)
    # the last mark lies near the start too: it is found after the previous mark (at 3995 m)
    assert out[3995] == pytest.approx(3600.0)
    assert out[-1] == pytest.approx(3700.0)
    assert np.all(np.diff(out) >= 0.0)
    # without an end distance the last factor continues
    out = calibrate_ground(ground, enu, time, marks[:2], None, frame)
    assert out[-1] == pytest.approx(2400.0 + 1500.0 * 1.5 / 1.5)
    assert calibrate_ground(ground, enu, time, [], None, frame) is not None


def test_config_validation():
    text = """
[[tracks]]
name = "walk"
gpx = "gpx"
distance_end_km = 100.0
distance_marks = [
  { km = 10, lat = 50.1, lon = 8.5 },
  { km = 30, time = 2026-10-10T17:57:06+02:00 },
]
"""
    config = Config.from_toml(text)
    marks = config.tracks[0].distance_marks
    assert marks[1].time.utcoffset().total_seconds() == 7200 and marks[0].lat == 50.1
    for bad in ("{ km = 10 }", "{ km = 10, lat = 50.0 }", "{ km = 1, time = 2026-10-10T17:57:06 }"):
        with pytest.raises(ConfigError):
            Config.from_toml(f'[[tracks]]\nname = "w"\ngpx = "g"\ndistance_marks = [{bad}]\n')
