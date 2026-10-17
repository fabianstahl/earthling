import datetime as dt
import json
import shutil

import numpy as np
import pytest

from earthling.core.config import Config
from earthling.core.geo import great_circle
from earthling.core.gpx import haversine_m, load_gpx_file
from earthling.data import lines
from helpers import DEMO

UTC2 = dt.timezone(dt.timedelta(hours=2))
START, END = dt.datetime(2026, 9, 1, 8, 0, tzinfo=UTC2), dt.datetime(2026, 9, 1, 10, 0, tzinfo=UTC2)


def test_great_circle_and_arc():
    pts = great_circle((50.0, 8.6), (40.1, 116.6), step_km=50.0)
    assert pts[0] == pytest.approx((50.0, 8.6)) and pts[-1] == pytest.approx((40.1, 116.6))
    assert max(p[0] for p in pts) > 55.0  # the great circle bulges north
    heights = lines.arc_heights(pts)
    assert heights[0] == pytest.approx(0.0, abs=1e-6) and heights.max() > 100_000.0


def test_make_line(tmp_path):
    path = lines.make_line(tmp_path / "f.gpx", "Flight", "flight", (46.0, 7.0), (47.0, 8.0),
                           START, END, tmp_path)  # fmt: skip
    (track,) = load_gpx_file(path)
    seg = track.segments[0]
    assert seg.ele.max() >= 10_000.0
    assert seg.time[0] == np.datetime64(START.astimezone(dt.UTC).replace(tzinfo=None))
    assert seg.time[-1] == np.datetime64(END.astimezone(dt.UTC).replace(tzinfo=None))
    straight = load_gpx_file(lines.make_line(tmp_path / "s.gpx", "S", "straight", (46.0, 7.0),
                                             (46.0, 7.1), START, END, tmp_path))[0]  # fmt: skip
    assert len(straight.segments[0].lat) > 3


def test_road_route_is_cached(tmp_path, monkeypatch):
    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            calls.append(url)
            body = {"routes": [{"geometry": {"coordinates": [[7.001, 46.0], [7.05, 46.02]]}}]}
            return type("R", (), {"text": json.dumps(body), "raise_for_status": lambda s: None})()

    monkeypatch.setattr(lines, "http_client", Client)
    pts = lines.road_route((46.0, 7.0), (46.03, 7.06), tmp_path)
    assert pts[0] == (46.0, 7.0) and pts[-1] == (46.03, 7.06) and len(pts) == 4
    lines.road_route((46.0, 7.0), (46.03, 7.06), tmp_path)
    assert len(calls) == 1


def test_new_line_adds_a_group(qtbot, tmp_path, monkeypatch):
    from earthling.app import line_dialog
    from earthling.app.main_window import MainWindow

    project = tmp_path / "demo"
    shutil.copytree(DEMO, project)
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(project)
    text = line_dialog.add_group_text("", window.session, "flights", True)
    assert Config.from_toml((project / "earthling.toml").read_text(encoding="utf-8") + text)
    places = iter([(7.0, 46.0), (7.2, 46.1)])
    dialog_class = line_dialog.LineDialog

    class Dialog(dialog_class):
        def exec(self):
            self._choose("from", "From")
            self._choose("to", "To")
            self.group.setCurrentIndex(self.group.findData(line_dialog.NEW_GROUP))
            self.new_group.setText("Bus rides")
            self.name.setText("to the pass")
            return True

    monkeypatch.setattr(line_dialog, "LineDialog", Dialog)
    window.choose_place = lambda title, start: next(places)
    window.new_line()
    group = window.session.group("bus_rides")
    assert group is not None and [t.name for t in group.tracks] == ["to the pass"]
    a = group.tracks[0].segments[0]
    assert haversine_m(a.lat[0], a.lon[0], 46.0, 7.0) < 1.0
    assert window.session.group("tracks") is not None  # (the demo's own group stays)
