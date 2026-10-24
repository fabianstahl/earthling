"""The scene kit's public contract (private scene scripts depend on it): a synthetic two-day
trip on a sloped terrain stub, no GPU and no downloads."""

import datetime as dt
import math

import numpy as np
import pytest

from earthling import scenekit as sk
from earthling.core.animation import Interp
from earthling.session import Session
from helpers import gpx_text, make_project, walk

DAY1 = "2026-07-01T06:00:00"  # 08:00 in Zurich
DAY2 = "2026-07-02T05:30:00"
TRACKS = """
[[tracks]]
name = "hike"
gpx = "gpx/walked"

[[tracks]]
name = "route"
gpx = "gpx/route.gpx"
role = "planned"
"""


def utc(text: str) -> float:
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.UTC).timestamp()


class SlopeTerrain:
    """Rises 20 m per 1/1000 degree to the north (no downloads in tests)."""

    def height_at(self, lon, lat, z=None):
        return 1000.0 + 20_000.0 * (lat - 46.0)

    def heights_at(self, lon, lat, z=None):
        return 1000.0 + 20_000.0 * (np.asarray(lat, dtype=np.float64) - 46.0)


@pytest.fixture(autouse=True)
def _terrain(monkeypatch):
    monkeypatch.setattr(Session, "terrain_data", lambda self, on_demand=True: SlopeTerrain())
    yield
    sk.set_current(None)


@pytest.fixture
def project(tmp_path):
    """Day 1 walks 1.2 km north with a 12 minute break, day 2 1.2 km east the next morning;
    the files are named against the time order; a planned route without timestamps."""
    day1 = walk(46.0, 7.0, DAY1, 60, north_m=25.0, pause=(20, 32))
    day2 = walk(day1[-1][0], day1[-1][1], DAY2, 40, east_m=30.0)
    route = [day1[0][:3] + (None,), day2[-1][:3] + (None,)]
    gpx = {
        "walked/a_second.gpx": gpx_text({"Second day": day2}),
        "walked/b_first.gpx": gpx_text({"First day": day1}),
        "route.gpx": gpx_text({"Route": route}),
    }
    return make_project(tmp_path / "trip", gpx, extra=TRACKS)


@pytest.fixture
def trip(project):
    return sk.load_journey(project)


@pytest.fixture
def untimed(tmp_path):
    plan = walk(46.0, 7.0, None, 40, north_m=30.0)  # no timestamps
    folder = make_project(tmp_path / "untimed", {"u.gpx": gpx_text({"Plan": plan})})
    return sk.load_journey(folder)


# --- the journey -------------------------------------------------------------------------------
def test_days_in_time_order_and_by_stem(trip):
    assert sk.current() is trip
    assert len(trip.path.tracks) == 2 and len(trip.gpu) == 3  # the planned route: no day
    assert trip.day_name(1) == "First day" and trip.day_name(2) == "Second day"
    assert trip.day_index("a_second") == 1 and trip.day("b_first") is trip.day(1)
    with pytest.raises(KeyError):
        trip.day_index("nope")
    assert trip.day_times(1) == (utc(DAY1), utc(DAY1) + 3600.0)
    assert trip.day_times(2) == (utc(DAY2), utc(DAY2) + 2400.0)
    assert trip.path.track_range(1)[0] == pytest.approx(trip.day(1).length_m)
    start = trip.start(1)
    start[0] += 99.0  # a copy: scripts may move it
    assert trip.start(1)[0] != start[0]
    assert np.linalg.norm(trip.end(1)[:2] - trip.start(2)[:2]) < 1.0  # day 2 goes on from there


def test_custom_day_convention(project):
    trip = sk.load_journey(project, select=lambda journey, day: {"mon": 0, "tue": 1}[day])
    assert trip.day_name("tue") == "Second day"


def test_coordinates_and_ground(trip):
    p = trip.enu(46.005, 7.002, 500.0)
    lon, lat = trip.lonlat(p)
    assert (lon, lat) == pytest.approx((7.002, 46.005), abs=1e-9)
    # the line lies 3 m above the terrain (z: the slope, minus a little earth curvature)
    end = trip.end(1)
    assert trip.ground_z(end) == pytest.approx(end[2] - 3.0, abs=0.5)
    assert trip.ground_z(end) > trip.ground_z(trip.start(1)) + 200.0
    a, b = trip.start(1), trip.end(1)
    assert trip.max_ground_z(a, b) == pytest.approx(trip.ground_z(b), abs=1.0)
    assert trip.max_ground_z(a, b, radius=500.0) > trip.ground_z(b) + 5.0  # looks beside it
    trip.data = type("NoData", (), {"height_at": lambda self, lon, lat: None})()
    assert math.isnan(trip.ground_z(a))


def test_origin_and_places(project):
    trip = sk.load_journey(project, (46.1, 7.1), places={"hut": (46.01, 7.0)})
    assert (trip.frame.lat, trip.frame.lon) == (46.1, 7.1)
    assert sk.new_scene(trip, 5.0).origin == (46.1, 7.1)
    hut = trip.place("hut")
    assert trip.lonlat(hut) == pytest.approx((7.0, 46.01))
    assert trip.ground_z(hut) == pytest.approx(hut[2], abs=0.01)  # on the terrain by default
    assert trip.place("hut", h=5000.0)[2] > hut[2] + 3000.0


def test_progress_values(trip):
    e0, e1 = trip.day_times(1)[0], trip.day_times(2)[1]
    assert trip.head_at_time(e0 - 3600.0) == 0.0  # clamped
    assert trip.head_at_time(e1 + 3600.0) == 1.0
    assert trip.time_progress(e0 - (e1 - e0)) == pytest.approx(-1.0)  # not clamped
    assert trip.head_at_time(trip.day_times(2)[0]) == pytest.approx(
        trip.path.progress_for_distance(trip.path.track_range(1)[0], "time"), abs=1e-6
    )
    assert trip.distance_progress(trip.path.total_m / 4) == pytest.approx(0.25)
    centre, direction = trip.smoothed(300.0, 100.0)
    assert direction == pytest.approx([0.0, 1.0, 0.0], abs=1e-3)  # day 1 goes north
    assert centre[:2] == pytest.approx(trip.path.position_at(300.0)[:2], abs=1.0)


def test_local_time_and_clock_epochs(trip):
    start = trip.day_times(1)[0]
    assert sk.local(start) == dt.datetime(2026, 7, 1, 8, 0)  # Zurich summer time, naive
    assert sk.epoch_at(trip, 2, "07:45") == utc("2026-07-02T05:45:00")
    sk.set_current(None)
    assert sk.local(start) == dt.datetime(2026, 7, 1, 6, 0)  # no journey: UTC


def test_stops(trip):
    [(start, end, pos)] = sk.stops(trip, 1)
    assert end - start == pytest.approx(13 * 60.0, abs=61.0)  # the 12 minute break
    assert pos[:2] == pytest.approx(trip.path.position_at(500.0)[:2], abs=30.0)
    assert sk.stops(trip, 1, min_minutes=20.0) == []
    assert sk.stops(trip, 2) == []
    middle = (start + end) / 2.0
    assert sk.stop_near(trip, 1, middle)[:2] == (start, end)
    assert sk.stop_near(trip, 1, start - 5000.0)[:2] == (start, end)  # the closest in time
    assert sk.stop_near(trip, 2, middle) is None


def test_places_on_the_tracks(trip):
    g2 = trip.day(2)
    lon, lat = trip.lonlat(g2.enu[20])
    d = sk.distance_near(trip, lon, lat, [2])
    assert d == pytest.approx(g2.offset_m + g2.dist[20], abs=10.0)
    assert sk.distance_near(trip, lon, lat, ["a_second"]) == d  # a stem works as well
    point, dist = sk.track_point_near(trip, 2, g2.enu[20] + [0.0, 40.0, 0.0])
    assert point == pytest.approx(g2.enu[20]) and dist == pytest.approx(g2.offset_m + g2.dist[20])
    trip.places["corner"] = (lat, lon)
    assert sk.track_point_near(trip, 2, "corner")[1] == pytest.approx(dist)
    top, d_top = sk.track_highest(trip, 1)  # the slope rises to the north
    assert top == pytest.approx(trip.end(1)) and d_top == pytest.approx(trip.day(1).length_m)
    reach = sk.first_reach(trip, 2, trip.end(2), radius_m=100.0)
    assert trip.path.total_m - 100.0 < reach < trip.path.total_m
    assert sk.first_reach(trip, 1, trip.end(2)) == pytest.approx(trip.day(1).length_m)  # never


def test_nearest_on():
    companion = sk.nearest_on(np.array([[0.0, 0.0, 5.0], [1000.0, 0.0, 7.0]]), max_gap=300.0)
    assert companion(np.array([900.0, 50.0, 0.0])) == pytest.approx([1000.0, 0.0, 7.0])
    far = np.array([5000.0, 0.0, 1.0])
    assert companion(far) == pytest.approx(far)  # nothing near: the centre itself


def test_track_time_at_stays_on_its_track(trip):
    path = trip.path
    g2 = trip.day(2)
    d = g2.offset_m + 600.0
    assert sk.track_time_at(path, d) == pytest.approx(np.interp(600.0, g2.dist, g2.time))
    assert sk.track_time_at(path, path.total_m + 1e4) == trip.day_times(2)[1]  # clamped
    # just before the end of day 1: day 1's evening, not a time in the night between the days
    end1 = trip.day(1).length_m
    assert sk.track_time_at(path, end1 - 1.0) < trip.day_times(1)[1] + 1.0


def test_untimed_tracks(untimed):
    assert not untimed.path.has_time
    assert sk.stops(untimed, 1) == []
    scene = sk.new_scene(untimed, 5.0)
    head_of = sk.head_keys_for(scene, untimed, 1, 0.0, 5.0, realtime=False)  # (raised before)
    assert head_of(0.0) == 0.0 and head_of(5.0) == pytest.approx(untimed.path.total_m)
    head = scene.animation.curves["progress.head"]
    assert head.evaluate(0.0) < 0.01 and head.evaluate(5.0) > 0.99
    # recorded time asked for, none there: an even speed
    head_of = sk.draw_keys(scene, untimed, 0.0, 1000.0, 0.0, 4.0, realtime=True, soft=False)
    assert head_of(2.0) == pytest.approx(500.0)


# --- scenes ------------------------------------------------------------------------------------
def test_new_scene_defaults(trip):
    scene = sk.new_scene(trip, 12.0, fps=30.0)
    s = scene.store
    assert (scene.animation.duration, scene.animation.fps) == (12.0, 30.0)
    assert s["export.resolution"] == "1920x1080" and s["camera.mode"] == "keys"
    assert not s["labels.visible"] and not s["stats.visible"]
    assert "trackgroup.route.color" in scene.registry  # the project's groups are keyable
    assert scene.origin == (trip.frame.lat, trip.frame.lon)


def test_key_and_visibility(trip):
    scene = sk.new_scene(trip, 10.0)
    sk.key(scene, "post.exposure", 0.0, 0.0)
    sk.key(scene, "post.exposure", 2.0, 1.0)
    assert scene.animation.curves["post.exposure"].keys[0].interp == Interp.EASE_IN_OUT
    pois = [sk.poi(pid, (46.0, 7.0), "flag") for pid in ("a", "b", "c")]
    scene.set_pois(pois)
    sk.show(scene, "a", None, 6.0)  # there from the start (no pop-in in frame 0)
    sk.show(scene, "b", 3.0)
    sk.show(scene, "c", 0.0)
    vis = {p: scene.animation.curves[f"poi.{p}.visible"] for p in "abc"}
    assert [vis["a"].evaluate(t) for t in (-0.5, 0.0, 5.9, 6.0)] == [True, True, True, False]
    assert [vis["b"].evaluate(t) for t in (0.0, 2.9, 3.0, 9.0)] == [False, False, True, True]
    assert vis["c"].evaluate(0.0) is True
    assert scene.store["poi.a.pin"] and not scene.store["poi.a.caption"]


def test_poi_places(trip):
    trip.places["hut"] = (46.01, 7.001)
    assert (sk.poi("h", "hut", "hut").lat, sk.poi("h", "hut", "hut").lon) == (46.01, 7.001)
    enu = trip.enu(46.002, 7.003, 1200.0)
    pin = sk.poi("p", enu, "peak", size=40.0, caption="Top")
    assert (pin.lon, pin.lat) == pytest.approx((7.003, 46.002))
    assert (pin.icon, pin.size_px, pin.caption) == ("builtin:peak", 40.0, "Top")
    other = sk.load_journey(trip.session.project.folder, places={"x": (1.0, 2.0)})
    sk.set_current(trip)
    assert sk.poi("x", "x", "flag", hike=other).lat == 1.0  # an explicit journey


def test_saved_scene_loads_back(trip):
    scene = sk.new_scene(trip, 4.0)
    head_of = sk.head_keys_for(scene, trip, 1, 0.0, 4.0, realtime=True)
    times = np.arange(0.0, 4.0 + 1e-9, 0.5)
    shots = sk.orbit_shots(trip, times, lambda t: trip.path.position_at(head_of(t)),
                           distance=3000.0, pitch=-30.0, heading=0.0, turn=90.0)  # fmt: skip
    sk.key_shots(scene, shots)
    scene.set_pois([sk.poi("top", trip.end(1), "peak")])
    sk.show(scene, "top", 2.0)
    sk.set_current(None)
    path = sk.save(scene, "s01_up", hike=trip)  # the journey given explicitly
    assert path == trip.session.project.folder / "scenes" / "s01_up.json"
    loaded, problems = trip.session.load_scene("scenes/s01_up.json")
    assert problems == []
    assert loaded.animation.duration == 4.0 and loaded.origin == scene.origin
    assert [p.id for p in loaded.pois] == ["top"]
    for pid in ("camera.pose", "progress.head", "poi.top.visible"):
        a, b = scene.animation.curves[pid], loaded.animation.curves[pid]
        for t in (0.0, 1.3, 2.0, 3.9):
            assert np.asarray(b.evaluate(t), float) == pytest.approx(
                np.asarray(a.evaluate(t), float), abs=1e-6
            )


def test_holds_without_camera_and_with_early_keys(trip):
    scene = sk.new_scene(trip, 5.0)
    sk.key(scene, "post.exposure", -1.0, 0.5, Interp.STEP)  # visible from the start
    sk.key(scene, "post.exposure", 2.0, 1.0)
    sk.add_holds(scene, 0.5)
    assert scene.animation.duration == 6.0
    assert [k.time for k in scene.animation.curves["post.exposure"].keys] == [-1.0, 2.5]
    assert "camera.pose" not in scene.animation.curves


def test_holds_start_and_end_at_rest(trip):
    scene = sk.new_scene(trip, 3.0)
    sk.key_shots(scene, [(0.0, np.array([0.0, -5000.0, 3000.0]), np.zeros(3)),
                         (3.0, np.array([4000.0, -5000.0, 3000.0]), np.zeros(3))])  # fmt: skip
    first = scene.animation.curves["camera.pose"].evaluate(0.0)
    last = scene.animation.curves["camera.pose"].evaluate(3.0)
    sk.add_holds(scene)
    cam = scene.animation.curves["camera.pose"]
    assert cam.evaluate(0.0) == pytest.approx(first) and cam.evaluate(5.0) == pytest.approx(last)
    step = 1.0 / 60.0  # speed zero where the motion starts and ends, full speed in between
    x = [cam.evaluate(t)[0] for t in (1.0, 1.0 + step, 2.5, 2.5 + step, 4.0 - step, 4.0)]
    assert abs(x[1] - x[0]) < 0.1 * abs(x[3] - x[2]) and abs(x[5] - x[4]) < 0.1 * abs(x[3] - x[2])


def test_head_keys_realtime_and_by_distance(trip):
    path = trip.path
    d0, d1 = path.track_range(1)
    scene = sk.new_scene(trip, 10.0)
    head_of = sk.head_keys_for(scene, trip, 2, 1.0, 9.0, realtime=True)
    assert head_of(0.0) == pytest.approx(d0) and head_of(10.0) == pytest.approx(d1)  # clamped
    head = scene.animation.curves["progress.head"]
    assert head.keys[0].time == 1.0 and head.keys[-1].time == pytest.approx(9.0)
    assert head.evaluate(9.0) == pytest.approx(1.0, abs=1e-3)  # the end of the journey
    # by distance: even speed in the middle, half way at half time
    scene = sk.new_scene(trip, 10.0)
    even = sk.head_keys_for(scene, trip, 1, 0.0, 10.0, realtime=False)
    assert even(5.0) == pytest.approx(trip.day(1).length_m / 2.0, rel=1e-3)
    # day 1 in its timestamps: the break pauses the line
    timed = sk.head_keys_for(scene, trip, 1, 0.0, 10.0, realtime=True)
    crawl = [t for t in np.arange(0.0, 10.0, 0.05) if timed(t + 0.05) - timed(t) < 2.0]
    assert len(crawl) * 0.05 > 1.0  # 13 of 60 minutes in 10 s (eased): a second or more
    assert all(2.0 < t < 7.0 for t in crawl)  # in the middle of the day, not at its ends
    stop = sk.head_keys_for(scene, trip, 1, 0.0, 10.0, realtime=False, d_end=600.0)
    assert stop(10.0) == pytest.approx(600.0)


def test_head_keys_slow_stretch(trip):
    scene = sk.new_scene(trip, 10.0)
    slow = sk.head_keys_for(scene, trip, 1, 0.0, 10.0, realtime=False, slow=(400.0, 800.0, 4.0))
    ts = np.arange(0.0, 10.0, 0.01)
    inside = sum(400.0 <= slow(t) <= 800.0 for t in ts) * 0.01
    assert inside > 1.6 * (400.0 / 1200.0 * 10.0)  # much longer than at an even speed
    assert sk.reached_at(slow, 800.0, 0.0, 10.0) == pytest.approx(
        ts[np.argmax([slow(t) >= 800.0 for t in ts])], abs=1.0 / 60.0
    )
    assert sk.reached_at(slow, 1e9, 0.0, 10.0) == 10.0  # never: the end


def test_key_head_follows_epochs(trip):
    scene = sk.new_scene(trip, 4.0)
    e0, e1 = trip.day_times(2)
    sk.key_head(scene, trip, [0.0, 4.0], [e0, e1])
    head = scene.animation.curves["progress.head"]
    assert head.evaluate(0.0) == pytest.approx(trip.head_at_time(e0))
    assert head.evaluate(4.0) == 1.0 and head.keys[0].interp == Interp.LINEAR


def test_travel_scene(trip):
    scene = sk.new_scene(trip, 7.0)
    e0, e1, shots = sk.travel_scene(scene, trip, "a_second", duration=7.0, near=1500.0)
    assert (e0, e1) == trip.day_times(2)
    assert len(shots) == 211 and shots[0][0] == 0.0 and shots[-1][0] == pytest.approx(7.0)
    assert shots[0][2] == pytest.approx(trip.start(2)) and shots[-1][2] == pytest.approx(
        trip.end(2)
    )
    head = scene.animation.curves["progress.head"]
    assert head.evaluate(0.5) == pytest.approx(trip.head_at_time(e0), abs=1e-6)  # still at start
    assert head.evaluate(6.5) == pytest.approx(1.0)
    assert "camera.pose" in scene.animation.curves and "haze.aerial" in scene.animation.curves
    for _, eye, _ in shots:  # above the terrain
        assert eye[2] >= trip.ground_z(eye) + 399.0
    dist = [np.linalg.norm(e - g) for _, e, g in shots]
    assert max(dist) > 1.4 * dist[0]  # zooms out half way
    # only the shots, shifted on the timeline
    other = sk.new_scene(trip, 7.0)
    *_, moved = sk.travel_scene(other, trip, "a_second", offset=2.0, apply=False, keep=300.0)
    assert other.animation.curves.get("progress.head") is None and moved[0][0] == 2.0
    assert sk.travel_time_of(e0, e0, e1, 1.0, 6.0) == 1.0
    assert sk.travel_time_of(e1 + 99.0, e0, e1, 1.0, 6.0) == pytest.approx(6.0, abs=1e-3)
    assert sk.travel_time_of((e0 + e1) / 2.0, e0, e1, 1.0, 6.0) == pytest.approx(3.5, abs=1e-3)


def test_night_light(trip):
    scene = sk.new_scene(trip, 10.0)
    sk.night_light(scene, 2.0, 5.0, ambient=3.0)
    amb = scene.animation.curves["light.night_ambient"]
    assert [amb.evaluate(t) for t in (0.0, 2.0, 5.0, 9.0)] == pytest.approx([1.0, 1.0, 3.0, 3.0])
    other = sk.new_scene(trip, 10.0)
    sk.night_light(other, 0.0, 0.0)
    assert [k.value for k in other.animation.curves["post.exposure"].keys] == [1.4]


def test_sun_keys_clamped_and_stepping_over_the_night(trip):
    path = trip.path
    scene = sk.new_scene(trip, 4.0)
    times = np.arange(0.0, 4.0 + 1e-9, 0.25)
    head_of = sk.draw_keys(scene, trip, 0.0, path.total_m, 0.0, 4.0, realtime=False, soft=False)
    sk.sun_keys(scene, trip, times, head_of, earliest=8.25, latest=8.5)
    assert scene.store["sun.follow_track"] is False
    sun = scene.animation.curves["sun.datetime"]
    clocks = [k.value for k in sun.keys]
    assert len(clocks) == len(times)
    assert all(dt.time(8, 15) <= c.time() <= dt.time(8, 30) for c in clocks)
    assert {c.date() for c in clocks} == {dt.date(2026, 7, 1), dt.date(2026, 7, 2)}
    change = [i for i in range(len(clocks) - 1) if clocks[i + 1].date() != clocks[i].date()]
    assert len(change) == 1 and sun.keys[change[0]].interp == Interp.STEP  # not through the night
    # again over the same times: the old keys are replaced
    sk.sun_keys(scene, trip, times, head_of, smooth_nights=True)
    assert len(scene.animation.curves["sun.datetime"].keys) == len(times)
    sk.daylight_keys(scene, trip, times, head_of)
    assert sk.local(trip.day_times(1)[0]) <= scene.animation.curves["sun.datetime"].keys[0].value


def test_clamp_sun(trip):
    scene = sk.new_scene(trip, 4.0)
    head_of = sk.head_keys_for(scene, trip, 1, 0.0, 4.0, realtime=True)
    sk.clamp_sun(scene, trip, head_of, [0.0, 2.0, 4.0], latest="08:30", earliest="08:10", day=1)
    clocks = [k.value for k in scene.animation.curves["sun.datetime"].keys]
    assert clocks[0] == dt.datetime(2026, 7, 1, 8, 10) and clocks[-1].time() == dt.time(8, 30)


def test_head_distance_of_reads_the_keys(trip):
    scene = sk.new_scene(trip, 6.0)
    assert sk.head_distance_of(scene, trip.path)(3.0) == pytest.approx(
        trip.path.distance_for(scene.store["progress.head"], scene.store["progress.mode"])
    )  # no keys: the stored value
    head_of = sk.draw_keys(scene, trip, 100.0, 2000.0, 1.0, 5.0, realtime=True)
    read = sk.head_distance_of(scene, trip.path)
    for t in (1.0, 2.2, 3.5, 5.0):
        assert read(t) == pytest.approx(head_of(t), abs=2.0)


def test_draw_and_clear_keys(trip):
    scene = sk.new_scene(trip, 10.0)
    sk.key(scene, "progress.head", 0.0, 0.0)
    sk.key(scene, "progress.head", 9.0, 1.0)
    sk.draw_keys(scene, trip, 0.0, 500.0, 2.0, 4.0, realtime=False)
    times = [k.time for k in scene.animation.curves["progress.head"].keys]
    assert times[0] == 0.0 and times[-1] == 9.0 and len(times) == 33  # 2..4 every 1/15 s
    sk.clear_keys(scene, "progress.head", 2.0, 4.0)
    assert [k.time for k in scene.animation.curves["progress.head"].keys] == [0.0, 9.0]
    sk.clear_keys(scene, "no.such.curve", 0.0, 1.0)  # nothing to clear


# --- cameras -----------------------------------------------------------------------------------
def test_directions():
    assert sk.bearing([0, 0, 0], [10, 0, 0]) == pytest.approx(90.0)
    assert sk.heading_of([0, 0, 0], [0, -10, 5]) == pytest.approx(180.0)
    eye = sk.orbit_eye([0.0, 0.0, 0.0], 90.0, -45.0, 1000.0)
    assert eye == pytest.approx([-707.107, 0.0, 707.107], abs=1e-3)  # looking east and down


def test_geo_orbit_eye(trip):
    lat, lon = trip.frame.lat, trip.frame.lon
    eye, target = sk.geo_orbit_eye(trip, lat, lon, 0.0, 30.0, -40.0, 2000.0)
    assert eye == pytest.approx(sk.orbit_eye(target, 30.0, -40.0, 2000.0), abs=1e-6)
    eye, target = sk.geo_orbit_eye(trip, lat + 20.0, lon, 0.0, 0.0, -90.0, 5000.0)
    assert np.linalg.norm(eye - target) == pytest.approx(5000.0)
    up = trip.enu(lat + 20.0, lon, 1.0) - trip.enu(lat + 20.0, lon, 0.0)
    assert np.dot(eye - target, up) == pytest.approx(5000.0, rel=1e-6)  # straight above there


def test_level_roll_far_from_the_origin(trip):
    target = trip.enu(trip.frame.lat, trip.frame.lon + 15.0, 0.0)  # ~1000 km east
    eye = sk.orbit_eye(target, 0.0, -30.0, 50_000.0)
    assert abs(sk.pose(eye, target)[5]) > 5.0  # the local vertical there is tilted
    trip.level_horizon = False
    assert sk.pose(eye, target)[5] == pytest.approx(0.0, abs=1e-6)
    sk.set_current(None)
    assert sk.level_roll(eye, target, 0.0, -30.0) == 0.0


def test_plan_camera_through_views(trip):
    scene = sk.new_scene(trip, 4.0)
    views = [(0.0, sk.View((0.0, 0.0, 1000.0), 350.0, -30.0, 2000.0)),
             (4.0, sk.View((500.0, 0.0, 1000.0), 10.0, -30.0, 8000.0))]  # fmt: skip
    sk.plan_camera(scene, None, views)
    cam = scene.animation.curves["camera.pose"]
    assert len(cam.keys) == 121
    assert cam.evaluate(0.0)[:3] == pytest.approx(sk.orbit_eye((0, 0, 1000), 350.0, -30.0, 2000))
    heading = cam.evaluate(2.0)[3] % 360.0
    assert heading < 5.0 or heading > 355.0  # the short way round, over north
    eye = np.array(cam.evaluate(2.0)[:3])
    assert np.linalg.norm(eye - [250.0, 0.0, 1000.0]) == pytest.approx(4000.0, rel=0.05)  # log
    low = sk.new_scene(trip, 4.0)  # a view below the terrain: lifted above it
    under = sk.View((0.0, 0.0, 0.0), 0.0, -5.0, 1000.0)
    sk.plan_camera(low, trip, [(0.0, under), (4.0, under)], clearance=120.0)
    for k in low.animation.curves["camera.pose"].keys:
        assert k.value[2] >= trip.ground_z(k.value[:3]) + 119.0


def test_line_of_sight(trip):
    ground = trip.ground_z([0.0, 0.0, 0.0])
    assert sk.line_of_sight(trip, [0.0, -5000.0, ground + 2000.0], [0.0, 0.0, ground])
    assert not sk.line_of_sight(trip, [0.0, 5000.0, ground + 50.0], [0.0, 0.0, ground])  # uphill


def test_follow_shots(trip):
    scene = sk.new_scene(trip, 8.0)
    head_of = sk.head_keys_for(scene, trip, 1, 0.0, 8.0, realtime=False)
    times = np.arange(0.0, 8.0 + 1e-9, 0.25)
    shots = sk.follow_shots(trip, times, head_of, distance=1500.0, pitch=-30.0, side=0.0,
                            clearance=200.0)  # fmt: skip
    assert [s[0] for s in shots] == pytest.approx(list(times))
    for t, eye, target in shots:
        assert eye[2] >= trip.ground_z(eye) + 0.6 * 200.0 - 1e-6
        assert np.linalg.norm(target[:2] - trip.path.position_at(head_of(t))[:2]) < 600.0
        assert eye[1] < target[1]  # behind the walker (day 1 goes north)
    route = np.array([[2000.0, y, 0.0] for y in np.arange(-500.0, 2000.0, 50.0)])
    pulled = sk.follow_shots(trip, times, head_of, distance=1500.0, pitch=-30.0, side=0.0,
                             companion=sk.nearest_on(route))  # fmt: skip
    assert pulled[16][2][0] > shots[16][2][0] + 500.0  # aims between the walker and the route


def test_map_shots_fit_the_points(trip):
    times = np.arange(0.0, 6.0 + 1e-9, 0.25)
    head_of = lambda t: trip.path.total_m * t / 6.0  # noqa: E731
    shots = sk.map_shots(trip, times, head_of, distance_of=lambda t: 1000.0)
    _, eye, target = shots[12]
    h, p = sk.pose(eye, target)[3:5]
    assert abs((h + 180.0) % 360.0 - 180.0) < 1.0 and p == pytest.approx(-60.0, abs=1.0)
    far = [10_000.0, 10_000.0, 0.0]
    wide = sk.map_shots(trip, times, head_of, distance_of=lambda t: 1000.0,
                        points_of=lambda t: [far])  # fmt: skip
    for (_, e1, g1), (_, e2, g2) in zip(shots, wide, strict=True):
        assert np.linalg.norm(e2 - g2) > 2.0 * np.linalg.norm(e1 - g1)


def test_orbit_line_blend_and_still_shots(trip):
    times = np.linspace(0.0, 4.0, 41)
    centre = np.array([0.0, 0.0, 1000.0])
    orbit = sk.orbit_shots(None, times, lambda t: centre, distance=3000.0, pitch=-20.0,
                           heading=0.0, turn=90.0)  # fmt: skip
    assert sk.heading_of(orbit[0][1], centre) == pytest.approx(0.0, abs=1e-6)
    assert sk.heading_of(orbit[-1][1], centre) == pytest.approx(90.0, abs=1e-6)
    assert {round(float(np.linalg.norm(e - centre)), 6) for _, e, _ in orbit} == {3000.0}
    head_of = lambda t: trip.path.total_m * t / 4.0  # noqa: E731
    line = sk.line_shots(trip, times, head_of, near=2000.0, far=8000.0, pitch=-40.0, side=90.0,
                         clearance=0.0)  # fmt: skip
    dist = [np.linalg.norm(e - g) for _, e, g in line]
    assert dist[0] == pytest.approx(2000.0) and dist[20] == pytest.approx(8000.0)
    still = sk.still_shots(times, [1.0, 2.0, 3.0], centre)
    blend = sk.blend_shots(orbit, still, 1.0, 3.0)
    assert blend[5][1] == pytest.approx(orbit[5][1]) and blend[35][1] == pytest.approx([1, 2, 3])
    assert np.linalg.norm(blend[20][1] - orbit[20][1]) > 100.0  # mixed in between


def test_key_shots_resample(trip):
    scene = sk.new_scene(trip, 1.0)
    shots = [(t, np.array([t * 100.0, -2000.0, 1500.0]), np.zeros(3)) for t in (0.0, 0.5, 1.01)]
    sk.key_shots(scene, shots)
    times = [k.time for k in scene.animation.curves["camera.pose"].keys]
    assert times[1] == pytest.approx(sk.KEY_STEP, abs=1e-4)
    assert times[-1] == pytest.approx(1.01, abs=1.0 / 60.0)  # the last one too (on a frame)
    assert scene.animation.curves["camera.pose"].evaluate(0.5)[0] == pytest.approx(50.0, abs=0.5)


def test_plan_look(trip):
    scene = sk.new_scene(trip, 2.0)
    points = [(0.0, [0.0, -3000.0, 0.0], [0.0, 0.0, 0.0]), (2.0, [0.0, -1000.0, 0.0], [0, 0, 0])]
    shots = sk.plan_look(scene, trip, points, step=0.5, clearance=150.0)
    assert [s[0] for s in shots] == [0.0, 0.5, 1.0, 1.5, 2.0]
    for _, eye, _ in shots:  # lifted above the slope
        assert eye[2] == pytest.approx(trip.ground_z(eye) + 150.0, abs=0.5)
    assert scene.animation.curves["camera.pose"].keys[-1].time == 2.0
