"""Whole workflows and scaling (slow suite, offline): GPX to a checked video on the command
line, a session in the application from opening to exporting, a month-long journey, and the
most expensive settings - each within a time budget, so that no combination makes a render
take days. Budgets are generous (machines differ); the measured values are printed."""

import datetime as dt
import time

import numpy as np
import pytest

from earthling import scenekit as sk
from earthling.cli import main
from earthling.core.geo import lonlat_to_tile
from earthling.render import lod
from earthling.session import Session
from helpers import FakeTerrainData, gpx_text, make_project, walk
from test_ui_actions import window  # noqa: F401 (the fixture)

pytestmark = pytest.mark.slow


class Terrain(FakeTerrainData):
    def height_at(self, lon, lat, z=None):
        return 1500.0

    def heights_at(self, lon, lat, z=None):
        return np.full(np.shape(lon), 1500.0)


@pytest.fixture
def offline(monkeypatch):
    """A terrain stub instead of tiles, no labels (the network is refused anyway)."""
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    monkeypatch.setattr(Session, "terrain_data", lambda self, on_demand=True: Terrain())
    monkeypatch.setattr(Session, "terrain_nodes", lambda self: lod.NodeSet({10: [(tx, ty)]}))
    monkeypatch.setattr(Session, "labels", None)
    yield
    sk.set_current(None)


def two_days(folder):
    days = {"day1.gpx": gpx_text({"Day 1": walk(46.0, 7.0, "2026-07-01T06:00:00", 240,
                                                north_m=60.0, pause=(100, 130))}),
            "day2.gpx": gpx_text({"Day 2": walk(46.13, 7.0, "2026-07-02T06:30:00", 200,
                                                east_m=70.0)})}  # fmt: skip
    return make_project(folder, days)


def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


# --- 1: GPX to a checked video ---------------------------------------------------------------
def test_gpx_to_a_checked_video(offline, gl_ctx, tmp_path, capsys):
    project = two_days(tmp_path / "trip")
    journey = sk.load_journey(project)
    scene = sk.new_scene(journey, 4.0, fps=30.0)
    head_of = sk.head_keys_for(scene, journey, 1, 0.0, 4.0, realtime=True)
    times = np.arange(0.0, 4.0 + 1e-9, 1.0 / 15.0)
    sk.key_shots(scene, sk.follow_shots(journey, times, head_of, distance=3000.0, pitch=-30.0,
                                        side=30.0, sigma_s=0.5, heading_sigma_s=1.0))  # fmt: skip
    sk.daylight_keys(scene, journey, times, head_of)
    sk.save(scene, "trip")
    video = project / "renders" / "trip.mp4"
    video.parent.mkdir()
    argv = ["render", str(project), "--scene", "scenes/trip.json", "--preset", "h264",
            "--resolution", "640x360", "--out", str(video)]  # fmt: skip
    code, seconds = timed(lambda: main(argv))
    assert code == 0 and video.stat().st_size > 10_000
    assert main(["qa", str(video), "--out", str(project / "qa")]) == 0
    report = capsys.readouterr().out
    assert "trip: 120 frames at 30 fps" in report
    print(f"\n4 s at 640x360 rendered in {seconds:.1f} s")
    assert seconds < 120.0


# --- 2: a session in the application ---------------------------------------------------------
def test_a_session_in_the_application(window, qtbot):  # noqa: F811
    """Open, add a POI and camera keys, apply a look, animate a value, save, reload: all of it
    is still there; then export a short video through the export dialog's own function."""
    from test_ui_actions import menu_actions

    actions = dict(menu_actions(window.menuBar()))
    pois = len(window.scene.pois)
    actions["Add › POI at the Marker"].trigger()
    assert len(window.scene.pois) == pois + 1
    window.timeline.set_time(0.0)
    window.add_camera_key()
    window.timeline.set_time(3.0)
    window.viewport.orbit.rotate(40.0, -5.0)
    window.add_camera_key()
    assert len(window.scene.animation.curves["camera.pose"].keys) >= 2
    next(e for e in window._look_entries() if e.title == "Golden hour").run()
    window.keys.set_key("fog.density", 0.2)
    window.timeline.set_time(4.0)
    window.keys.set_key("fog.density", 2.0)
    assert window.scene.animation.is_animated("fog.density")
    assert window.save_scene()
    path = window.scene.path
    window.open_project(window.project.folder)
    window.load_scene(path)
    assert len(window.scene.pois) == pois + 1
    assert window.scene.animation.is_animated("fog.density")
    assert len(window.scene.animation.curves["camera.pose"].keys) >= 2
    import moderngl

    from earthling.export.frames import FrameRenderer, scene_renderer
    from earthling.export.video import PRESETS, export_video

    out = window.project.folder / "renders" / "session.mp4"
    out.parent.mkdir(exist_ok=True)
    renderer = scene_renderer(moderngl.create_standalone_context(require=430),
                              window.session, window.scene)  # fmt: skip
    frames = FrameRenderer(renderer, window.scene.animation)
    try:
        count, seconds = timed(lambda: export_video(frames, out, 320, 180, 30.0,
                                                    PRESETS["h264"], 0.0, 1.0))  # fmt: skip
    finally:
        renderer.terrain.shutdown()
    assert count == 30 and out.stat().st_size > 1000
    print(f"\n1 s of the demo exported at 320x180 in {seconds:.1f} s")


# --- 3: a month-long journey -----------------------------------------------------------------
def month(folder, days: int = 30, every_s: int = 5):
    """``days`` days of eight hours, a point every ``every_s`` seconds (~170 000 points)."""
    rng = np.random.default_rng(1)
    gpx = {}
    lat, lon = 46.0, 7.0
    for day in range(days):
        n = 8 * 3600 // every_s
        heading = np.cumsum(rng.normal(0.0, 0.05, n)) + day * 0.3
        step = 1.2 * every_s  # m
        lats = lat + np.cumsum(np.cos(heading) * step) / 111_320.0
        lons = lon + np.cumsum(np.sin(heading) * step) / (111_320.0 * np.cos(np.radians(lat)))
        start = dt.datetime(2026, 6, 1 + day, 6, 0)
        points = [(la, lo, 1000.0 + 300.0 * np.sin(i / 900.0),
                   (start + dt.timedelta(seconds=i * every_s)).strftime("%Y-%m-%dT%H:%M:%SZ"))
                  for i, (la, lo) in enumerate(zip(lats, lons, strict=True))]  # fmt: skip
        gpx[f"day{day + 1:02d}.gpx"] = gpx_text({f"Day {day + 1}": points})
        lat, lon = float(lats[-1]), float(lons[-1])
    return make_project(folder, gpx)


def test_a_month_long_journey_scales(offline, gl_ctx, tmp_path, qtbot):
    project, made = timed(lambda: month(tmp_path / "month"))
    journey, loaded = timed(lambda: sk.load_journey(project))
    assert len(journey.path.tracks) == 30
    # loading grows with the journey's length, not faster (a third of it: 10 days)
    _, loaded_10 = timed(lambda: sk.load_journey(month(tmp_path / "ten", days=10)))
    sk.set_current(journey)
    scene = sk.new_scene(journey, 20.0, fps=30.0)

    def build():
        times = np.arange(0.0, 20.0 + 1e-9, 1.0 / 15.0)
        head_of = lambda t: journey.path.total_m * t / 20.0  # noqa: E731
        sk.draw_keys(scene, journey, 0.0, journey.path.total_m - 1.0, 0.0, 20.0, realtime=False)
        sk.key_shots(
            scene,
            sk.map_shots(
                journey,
                times,
                head_of,
                distance_of=lambda t: 40_000.0,
                heading_of=lambda t: 0.0,
                pitch_of=lambda t: -50.0,
            ),
        )
        sk.save(scene, "month")  # fmt: skip

    _, built = timed(build)
    out = tmp_path / "stills"
    argv = ["stills", str(project), "2", "10", "19", "--scene", "scenes/month.json",
            "--size", "640x360", "--out", str(out)]  # fmt: skip
    code, rendered = timed(lambda: main(argv))
    assert code == 0
    # the application opens it without freezing for long
    from earthling.app.main_window import MainWindow
    from helpers import longest_freeze

    app = MainWindow()
    qtbot.addWidget(app)
    app.show()
    frozen = longest_freeze(qtbot, lambda: app.open_project(project),
                            until=lambda: app.session is not None, timeout=60.0)  # fmt: skip
    assert app.scene_list.tree.topLevelItemCount() > 0
    print(
        f"\nmonth: GPX written {made:.1f} s, loaded {loaded:.1f} s (10 days: {loaded_10:.1f} s),"
        f" scene built {built:.1f} s, 3 stills {rendered:.1f} s, the app froze {frozen:.1f} s"
        " while opening it"
    )
    assert loaded < 30.0 and built < 30.0 and rendered < 60.0 and frozen < 15.0
    assert loaded < 5.0 * loaded_10  # (linear: 3x)


# --- 4: the most expensive settings ----------------------------------------------------------
WORST = {
    "anti-aliasing 16 with motion blur": ["--samples", "16", "--set", "export.motion_blur=1.0"],
    "finest terrain detail": ["--set", "export.detail=0.25", "--set", "terrain.detail=0.25"],
    "largest shadows": ["--set", "export.shadow_resolution=8192",
                        "--set", "shadows.distance=200"],
    "every weather layer, rain, lightning": [
        "--set", "weather.enabled=true", "--set", "weather.layer1.enabled=true",
        "--set", "weather.layer2.enabled=true", "--set", "weather.layer3.enabled=true",
        "--set", "rain.enabled=true", "--set", "rain.intensity=1", "--set", "lightning.rate=120"],
    "the largest glow": ["--set", "glow.levels=7", "--set", "glow.radius=3"],
}  # fmt: skip


def test_the_most_expensive_settings_stay_affordable(offline, gl_ctx, tmp_path):
    project = two_days(tmp_path / "trip")
    journey = sk.load_journey(project)
    scene = sk.new_scene(journey, 2.0, fps=30.0)
    sk.key_shots(scene, sk.still_shots([0.0, 2.0], journey.enu(46.0, 6.97, 4000.0),
                                       journey.enu(46.05, 7.0, 1500.0)))  # fmt: skip
    sk.save(scene, "worst")

    def frame(*extra) -> float:
        argv = ["stills", str(project), "1.0", "--scene", "scenes/worst.json",
                "--size", "640x360", "--out", str(tmp_path / "w"), *extra]  # fmt: skip
        code, seconds = timed(lambda: main(argv))
        assert code == 0
        return seconds

    frame()  # (warm up: shaders, caches)
    base = frame()
    times = {name: frame(*args) for name, args in WORST.items()}
    times["all of them"] = frame(*[a for args in WORST.values() for a in args])
    print(f"\none still at 640x360: {base:.2f} s with the defaults")
    for name, seconds in times.items():
        print(f"  {seconds:6.2f} s  {name}")
    # a whole 4K frame of the worst case (24x the pixels) stays well under a minute: a minute of
    # 4K video at 30 fps renders within a day
    assert times["all of them"] * 24 < 60.0, times
