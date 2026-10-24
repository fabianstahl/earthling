"""The smaller dialogs as the user fills them in: choosing a place, a new line, downloads (in a
background thread, cancellable), the video export (with a fake writer) and the Animation menu."""

import datetime as dt
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox

from earthling.data.downloader import DownloadProgress
from earthling.data.jobs import DownloadJob
from helpers import FRAME, gpx_text, make_project, walk
from test_scenekit import journey  # noqa: F401 (fixture: the demo journey on flat terrain)


def built_track(name="Day 1"):
    """A track as drawn: 1 km north, rising 100 m to the middle, then down again."""
    north = np.linspace(0.0, 1000.0, 11)
    ele = np.concatenate([np.linspace(1000, 1100, 6), np.linspace(1090, 1050, 5)])
    enu = np.column_stack([np.zeros(11), north, ele])
    return SimpleNamespace(track=SimpleNamespace(name=name), enu=enu, ele=ele, ground=None,
                           dist=north)  # fmt: skip


# --- place dialog ----------------------------------------------------------------------------
def test_place_dialog_coordinates_or_a_track_point(qtbot):
    from earthling.app.place_dialog import PlaceDialog

    alone = PlaceDialog("Where?", [], FRAME, (7.1, 46.2))
    qtbot.addWidget(alone)
    assert alone.by_coords.isChecked() and not alone.on_track.isEnabled()  # (no tracks)
    assert alone.lonlat() == (pytest.approx(7.1), pytest.approx(46.2))
    dialog = PlaceDialog("Where?", [built_track(), built_track("Day 2")], FRAME, None)
    qtbot.addWidget(dialog)
    assert dialog.on_track.isChecked() and not dialog.lat.isEnabled()
    assert dialog.track.count() == 2 and not dialog.km.isEnabled()

    def place(point, km=0.0):
        dialog.point.setCurrentIndex(dialog.point.findData(point))
        dialog.km.setValue(km)
        lon, lat = dialog.lonlat()
        return FRAME.geodetic_to_enu(lat, lon, 0.0)[:2]  # (not at its height: ~0.2 m off)

    assert place("start") == pytest.approx([0.0, 0.0], abs=0.5)
    assert place("end") == pytest.approx([0.0, 1000.0], abs=0.5)
    assert place("highest") == pytest.approx([0.0, 500.0], abs=0.5)
    assert place("lowest") == pytest.approx([0.0, 0.0], abs=0.5)
    assert place("distance", 0.25) == pytest.approx([0.0, 250.0], abs=0.5)
    assert dialog.km.isEnabled()  # ("at distance" needs the kilometres)
    dialog.by_coords.setChecked(True)  # back to typed coordinates
    assert dialog.lat.isEnabled() and not dialog.track.isEnabled() and not dialog.km.isEnabled()
    assert dialog.lonlat() == (0.0, 0.0)


# --- new line --------------------------------------------------------------------------------
def test_line_dialog_needs_both_places_and_names_things_safely(qtbot):
    from earthling.app.line_dialog import NEW_GROUP, LineDialog

    places = iter([None, (7.0, 46.0), (7.5, 46.5), (7.2, 46.2)])
    asked = []
    start = dt.datetime(2026, 7, 1, 9, 30)

    def choose(title, at):
        asked.append((title, at))
        return next(places)

    dialog = LineDialog(["tracks"], "Europe/Zurich", choose, start)
    qtbot.addWidget(dialog)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled() and not dialog.new_group.isEnabled()
    dialog._choose("from", "From")  # (cancelled)
    assert not ok.isEnabled() and dialog.labels["from"].text() == "–"
    dialog._choose("from", "From")
    assert dialog.labels["from"].text() == "46.00000, 7.00000" and not ok.isEnabled()
    dialog._choose("to", "To")
    assert ok.isEnabled()
    dialog._choose("from", "From")  # choosing again starts at the place chosen before
    assert asked[-1] == ("From…", (7.0, 46.0)) and dialog.places["from"] == (46.2, 7.2)
    assert dialog.group_name() == "tracks"
    dialog.group.setCurrentIndex(dialog.group.findData(NEW_GROUP))
    assert dialog.new_group.isEnabled()
    dialog.new_group.setText("Bus Rides!")
    assert dialog.group_name() == "bus_rides"
    dialog.new_group.setText("  ")
    assert dialog.group_name() == "travel"
    dialog.name.setText("to the pass / 2")
    assert dialog.file_name() == "to_the_pass_2"
    dialog.name.setText("")
    assert dialog.file_name() == "line"
    depart, arrive = dialog.times()
    zone = ZoneInfo("Europe/Zurich")
    assert depart == dt.datetime(2026, 7, 1, 9, 30, tzinfo=zone)
    assert arrive - depart == dt.timedelta(hours=1)


def test_line_groups_and_where_their_files_go(tmp_path):
    from earthling.app.line_dialog import add_group_text, group_folder

    (tmp_path / "gpx" / "walked").mkdir(parents=True)
    groups = {"walked": "gpx/walked", "single": "gpx/one.gpx", "glob": "gpx/day*.gpx",
              "abs": str(tmp_path / "gpx" / "walked")}  # fmt: skip
    session = SimpleNamespace(
        project=SimpleNamespace(folder=tmp_path),
        group=lambda name: SimpleNamespace(definition=SimpleNamespace(gpx=groups[name]))
        if name in groups else None,
        config=SimpleNamespace(tracks=[], project=SimpleNamespace(gpx_dir=tmp_path / "gpx")),
    )  # fmt: skip
    assert group_folder(session, "new") == tmp_path / "gpx" / "new"
    assert group_folder(session, "walked") == tmp_path / "gpx" / "walked"
    assert group_folder(session, "abs") == tmp_path / "gpx" / "walked"
    assert group_folder(session, "single") is None  # (one file: no folder to add to)
    assert group_folder(session, "glob") == tmp_path / "gpx"
    text = add_group_text("[project]\n", session, "ferries", flight=False)
    assert text.count("[[tracks]]") == 2 and 'name = "tracks"' in text  # (the implicit group)
    session.config.tracks = ["walked"]
    text = add_group_text("[project]\n\n\n", session, "flights", flight=True)
    assert text.count("[[tracks]]") == 1 and 'elevation = "gpx"' in text


def test_new_line_into_a_single_file_group_or_offline(qtbot, tmp_path, monkeypatch):
    from earthling.app import line_dialog
    from earthling.app.main_window import MainWindow

    gpx = gpx_text({"walk": walk(46.0, 7.0, "2026-07-01T08:00:00Z", 10, north_m=60.0)})
    folder = make_project(tmp_path / "trip", {"walk.gpx": gpx},
                          extra='\n[[tracks]]\nname = "walk"\ngpx = "gpx/walk.gpx"\n')  # fmt: skip
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(folder)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    choices = {"accept": False, "group": "walk", "kind": "straight"}

    class Dialog(line_dialog.LineDialog):
        def exec(self):
            self.places = {"from": (46.0, 7.0), "to": (46.1, 7.1)}
            if choices["group"] == "new":
                self.group.setCurrentIndex(self.group.findData(line_dialog.NEW_GROUP))
            self.kind.setCurrentIndex(self.kind.findData(choices["kind"]))
            return choices["accept"]

    monkeypatch.setattr(line_dialog, "LineDialog", Dialog)
    window.new_line()  # (cancelled)
    assert warnings == []
    choices["accept"] = True
    window.new_line()
    assert "is a single GPX file" in warnings[-1]
    choices.update(group="new", kind="road")  # the road route needs the internet
    window.new_line()
    assert warnings[-1].startswith("Cannot make the line")
    assert window.session.group("travel") is None and len(window.tracks) == 1


# --- downloads -------------------------------------------------------------------------------
def fake_job(name, total=3, errors=(), fail=None, gate=None):
    """A download of ``total`` tiles; ``gate``: an Event the job waits for (until cancelled)."""
    cancelled = threading.Event()

    def run(report):
        for done in range(1, total + 1):
            if gate is not None:
                while not (gate.is_set() or cancelled.is_set()):
                    gate.wait(0.01)
            report(DownloadProgress(total=total, done=done, downloaded=done, bytes=1000 * done))
        if fail:
            raise RuntimeError(fail)
        return DownloadProgress(total=total, done=total, downloaded=total - len(errors),
                                failed=len(errors), errors=list(errors))  # fmt: skip

    job = DownloadJob(name, run, cancelled.set)
    job.cancelled = cancelled
    return job


@pytest.fixture
def session(tmp_path):
    from earthling.core.config import Project
    from earthling.session import Session

    gpx = gpx_text({"walk": walk(46.0, 7.0, None, 20, north_m=60.0)})
    return Session(Project.load(make_project(tmp_path / "trip", {"walk.gpx": gpx})))


def test_download_dialog_runs_jobs_in_the_background(qtbot, session):
    from earthling.app.download_dialog import DownloadDialog

    requests, queue = [], []
    dialog = DownloadDialog(session, lambda kinds: requests.append(kinds) or queue.pop(0))
    qtbot.addWidget(dialog)
    assert "tiles" in dialog.imagery_box.text() and "Elevation" in dialog.dem_box.text()
    assert dialog.topo_box.isEnabled() == bool(session.topo_providers)
    assert not dialog.topo_box.isChecked()  # (topographic tiles: on demand unless chosen)
    changed = []
    dialog.data_changed.connect(lambda: changed.append(1))
    dialog.imagery_box.setChecked(False)
    dialog.dem_box.setChecked(False)
    queue.append([])
    dialog.start_btn.click()  # nothing chosen: nothing to do
    assert requests == [set()] and dialog.start_btn.isEnabled()
    dialog.imagery_box.setChecked(True)
    dialog.dem_box.setChecked(True)
    queue.append([fake_job("imagery", errors=["1/2/3: HTTP 500"]), fake_job("dem")])
    dialog.start_btn.click()
    assert requests[-1] == {"imagery", "dem"}
    assert not dialog.start_btn.isEnabled() and dialog.cancel_btn.isEnabled()
    qtbot.waitUntil(dialog.start_btn.isEnabled, timeout=5000)
    log = dialog.log.toPlainText()
    assert "imagery: done – 2 downloaded, 0 already cached, 0 unavailable, 1 failed" in log
    assert "  1/2/3: HTTP 500" in log and "dem: done – 3 downloaded" in log
    assert changed == [1, 1] and dialog.bar.value() == dialog.bar.maximum() == 3
    assert not dialog.cancel_btn.isEnabled()
    # an unexpected error is shown, the dialog stays usable
    queue.append([fake_job("imagery", fail="boom")])
    dialog.start_btn.click()
    qtbot.waitUntil(dialog.start_btn.isEnabled, timeout=5000)
    assert "ERROR: RuntimeError: boom" in dialog.log.toPlainText()


def test_download_dialog_cancels_and_closes_while_running(qtbot, session):
    from earthling.app.download_dialog import DownloadDialog

    gate = threading.Event()
    jobs = [fake_job("imagery", gate=gate), fake_job("dem")]
    dialog = DownloadDialog(session, lambda kinds: jobs)
    qtbot.addWidget(dialog)
    dialog.start_btn.click()
    qtbot.waitUntil(lambda: dialog._worker._current is jobs[0], timeout=5000)
    dialog.cancel_btn.click()
    assert "cancelling…" in dialog.log.toPlainText() and jobs[0].cancelled.is_set()
    qtbot.waitUntil(dialog.start_btn.isEnabled, timeout=5000)
    assert "dem: done" not in dialog.log.toPlainText()  # (the next job did not start)
    # closing while a download runs cancels it and waits for the thread
    jobs[:] = [fake_job("imagery", gate=threading.Event())]
    dialog.start_btn.click()
    qtbot.waitUntil(lambda: dialog._worker._current is jobs[0], timeout=5000)
    dialog.close()
    assert jobs[0].cancelled.is_set() and not dialog._thread.isRunning()


# --- video export ----------------------------------------------------------------------------
class FakeWriter:
    """Records frames; ``fail_at``: write raises at that frame; ``fail_close``: close raises."""

    instances: list = []
    fail_at = None
    fail_close = False
    fail_open = False

    def __init__(self, path, width, height, fps, preset):
        if FakeWriter.fail_open:
            raise OSError("disk full")
        self.path, self.size, self.frames, self.state = path, (width, height), 0, "open"
        FakeWriter.instances.append(self)

    def write(self, image):
        from earthling.export.video import VideoWriterError

        if self.frames == FakeWriter.fail_at:
            raise VideoWriterError("encoder died")
        self.frames += 1

    def close(self):
        from earthling.export.video import VideoWriterError

        self.state = "closed"
        if FakeWriter.fail_close:
            raise VideoWriterError("cannot finish the file")

    def abort(self):
        self.state = "aborted"


class FakeFrames:
    released = 0

    def iter_frames(self, times, width, height, bits):
        for _ in times:
            yield np.zeros((height, width, 3), dtype=np.uint8)

    def release(self):
        FakeFrames.released += 1


@pytest.fixture
def export(qtbot, monkeypatch):
    """The export dialog of a window without a project; frames and the writer are fakes."""
    from earthling.app import export_dialog
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.viewport._timer.stop()  # (the fake renderer below cannot paint)
    monkeypatch.setattr(export_dialog, "VideoWriter", FakeWriter)
    monkeypatch.setattr(FakeWriter, "instances", [])
    monkeypatch.setattr(window, "frame_renderer", FakeFrames)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    dialog = export_dialog.ExportDialog(window)
    qtbot.addWidget(dialog)
    dialog.errors = errors
    return dialog


def test_export_dialog_names_the_file_and_needs_the_gl_context(export, monkeypatch):
    window = export.main_window
    presets = [export.preset.itemData(i) for i in range(export.preset.count())]
    assert export.path.text().endswith(f"earthling{presets[0].extension}")
    if len(presets) > 1:
        export.preset.setCurrentIndex(1)
        assert export.path.text().endswith(presets[1].extension)
    chosen = {"path": ""}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (chosen["path"], ""))
    before = export.path.text()
    export._browse()  # (cancelled)
    assert export.path.text() == before
    chosen["path"] = "C:/videos/trip.mov"
    export._browse()
    assert export.path.text() == "C:/videos/trip.mov"
    export.start_export()  # (no GL context yet: nothing happens)
    assert not export.running and FakeWriter.instances == []
    window.viewport.renderer = object()
    export.end.setValue(0.0)  # (an empty range)
    export.start_export()
    assert not export.running
    window.viewport.renderer = None


def test_export_dialog_renders_frame_by_frame(export, qtbot, monkeypatch):
    from earthling.app import export_dialog

    window = export.main_window
    window.viewport.renderer = object()
    # a first frame faster than the clock ticks (15.6 ms on Windows) divided by zero
    monkeypatch.setattr(export_dialog.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(FakeWriter, "fail_at", None)
    export.resolution.setCurrentIndex(export.resolution.findData("1280x720"))
    export.end.setValue(0.2)
    export.export_btn.click()
    assert export.running and window.viewport.suspended and export.cancel_btn.text() == "Cancel"
    qtbot.waitUntil(lambda: not export.running, timeout=5000)
    writer = FakeWriter.instances[-1]
    fps = window.scene.animation.fps
    assert writer.size == (1280, 720) and writer.frames == round(0.2 * fps)
    assert writer.state == "closed" and export.status.text().startswith("Done")
    assert not window.viewport.suspended and export.export_btn.isEnabled()
    assert window.timeline.time == pytest.approx((writer.frames - 1) / fps)
    # the writer fails mid-way: an error, the file is aborted
    monkeypatch.setattr(FakeWriter, "fail_at", 2)
    export.start_export()
    qtbot.waitUntil(lambda: not export.running, timeout=5000)
    assert FakeWriter.instances[-1].state == "aborted"
    assert export.status.text() == "Failed: encoder died" and export.errors == ["encoder died"]
    # finishing the file fails
    monkeypatch.setattr(FakeWriter, "fail_at", None)
    monkeypatch.setattr(FakeWriter, "fail_close", True)
    export.start_export()
    qtbot.waitUntil(lambda: not export.running, timeout=5000)
    assert export.status.text() == "Failed: cannot finish the file"
    # the file cannot be created at all
    monkeypatch.setattr(FakeWriter, "fail_open", True)
    export.start_export()
    assert not export.running and export.errors[-1] == "disk full"
    monkeypatch.setattr(FakeWriter, "fail_open", False)
    # closing the dialog while it renders cancels
    export.end.setValue(2.0)
    export.start_export()
    export.close()
    assert not export.running and export.status.text() == "Cancelled"
    assert FakeWriter.instances[-1].state == "aborted" and not window.viewport.suspended
    window.viewport.renderer = None


# --- the Animation menu ----------------------------------------------------------------------
def test_animation_menu_generators_are_undoable(qtbot, journey, monkeypatch):  # noqa: F811
    from earthling.app import generators, main_window

    window = main_window.MainWindow()
    qtbot.addWidget(window)
    anim = window.scene.animation
    anim.duration = 2.0
    monkeypatch.setattr(generators, "window_journey", lambda w: None)
    generators.draw_tracks(window)  # (no project: a hint, no dialog)
    assert "tracks first" in window.statusBar().currentMessage()
    monkeypatch.setattr(generators, "window_journey", lambda w: journey)
    accept = {"value": 0}

    def orbit_the_target(dialog):
        dialog.kind.setCurrentIndex(dialog.kind.findData("orbit"))
        dialog.centre.setCurrentIndex(dialog.centre.findData("target"))
        return accept["value"]

    def alone(dialog):
        dialog.alone.setChecked(True)
        dialog.first.setCurrentIndex(1)
        return accept["value"]

    def nights(dialog):
        dialog.use_earliest.setChecked(True)
        dialog.use_latest.setChecked(True)
        return accept["value"]

    monkeypatch.setattr(generators.CameraDialog, "exec", orbit_the_target)
    monkeypatch.setattr(generators.DrawDialog, "exec", alone)
    monkeypatch.setattr(generators.SunDialog, "exec", nights)
    for generate in (generators.generate_camera, generators.draw_tracks,
                     generators.sun_from_tracks):  # fmt: skip
        generate(window)  # (cancelled)
    assert window.undo_stack.count() == 0
    accept["value"] = 1
    window.scene.store.set("camera.target", tuple(journey.start(1)))
    window.scene.store.set("camera.mode", "follow")
    generators.generate_camera(window)
    assert window.scene.store["camera.mode"] == "keys" and anim.is_animated("camera.pose")
    target = np.asarray(journey.start(1))
    pose = anim.curves["camera.pose"].keys[0].value
    assert np.linalg.norm(np.asarray(pose[:3]) - target) == pytest.approx(2500.0, rel=0.05)
    generators.draw_tracks(window)
    assert anim.is_animated("progress.head") and window.scene.store["progress.tail"] > 0.0
    generators.sun_from_tracks(window)
    hours = [k.value.hour for k in anim.curves["sun.datetime"].keys]
    assert min(hours) >= 8 and max(hours) <= 19 and not window.scene.store["sun.follow_track"]
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (2.0, False))
    generators.add_still_seconds(window)  # (cancelled)
    assert anim.duration == 2.0
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (2.0, True))
    generators.add_still_seconds(window)
    assert anim.duration == 6.0
    assert window.undo_stack.count() == 4  # one step each
    while window.undo_stack.canUndo():
        window.undo_stack.undo()
    assert anim.curves.get("camera.pose") is None or not anim.curves["camera.pose"].keys
    assert anim.duration == 2.0 and window.scene.store["camera.mode"] == "follow"
    window.undo_stack.setClean()
