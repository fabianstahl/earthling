"""The main window beyond its menu entries: dropping files, saving and reverting scenes, unsaved
changes, recent projects, the scene frame, the tracks dock, stills and videos, picking."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from earthling.app.main_window import MainWindow
from earthling.app.window_project import MAX_RECENT
from helpers import gpx_text, make_project, walk
from test_photo_editor import photo_file
from test_ui_actions import menu_actions, window  # noqa: F401 (the fixture)

FIXTURES = Path(__file__).parent / "fixtures"


def drop(widget, paths, text: str = "") -> bool:
    """Drag ``paths`` (or ``text``) over ``widget`` and drop them; whether the drag was
    accepted."""
    mime = QMimeData()
    if paths:
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    else:
        mime.setText(text)
    copy, left = Qt.DropAction.CopyAction, Qt.MouseButton.LeftButton
    enter = QDragEnterEvent(QPoint(20, 20), copy, mime, left, Qt.KeyboardModifier.NoModifier)
    enter.ignore()
    QApplication.sendEvent(widget, enter)
    if not enter.isAccepted():
        return False
    QApplication.sendEvent(widget, QDropEvent(QPointF(20, 20), copy, mime, left,
                                              Qt.KeyboardModifier.NoModifier))  # fmt: skip
    return True


def click(widget, pos=(50, 50)) -> None:
    point = QPointF(*pos)
    left = Qt.MouseButton.LeftButton
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
        event = QMouseEvent(kind, point, point, left, left, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, event)


def action(window, path):  # noqa: F811
    return dict(menu_actions(window.menuBar()))[path]


def answer(monkeypatch, button) -> list:
    """The "unsaved changes" question answers ``button``; returns the questions asked."""
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a[1]) or button)  # fmt: skip
    return asked


def test_dropped_gpx_pictures_and_project_folders(window, tmp_path):  # noqa: F811
    tracks = len(window.tracks)
    picture = photo_file(tmp_path / "summit.jpg", (46.0, 7.0))
    assert not drop(window.scene_list, [], "some text")  # (no files: the drag is refused)
    assert drop(window.scene_list, [FIXTURES / "two_tracks.gpx", picture])
    assert (window.project.gpx_dir / "two_tracks.gpx").exists()
    assert len(window.tracks) == tracks + 2
    assert [p.name for p in window.scene.photos] == ["summit"]
    assert window.scene.photos[0].lat == pytest.approx(46.0)
    assert ("photo", 0) in [window.scene_list.current_key()]  # (selected)
    window.undo_stack.setClean()
    other = make_project(tmp_path / "other", {"a.gpx": gpx_text(
        {"walk": walk(46.0, 7.0, "2026-07-01T08:00:00Z", 20, north_m=50.0)})})  # fmt: skip
    assert drop(window.scene_list, [tmp_path / "other"])  # a project folder opens
    assert window.project.folder == other and window.windowTitle().startswith("other")


def test_save_revert_save_as_and_the_window_layout(window, tmp_path, monkeypatch):  # noqa: F811
    store = window.scene.store
    window.set_property("view.fov", 35.0)
    window.history_dock.show()
    assert window.windowTitle().startswith("alps_demo – scene.json*")
    action(window, "File › Save Scene").trigger()
    saved = window.project.folder / "scene.json"
    assert saved.exists() and "*" not in window.windowTitle()
    assert window.scene.ui["window_state"]  # the dock layout is part of the scene
    window.set_property("view.fov", 70.0)
    window.history_dock.hide()
    action(window, "File › Revert Scene").trigger()
    assert store["view.fov"] == 35.0 and not window.scene.dirty
    assert not window.history_dock.isHidden()  # the layout came back with the scene
    assert not window.undo_stack.canUndo()
    # Save As: a new file, the title follows; Open Scene loads another one
    copy = tmp_path / "copy.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(copy), ""))
    window.set_property("view.fov", 45.0)
    action(window, "File › Save Scene As…").trigger()
    assert copy.exists() and window.scene.path == copy and "copy.json" in window.windowTitle()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(saved), ""))
    action(window, "File › Open Scene…").trigger()
    assert window.scene.path == saved and store["view.fov"] == 35.0
    # a scene file that cannot be read: an error, the open scene stays
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[1]))
    assert not window.load_scene(broken)
    assert errors == ["Cannot open scene"] and window.scene.path == saved


def test_unsaved_changes_when_closing_or_opening(window, tmp_path, monkeypatch):  # noqa: F811
    window.set_property("view.fov", 30.0)
    asked = answer(monkeypatch, QMessageBox.StandardButton.Cancel)
    assert not window.close()  # cancelled: the window stays open, the changes too
    assert window.isVisible() and window.scene.dirty and asked == ["Unsaved changes"]
    other = make_project(tmp_path / "other", {})
    assert not window.open_project(other)
    assert window.project.folder.name == "alps_demo"
    window._open_scene_dialog()  # (cancelled before the file dialog)
    assert window.scene.dirty
    answer(monkeypatch, QMessageBox.StandardButton.Save)
    assert window.close()  # Save: saved, then closed
    assert (window.project.folder / "scene.json").exists() and not window.scene.dirty


def test_recent_projects(qtbot, tmp_path, monkeypatch):
    window = MainWindow()  # noqa: F811 (not the fixture)
    qtbot.addWidget(window)
    assert not window.recent_menu.isEnabled()  # (nothing opened yet)
    track = {"a.gpx": gpx_text({"walk": walk(46.0, 7.0, None, 10, north_m=50.0)})}
    first = make_project(tmp_path / "first", track)
    second = make_project(tmp_path / "second", track)
    assert window.open_project(first) and window.open_project(second)
    entries = [a.text() for a in window.recent_menu.actions()]
    assert entries == [str(second), str(first)]  # the latest first
    window.recent_menu.actions()[1].trigger()
    assert window.project.folder == first
    assert [a.text() for a in window.recent_menu.actions()] == [str(first), str(second)]
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[1]))
    assert not window.open_project(tmp_path / "missing")
    assert errors == ["Cannot open project"] and window.project.folder == first
    window.settings.setValue("recent_projects", [f"p{i}" for i in range(20)])
    window._remember_recent(second)
    assert window._recent()[0] == str(second) and len(window._recent()) == MAX_RECENT
    window.settings.setValue("recent_projects", "only")  # (QSettings: one entry is a string)
    assert window._recent() == ["only"]
    # the start screen lists them; leaving it brings the workspace back
    window.show()
    window.show_start(True)
    assert window.central.currentWidget() is window.start_screen
    assert window.toolbar.isHidden() and window.scene_dock.isHidden()
    window.show_start(False)
    assert window.central.currentWidget() is window.viewport
    assert not window.toolbar.isHidden() and not window.scene_dock.isHidden()


def test_new_project_from_gpx_files(qtbot, tmp_path, monkeypatch):
    window = MainWindow()  # noqa: F811 (not the fixture)
    qtbot.addWidget(window)
    gpx = tmp_path / "walk.gpx"
    gpx.write_text(gpx_text({"walk": walk(46.0, 7.0, None, 10, north_m=50.0)}), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(gpx)], ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    window._choose_new_project_gpx()  # (no folder chosen: nothing happens)
    assert window.session is None
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(tmp_path / "trip"))  # fmt: skip
    window.start_screen.choose_new.emit()
    assert window.project.folder == tmp_path / "trip" and len(window.tracks) == 1
    assert (tmp_path / "trip" / "gpx" / "walk.gpx").exists()
    # Add GPX files: copies into the project's folder
    other = tmp_path / "other.gpx"
    other.write_text(gpx_text({"more": walk(46.01, 7.0, None, 10, east_m=50.0)}),
                     encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(other)], ""))
    window.add_gpx_dialog()
    assert sorted(t.name for t in window.tracks) == ["more", "walk"]
    window.undo_stack.setClean()


def test_scene_frame_on_the_view_and_the_tracks_dock(window, monkeypatch):  # noqa: F811
    vp = window.viewport
    frame = window.session.frame
    centre = frame.geodetic_to_enu(46.02, 6.95, 1500.0)
    monkeypatch.setattr(vp, "pick_center", lambda: centre)
    camera_before = frame.enu_to_geodetic(vp.camera.position)
    action(window, "View › Centre Scene Frame on View").trigger()
    assert window.scene.origin == (pytest.approx(46.02), pytest.approx(6.95))
    assert window.session.frame.lat == pytest.approx(46.02)
    # the camera stays where it was on the earth
    camera_after = window.session.frame.enu_to_geodetic(vp.camera.position)
    assert np.allclose(camera_after, camera_before, atol=1e-6)
    action(window, "View › Reset Scene Frame").trigger()
    assert window.scene.origin is None
    window.undo_stack.undo()
    assert window.scene.origin is not None
    window.undo_stack.undo()
    assert window.scene.origin is None
    # no surface under the centre: the orbit target is the centre
    monkeypatch.setattr(vp, "pick_center", lambda: None)
    vp.orbit.target = window.session.frame.geodetic_to_enu(46.0, 6.9, 0.0)
    lat, lon = window._view_centre()
    assert (lat, lon) == (pytest.approx(46.0), pytest.approx(6.9))
    # the tracks dock: hiding a track shows in the scene list; double-click frames it
    item = window.tracks_dock.tree.topLevelItem(0)
    item.setCheckState(0, Qt.CheckState.Unchecked)
    assert not window.tracks[0].visible
    assert window.scene_list._find(("track", 0)).checkState(0) == Qt.CheckState.Unchecked
    item.setCheckState(0, Qt.CheckState.Checked)
    vp.set_mode(vp.FLY)
    window.tracks_dock.tree.itemDoubleClicked.emit(item, 0)
    lat, lon, _ = window.tracks[0].all_points()
    bbox = window.session.frame.geodetic_to_enu(np.array([lat.min(), lat.max()]),
                                                np.array([lon.min(), lon.max()]),
                                                np.zeros(2))  # fmt: skip
    assert vp.mode == vp.ORBIT
    assert bbox[0][0] <= vp.orbit.target[0] <= bbox[1][0]
    assert bbox[0][1] <= vp.orbit.target[1] <= bbox[1][1]
    window._frame_track(99)  # (no such track)
    window.undo_stack.setClean()


def test_stills_and_videos(qtbot, tmp_path, monkeypatch):
    """Render Still asks for a file and renders at the export resolution; Export Video opens
    its dialog; both need the GL context. (No project: no terrain to wait for.)"""
    from earthling.app import export_dialog

    window = MainWindow()  # noqa: F811 (not the fixture)
    qtbot.addWidget(window)
    rendered, opened = [], []
    real_render = window.render_still
    monkeypatch.setattr(window, "render_still", lambda *a: rendered.append(a))
    monkeypatch.setattr(export_dialog.ExportDialog, "exec", lambda d: opened.append(d) or 0)
    window.still_action.trigger()  # (before the GL context: nothing to render with)
    window.export_action.trigger()
    assert rendered == [] and opened == []
    window.show_start(False)
    window.show()
    try:
        qtbot.waitUntil(lambda: window.viewport.renderer is not None, timeout=5000)
    except AssertionError:
        pytest.skip("no OpenGL 4.3 context for the viewport")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    window.still_action.trigger()  # (the file dialog was cancelled)
    assert rendered == []
    still = tmp_path / "still.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(still), ""))
    window.set_property("export.resolution", "1280x720")
    window.still_action.trigger()
    assert rendered == [(still, 1280, 720)]
    assert window.statusBar().currentMessage() == f"Saved {still}"
    window.export_action.trigger()
    assert len(opened) == 1 and opened[0].path.text().endswith("earthling.mov")
    real_render(still, 160, 90)  # a real (small) still at the playhead
    from PIL import Image

    assert Image.open(still).size == (160, 90)
    window.undo_stack.setClean()


def test_picking_places_the_look_at_target_and_pois(window, monkeypatch):  # noqa: F811
    vp = window.viewport
    point = window.session.frame.geodetic_to_enu(45.95, 6.9, 1200.0)
    monkeypatch.setattr(vp, "pick", lambda x, y: point)
    action(window, "View › Pick Look-at Target (T)").trigger()
    assert "look-at target" in window.statusBar().currentMessage()
    click(vp)
    assert window.scene.store["camera.target"] == pytest.approx(tuple(point))
    window.poi_editor.add_by_click()
    click(vp)
    poi = window.scene.pois[0]
    assert (poi.lat, poi.lon) == (pytest.approx(45.95), pytest.approx(6.9))
    assert window.scene_list.current_key() == ("poi", 0)
    # Go To: the camera flies to the coordinates
    from earthling.app.goto_dialog import GoToDialog

    monkeypatch.setattr(GoToDialog, "exec", lambda d: 1)
    monkeypatch.setattr(GoToDialog, "values", lambda d: (46.1, 7.05, 800.0))
    action(window, "View › Go To Coordinate…").trigger()
    lat, lon, _ = vp.camera_geodetic()
    assert vp.mode == vp.FLY and (lat, lon) == (pytest.approx(46.1), pytest.approx(7.05))
    window.undo_stack.setClean()


def test_track_positions_for_pois_photos_and_places(window, monkeypatch):  # noqa: F811
    """The marker's place and the place at a photo's time come from the drawn tracks."""
    if window.viewport.renderer is None:
        pytest.skip("no OpenGL 4.3 context for the viewport")
    window.set_property("progress.head", 0.0)
    lon, lat = window.hiker_lonlat()
    assert (lat, lon) == (pytest.approx(45.9237, abs=1e-3), pytest.approx(6.8694, abs=1e-3))
    # 08:30 local time (Europe/Paris) is the first point of day one
    lon, lat = window.lonlat_at_time(dt.datetime(2026, 7, 1, 8, 30))
    assert (lat, lon) == (pytest.approx(45.9237, abs=1e-3), pytest.approx(6.8694, abs=1e-3))
    assert window.lonlat_at_time(dt.datetime(2026, 8, 1, 12, 0)) is None  # (not on the trip)
    # the place dialog chooses a point of a drawn track
    from earthling.app.place_dialog import PlaceDialog

    def choose_end(dialog):
        dialog.point.setCurrentIndex(dialog.point.findData("end"))
        return 1

    monkeypatch.setattr(PlaceDialog, "exec", choose_end)
    lon, lat = window.choose_place("Where?", None)
    end = window.built_tracks()[0].track.segments[-1]
    assert (lat, lon) == (pytest.approx(end.lat[-1], abs=1e-3),
                          pytest.approx(end.lon[-1], abs=1e-3))  # fmt: skip
    monkeypatch.setattr(PlaceDialog, "exec", lambda d: 0)
    assert window.choose_place("Where?", None) is None
    window.undo_stack.setClean()
