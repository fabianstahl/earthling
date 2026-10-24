import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from earthling.core.animation import Interp
from earthling.core.photos import FADE_S, Photo, reached_alpha, read_exif
from earthling.core.scene import Scene
from helpers import FRAME


def photo_file(path, gps=None, taken=None, offset=None):
    image = Image.new("RGB", (64, 48), (120, 160, 200))
    exif = image.getexif()
    if gps is not None:
        lat, lon = gps
        ifd = exif.get_ifd(0x8825)
        ifd[1], ifd[3] = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
        ifd[2] = (float(int(abs(lat))), float(abs(lat) % 1 * 60), 0.0)
        ifd[4] = (float(int(abs(lon))), float(abs(lon) % 1 * 60), 0.0)
    if taken is not None:
        info = exif.get_ifd(0x8769)
        info[36867] = taken
        if offset:
            info[36881] = offset
    image.save(path, exif=exif)
    return path


def test_read_exif(tmp_path):
    lat, lon, taken = read_exif(photo_file(tmp_path / "a.jpg", (45.5, -6.25),
                                           "2026:09:01 12:30:00", "+02:00"))  # fmt: skip
    assert lat == pytest.approx(45.5) and lon == pytest.approx(-6.25)
    assert taken == dt.datetime(2026, 9, 1, 12, 30, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert read_exif(photo_file(tmp_path / "b.jpg", taken="2026:09:01 08:00:00"))[2] == (
        dt.datetime(2026, 9, 1, 8, 0)
    )
    assert read_exif(photo_file(tmp_path / "c.jpg")) == (None, None, None)
    assert read_exif(photo_file(tmp_path / "d.jpg", (0.0, 0.0))) == (None, None, None)


def test_reached_alpha():
    assert reached_alpha(-1.0, 4.0) == 0.0
    assert reached_alpha(FADE_S + 2.0, 4.0) == 1.0
    assert 0.0 < reached_alpha(FADE_S / 2, 4.0) < 1.0
    assert reached_alpha(2 * FADE_S + 4.0, 4.0) == 0.0


def test_photo_editor_places_photos(qtbot, tmp_path):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    dock = window.photo_editor
    dock.place_at_time = lambda when: (7.5, 46.5)
    with_gps = dock.add_photo(photo_file(tmp_path / "gps.jpg", (45.5, 6.5)))
    by_time = dock.add_photo(photo_file(tmp_path / "time.jpg", taken="2026:09:01 08:00:00"))
    assert (with_gps.lat, with_gps.lon) == (pytest.approx(45.5), pytest.approx(6.5))
    assert (by_time.lat, by_time.lon) == (46.5, 7.5)
    assert "photo.time.appear" in window.property_panel.editors  # the selected photo
    assert [p.id for p in window.scene.photos] == ["gps", "time"]
    window.undo_stack.undo()
    assert [p.id for p in window.scene.photos] == ["gps"]


def test_photo_appears_when_reached(gl_ctx):
    from earthling.render.photos import PhotoLayer
    from earthling.render.shader_library import ShaderLibrary

    scene = Scene()
    anim = scene.animation
    anim.duration = 10.0
    anim.set_key("progress.head", 0.0, 0.0, Interp.LINEAR)
    anim.set_key("progress.head", 10.0, 1.0, Interp.LINEAR)
    photo = Photo("p", "P", "p.jpg", 46.0, 7.0)  # (the frame origin: path x = 0)
    scene.set_photos([photo])
    layer = PhotoLayer(gl_ctx, ShaderLibrary(gl_ctx))
    layer.path = SimpleNamespace(total_m=1000.0, distance_for=lambda head, mode: head * 1000.0,
                                 nearest_distances=lambda xy: xy[:, 0] + 300.0)  # fmt: skip
    layer.frame, layer.animation = FRAME, anim
    layer.time = 2.0  # (reached at 3 s)
    assert layer._reached(photo, "photo.p.", scene.store) == 0.0
    layer.time = 5.0
    assert layer._reached(photo, "photo.p.", scene.store) == pytest.approx(1.0)
    assert np.isfinite(layer._passed[next(iter(layer._passed))])


def test_photo_form_adds_renames_and_changes_files(qtbot, tmp_path, monkeypatch):
    """Photos are placed by their GPS, else by their time on the tracks, else at the marker;
    files inside the scene folder are kept relative."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QUndoStack
    from PyQt6.QtWidgets import QFileDialog

    from earthling.app.photo_editor import PhotoEditor

    places = {"marker": (7.0, 46.0), "at_time": None}
    times = []
    base = tmp_path / "scene"  # (the scene folder)
    base.mkdir()

    def at_time(when):
        times.append(when)
        return places["at_time"]

    editor = PhotoEditor(Scene(), QUndoStack(), lambda: places["marker"], lambda cb: None,
                         lambda: base, place_at_time=at_time)  # fmt: skip
    qtbot.addWidget(editor)
    gps = photo_file(base / "gps.jpg", (45.5, 6.5))
    timed = photo_file(base / "timed.jpg", taken="2026:09:01 08:00:00")
    plain = photo_file(base / "plain.jpg")
    chosen = {"files": []}
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: (chosen["files"], ""))
    editor.add_dialog()  # (cancelled)
    assert editor.items() == []
    chosen["files"] = [str(gps), str(timed), str(plain)]
    editor.add_dialog()
    photos = editor.items()
    assert [p.path for p in photos] == ["gps.jpg", "timed.jpg", "plain.jpg"]  # (relative)
    assert (photos[0].lon, photos[0].lat) == (pytest.approx(6.5), pytest.approx(45.5))
    assert times == [dt.datetime(2026, 9, 1, 8, 0)]  # (not on the tracks: at the marker)
    assert (photos[1].lon, photos[1].lat) == (7.0, 46.0) == (photos[2].lon, photos[2].lat)
    places["marker"] = None  # (no tracks drawn either: no place)
    assert editor.add_photo(plain).lat is None
    assert editor.item_tooltip(editor.items()[-1]) == ""
    assert not editor.item_icon(photos[0]).isNull()
    # renaming by typing, another file through the form
    editor.select(0)
    assert editor.file.text() == "gps.jpg"
    editor.name.selectAll()
    qtbot.keyClicks(editor.name, "Summit")
    qtbot.keyClick(editor.name, Qt.Key.Key_Return)
    assert editor.items()[0].name == "Summit" and editor.items()[0].id == "gps"
    other = photo_file(tmp_path / "elsewhere.jpg")  # (outside the scene folder)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), ""))
    editor.change_file.click()
    assert editor.items()[0].path == str(other) and editor.file.text() == str(other)
    assert not editor.item_icon(editor.items()[0]).isNull()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    editor.change_file.click()  # (cancelled)
    assert editor.items()[0].path == str(other)
    editor.undo_stack.undo()
    assert editor.items()[0].path == "gps.jpg"
    editor.select(-1)
    editor._change_file()  # (no photo selected: nothing happens)
