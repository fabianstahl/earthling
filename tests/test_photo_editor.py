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
