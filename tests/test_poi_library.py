import numpy as np
import pytest

from earthling.core.poi_library import KINDS, add_user_image, load_user_images
from earthling.core.pois import Poi
from earthling.render.pois import BADGE, badge_image, resolve_icon
from earthling.render.track_geometry import BuiltTrack, track_point
from test_pois import render_poi


def test_every_kind_has_an_icon():
    for kind in KINDS:
        if kind.icon.startswith("builtin:"):
            assert resolve_icon(kind.icon, None) is not None, kind
        else:
            assert kind.icon in (BADGE, "none") and kind.caption, kind
    names = [k.name for k in KINDS]
    assert len(names) == len(set(names))
    assert "100 km" in names and "99 km" not in names


def test_user_images(tmp_path):
    library = tmp_path / "lib" / "poi_images.json"
    image = tmp_path / "hut.png"
    badge_image("Hut").save(image)
    assert load_user_images(library) == []
    kind = add_user_image(library, image)
    add_user_image(library, image)  # (only once)
    assert [k.icon for k in load_user_images(library)] == [kind.icon] and kind.name == "hut"
    image.unlink()
    assert load_user_images(library) == []  # missing files are left out


def test_badge(gl_ctx):
    short, long = badge_image("5 km"), badge_image("42,2 km")
    assert short.height == long.height == 100 and long.width > short.width
    poi = Poi("km10", "10 km", 7.0, 46.0, BADGE, size_px=120.0, caption="10 km",
              height_offset_m=2600.0)  # fmt: skip
    renderer, img = render_poi(gl_ctx, poi)
    assert renderer.pois.last_drawn == ["km10"]
    orange = (img[..., 0] > 200) & (img[..., 1] > 70) & (img[..., 1] < 140) & (img[..., 2] < 80)
    assert orange.sum() > 50  # the rim


def test_track_points():
    n = 101
    enu = np.column_stack([np.linspace(0, 1000, n), np.zeros(n), np.zeros(n)])
    ele = 500.0 + 100.0 * np.sin(np.linspace(0, np.pi, n))
    g = BuiltTrack(None, 0, enu[0], 1000.0, enu=enu, dist=np.linspace(0, 1000, n),
                   ele=ele, ground=np.linspace(0, 2000, n))  # (marks: 2 km walked)  # fmt: skip
    assert track_point(g, "start")[0] == 0.0 and track_point(g, "end")[0] == 1000.0
    assert track_point(g, "highest")[0] == pytest.approx(500.0)
    assert track_point(g, "lowest")[0] in (0.0, 1000.0)
    assert track_point(g, "distance", 1.5)[0] == pytest.approx(750.0)  # 1.5 of 2 walked km
    assert track_point(g, "distance", 9.0)[0] == 1000.0  # clamped


def test_new_poi_of_a_kind_and_by_place(qtbot, monkeypatch):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    dock = window.poi_editor
    km = next(k for k in KINDS if k.name == "30 km")
    poi = dock.add_poi(7.0, 46.0, kind=km)
    assert (poi.icon, poi.caption, poi.name) == (BADGE, "30 km", "30 km")
    dock.kind.setCurrentIndex(dock.kind.findText("Viewpoint"))
    assert dock.add_poi(7.1, 46.1).icon == "builtin:viewpoint"
    dock.choose_place = lambda title, start: (8.0, 47.0)
    dock._move_to_place()
    assert (window.scene.pois[-1].lon, window.scene.pois[-1].lat) == (8.0, 47.0)
    dock.lat.setValue(46.5)
    dock.lat.editingFinished.emit()
    assert window.scene.pois[-1].lat == 46.5


def test_pois_appear_when_the_line_reaches_them(gl_ctx):
    from types import SimpleNamespace

    from earthling.core.animation import Interp
    from earthling.core.scene import Scene
    from earthling.render.pois import PoiLayer
    from earthling.render.progress import ReachTimes
    from earthling.render.shader_library import ShaderLibrary

    scene = Scene()
    anim = scene.animation
    anim.duration = 10.0
    anim.set_key("progress.head", 0.0, 0.0, Interp.LINEAR)
    anim.set_key("progress.head", 10.0, 1.0, Interp.LINEAR)
    path = SimpleNamespace(total_m=1000.0, distance_for=lambda head, mode: head * 1000.0,
                           nearest_distances=lambda xy: xy[:, 0])  # fmt: skip
    reach = ReachTimes()
    assert reach.time_of(500.0, path, anim, scene.store) == pytest.approx(5.0, abs=0.02)
    assert reach.time_of(2000.0, path, anim, scene.store) == float("inf")
    layer = PoiLayer(gl_ctx, ShaderLibrary(gl_ctx), None)
    layer.path = path
    poi = Poi("hut", "Hut", 7.0, 46.0)
    anchor = np.array([250.0, 40.0, 0.0])
    assert layer._reached_since(poi, anchor, 2.0, anim, scene.store) is None
    since = layer._reached_since(poi, anchor, 3.0, anim, scene.store)
    assert since == pytest.approx(0.5, abs=0.02)
