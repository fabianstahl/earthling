import pytest

from earthling.core.pois import Poi


def test_poi_dock_add_edit_remove_with_undo(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    dock = window.poi_dock
    scene = window.scene
    dock.add_poi(6.9, 45.9, "Coffee stop")
    assert [p.id for p in scene.pois] == ["coffee_stop"]
    assert "poi.coffee_stop.effect" in window.property_panel.editors  # own panel section
    assert dock.list.count() == 1
    dock.caption.setText("Cappuccino!")
    dock._edit_field("caption", "Cappuccino!")
    assert scene.pois[0].caption == "Cappuccino!"
    scene.store.set("poi.coffee_stop.effect", "bounce")
    scene.animation.set_key("poi.coffee_stop.opacity", 1.0, 0.5)
    dock._icon_chosen(dock.icon.findData("builtin:coffee"))
    assert scene.pois[0].icon == "builtin:coffee"
    dock._move_to(7.0, 46.0)
    assert (scene.pois[0].lon, scene.pois[0].lat) == (7.0, 46.0)
    dock.add_poi(6.8, 45.8, "Closed path")
    dock.list.setCurrentRow(1)
    dock._remove()
    assert [p.id for p in scene.pois] == ["coffee_stop"]
    stack = window.undo_stack
    stack.undo()  # remove
    assert len(scene.pois) == 2
    stack.undo()  # add closed path
    stack.undo()  # move
    assert scene.pois[0].lon == pytest.approx(6.9)
    assert scene.store["poi.coffee_stop.effect"] == "bounce"  # values survive the snapshots
    assert scene.animation.is_animated("poi.coffee_stop.opacity")
    stack.undo()  # icon
    stack.undo()  # caption
    stack.undo()  # add coffee stop
    assert scene.pois == [] and "poi.coffee_stop.effect" not in window.property_panel.editors
    stack.redo()
    assert [p.id for p in scene.pois] == ["coffee_stop"]
    window.undo_stack.setClean()


def test_scene_load_restores_pois_in_the_dock(qtbot, tmp_path):
    from earthling.app.main_window import MainWindow
    from earthling.core.scene import Scene

    scene = Scene()
    scene.set_pois([Poi("camp", "Camp", 6.9, 45.9, "builtin:tent")])
    path = scene.save(tmp_path / "s.json")
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.load_scene(path)
    assert window.poi_dock.list.count() == 1 and window.poi_dock.name.text() == "Camp"
    assert "poi.camp.visible" in window.property_panel.editors
    window.undo_stack.setClean()
