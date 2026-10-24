from pathlib import Path

import pytest

from earthling.core.pois import Poi


def test_poi_editor_add_edit_remove_with_undo(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    dock = window.poi_editor
    scene = window.scene
    dock.add_poi(6.9, 45.9, "Coffee stop")
    assert [p.id for p in scene.pois] == ["coffee_stop"]
    assert "poi.coffee_stop.effect" in window.property_panel.editors  # own panel section
    assert window.scene_list.current_key() == ("poi", 0)  # selected
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
    dock.select(1)
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


def test_scene_load_restores_pois_in_the_scene_list(qtbot, tmp_path):
    from earthling.app.main_window import MainWindow
    from earthling.core.scene import Scene

    scene = Scene()
    scene.set_pois([Poi("camp", "Camp", 6.9, 45.9, "builtin:tent")])
    path = scene.save(tmp_path / "s.json")
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.load_scene(path)
    window.scene_list.select(("poi", 0))
    assert window.poi_editor.name.text() == "Camp"
    assert "poi.camp.visible" in window.property_panel.editors
    window.undo_stack.setClean()


@pytest.fixture
def editor(qtbot, tmp_path, monkeypatch):
    """A POI form on its own: the marker at (7.0, 46.0); picks and place dialogs recorded;
    the user's image library in ``tmp_path``."""
    from types import SimpleNamespace

    from PyQt6.QtGui import QUndoStack

    from earthling.app import poi_editor
    from earthling.core.scene import Scene

    monkeypatch.setattr(poi_editor, "user_images_file", lambda: tmp_path / "poi_images.json")
    fake = SimpleNamespace(hiker=(7.0, 46.0), place=(7.1, 46.1), picks=[], places=[])
    form = poi_editor.PoiEditor(Scene(), QUndoStack(), lambda: fake.hiker, fake.picks.append,
                                lambda: tmp_path,
                                choose_place=lambda title, start: fake.places.append(
                                    (title, start)) or fake.place)  # fmt: skip
    qtbot.addWidget(form)
    form.fake = fake
    return form


def type_into(qtbot, widget, text: str) -> None:
    """Replace the text of a line edit or spin box by typing, then press Enter."""
    from PyQt6.QtCore import Qt

    widget.selectAll()
    qtbot.keyClicks(widget, text)
    qtbot.keyClick(widget, Qt.Key.Key_Return)


def test_new_pois_at_the_marker_a_click_or_a_place(editor):
    selected = []
    editor.current_changed.connect(selected.append)
    editor.fake.hiker = None
    assert editor.add_at_hiker() is None and editor.items() == []  # (no tracks drawn)
    editor.fake.hiker = (7.0, 46.0)
    first = editor.add_at_hiker()
    assert (first.lon, first.lat, first.name, first.icon) == (7.0, 46.0, "Pin", "builtin:pin")
    editor.add_by_click()  # the next click on the terrain places it
    editor.fake.picks[-1](7.2, 46.2)
    editor.add_at_place()
    assert editor.fake.places == [("New POI at…", (7.0, 46.0))]
    editor.fake.place = None  # (the place dialog was cancelled)
    editor.add_at_place()
    assert [(p.id, p.lon) for p in editor.items()] == [("pin", 7.0), ("pin_2", 7.2), ("pin_3", 7.1)]
    assert selected == [0, 1, 2] and editor.current() is editor.items()[2]
    assert editor.item_tooltip(first) == "46.00000, 7.00000"


def test_the_poi_form_edits_with_undo(editor, qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from earthling.app.poi_editor import ADD_IMAGE, CUSTOM
    from helpers import png

    stack = editor.undo_stack
    editor.add_poi(7.0, 46.0)
    poi = editor.current()
    type_into(qtbot, editor.caption, "Lunch")
    type_into(qtbot, editor.size_px, "96")
    type_into(qtbot, editor.cols, "4")
    assert (poi.caption, poi.size_px, poi.sprite_cols) == ("Lunch", 96.0, 4)
    # a kind gives its icon, caption and size; its name while the POI has a kind's name
    editor.kind.activated.emit(editor.kind.findText("Coffee"))
    assert (poi.name, poi.icon, poi.caption) == ("Coffee", "builtin:coffee", "")
    type_into(qtbot, editor.name, "Cafe Bel-Air")
    editor.kind.activated.emit(editor.kind.findText("Tent"))
    assert (poi.name, poi.icon) == ("Cafe Bel-Air", "builtin:tent")
    assert editor.items()[0].id == "pin"  # (renaming keeps the id: keys keep working)
    # the position: typed coordinates, the marker, a click, a place
    editor.lat.setValue(46.25)
    editor.lat.editingFinished.emit()
    assert poi.lat == 46.25
    editor.fake.hiker = (6.9, 45.9)
    editor.move_hiker.click()
    assert (poi.lon, poi.lat) == (6.9, 45.9)
    editor.move_click.click()
    editor.fake.picks[-1](6.8, 45.8)
    assert (poi.lon, poi.lat) == (6.8, 45.8)
    editor.move_place.click()
    assert editor.fake.places[-1] == ("Move Cafe Bel-Air to…", (6.8, 45.8))
    assert (poi.lon, poi.lat) == (7.1, 46.1)
    assert (editor.lat.value(), editor.lon.value()) == (46.1, 7.1)  # the form follows
    # a custom icon file (relative to the scene folder) shows in the icon list
    (tmp_path / "icons").mkdir()
    star = tmp_path / "icons" / "star.png"
    star.write_bytes(png((255, 200, 0), 32))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(star), ""))
    editor.icon.activated.emit(editor.icon.findData(CUSTOM))
    assert Path(poi.icon) == Path("icons/star.png")
    assert editor.icon.currentData() == poi.icon and not editor.item_icon(poi).isNull()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    editor.icon.activated.emit(editor.icon.findData(CUSTOM))  # cancelled: as it was
    assert editor.icon.currentData() == poi.icon
    # "Add image…" adds a kind to the user's own images and selects it
    editor.kind.activated.emit(editor.kind.findData(ADD_IMAGE))  # (cancelled)
    assert editor.current_kind().name == "Pin"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(star), ""))
    editor.kind.activated.emit(editor.kind.findData(ADD_IMAGE))
    assert editor.current_kind().name == "star" and editor.current_kind().category == "My images"
    assert (tmp_path / "poi_images.json").exists()
    # nothing changed: no undo step
    count = stack.count()
    editor._edit_field("caption", poi.caption)
    editor._move_to(poi.lon, poi.lat)
    assert stack.count() == count
    while stack.canUndo():
        stack.undo()
    assert editor.items() == []
    stack.redo()  # (the POI as it was added)
    assert (editor.items()[0].name, editor.items()[0].caption) == ("Pin", "")


def test_removing_pois_selects_a_neighbour(editor):
    from earthling.app.poi_editor import icon_for
    from earthling.render.pois import BADGE

    for lon in (7.0, 7.1, 7.2):
        editor.add_poi(lon, 46.0, f"P{lon}")
    editor.select(2)
    editor.remove.click()
    assert [p.name for p in editor.items()] == ["P7.0", "P7.1"] and editor.current_index == 1
    editor.remove_index(0)
    assert editor.current_index == 0 and editor.current().name == "P7.1"
    editor.remove_index(5)  # (no such POI)
    editor.remove.click()
    assert editor.items() == [] and editor.current() is None
    assert not editor.name.isEnabled() and not editor.remove.isEnabled()  # nothing to edit
    editor.move_place.click()  # (no POI: nothing happens)
    editor._move_to(1.0, 2.0)
    assert not icon_for(BADGE, "10 km").isNull() and icon_for("none").isNull()
