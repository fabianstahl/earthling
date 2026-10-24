from pathlib import Path

from PyQt6.QtCore import Qt

from helpers import DEMO

FIXTURES = Path(__file__).parent / "fixtures"


def make_window(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def keys(scene_list) -> list[tuple]:
    out = []

    def walk(item):
        out.append(item.data(0, Qt.ItemDataRole.UserRole))
        for i in range(item.childCount()):
            walk(item.child(i))

    for i in range(scene_list.tree.topLevelItemCount()):
        walk(scene_list.tree.topLevelItem(i))
    return out


def test_scene_list_shows_groups_tracks_and_selects_them(qtbot):
    window = make_window(qtbot)
    assert window.open_project(DEMO)
    shown = keys(window.scene_list)
    assert ("camera",) in shown and ("track", 0) in shown and ("track", 1) in shown
    group = window.session.groups[0]
    assert ("group", group.name) in shown
    window.scene_list.select(("group", group.name))
    assert window.inspector.title.text() == group.label
    assert f"Tracks: {group.label}" in window.inspector.sections
    window.scene_list.select(("camera",))
    assert window.inspector.category == "camera" and window.inspector.item is None
    track = window.scene_list._find(("track", 0))
    track.setCheckState(0, Qt.CheckState.Unchecked)  # hide a track
    assert not window.tracks[0].visible
    window.tracks[0].visible = True
    window.undo_stack.setClean()


def test_pois_are_added_and_shown_with_their_form(qtbot):
    window = make_window(qtbot)
    window.hiker_lonlat = lambda: (7.0, 46.0)
    window.poi_editor.hiker_position = window.hiker_lonlat
    poi = window.poi_editor.add_at_hiker()
    assert window.scene_list.current_key() == ("poi", 0)
    assert window.inspector.item["top"] is window.poi_editor and not window.poi_editor.isHidden()
    assert f"poi.{poi.id}.visible" in window.inspector.editors
    window.inspector.show_category("weather")  # the form survives other views
    window.scene_list.select(("poi", 0))
    assert window.poi_editor.name.text() == poi.name
    window.scene_list.select(("camera",))
    assert window.poi_editor.isHidden()
    window.poi_editor.remove_index(0)
    assert window.scene.pois == [] and ("poi", 0) not in keys(window.scene_list)
    window.undo_stack.setClean()


def test_dropped_gpx_files_join_the_project(qtbot, tmp_path):
    from earthling.core.config import Project

    Project.create(tmp_path / "trip")
    window = make_window(qtbot)
    assert window.open_project(tmp_path / "trip")
    assert window.tracks == []
    window.drop_paths([FIXTURES / "two_tracks.gpx"])
    assert (window.project.gpx_dir / "two_tracks.gpx").exists()
    assert len(window.tracks) == 2 and ("track", 1) in keys(window.scene_list)
    window.undo_stack.setClean()


def two_track_window(qtbot, tmp_path):
    """A window with a small project: one group of two tracks, a POI and a photo."""
    from earthling.core.photos import Photo
    from earthling.core.pois import Poi
    from helpers import gpx_text, make_project, walk

    gpx = gpx_text({"up": walk(46.0, 7.0, "2026-07-01T08:00:00Z", 20, north_m=60.0),
                    "down": walk(46.02, 7.0, "2026-07-02T08:00:00Z", 20, east_m=60.0)})  # fmt: skip
    window = make_window(qtbot)
    window.resize(1200, 800)
    assert window.open_project(make_project(tmp_path / "trip", {"walk.gpx": gpx}))
    window.scene.set_pois([Poi("hut", "Hut", 7.0, 46.01)])
    window.scene.set_photos([Photo("view", "View", "view.jpg", 46.0, 7.0)])
    window.scene_list.refresh()
    return window


def context_menu(scene_list, key) -> None:
    """Right-click on the row of ``key`` (None: on the empty area below the rows)."""
    from PyQt6.QtCore import QPoint

    tree = scene_list.tree
    tree.doItemsLayout()
    if key is None:
        pos = QPoint(5, tree.viewport().height() - 2)
    else:
        item = scene_list._find(key)
        tree.scrollToItem(item)
        pos = tree.visualItemRect(item).center()
    tree.customContextMenuRequested.emit(pos)


def test_scene_list_context_menus_add_remove_and_frame(qtbot, tmp_path, monkeypatch):
    from helpers import menu_choices

    window = two_track_window(qtbot, tmp_path)
    sl = window.scene_list
    sl.tree.resize(300, 600)
    framed = []
    monkeypatch.setattr(window, "_frame_track", framed.append)
    shown = menu_choices(monkeypatch, "Frame in the view", None, None, None, "Remove")
    context_menu(sl, ("track", 1))
    assert framed == [1] and list(shown[-1]) == ["Frame in the view"]
    context_menu(sl, ("header", "Tracks"))
    assert list(shown[-1]) == ["Add GPX files…", "New line (flight, ferry, road)…"]
    context_menu(sl, ("header", "Points of interest"))
    ends = ("at the marker", "where I click…", "at a place…")  # (of the kind chosen last)
    assert [t.endswith(end) for t, end in zip(shown[-1], ends, strict=True)] == [True] * 3
    context_menu(sl, ("header", "Photos"))
    assert list(shown[-1]) == ["Add photos…"]
    context_menu(sl, ("poi", 0))  # Remove (undoable)
    assert window.scene.pois == [] and ("poi", 0) not in keys(sl)
    window.undo_stack.undo()
    assert [p.id for p in window.scene.pois] == ["hut"] and ("poi", 0) in keys(sl)
    count = len(shown)
    context_menu(sl, None)  # (no row: no menu)
    assert len(shown) == count
    # double-click on a track frames it too
    sl.tree.itemDoubleClicked.emit(sl._find(("track", 0)), 0)
    sl.tree.itemDoubleClicked.emit(sl._find(("poi", 0)), 0)  # (other rows: nothing)
    assert framed == [1, 0]
    # the "+" button: tracks, POIs and photos in submenus
    menu = sl.add_button.menu()
    menu.aboutToShow.emit()
    assert [a.text() for a in menu.actions()] == ["Tracks", "Point of interest", "Photos"]
    window.undo_stack.setClean()


def test_scene_list_keys_checkboxes_and_categories(qtbot, tmp_path):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    window = two_track_window(qtbot, tmp_path)
    sl, store = window.scene_list, window.scene.store

    def delete() -> None:
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(sl, event)

    sl.select(("track", 0))
    delete()  # (tracks are not deleted from the scene list)
    assert len(window.tracks) == 2
    sl.select(("photo", 0))
    assert window.inspector.item["top"] is window.photo_editor
    delete()
    assert window.scene.photos == []
    sl.select(("poi", 0))
    delete()
    assert window.scene.pois == []
    # the group's checkbox hides the group (an undoable setting), overlays switch on
    group = window.session.groups[0].name
    sl._find(("group", group)).setCheckState(0, Qt.CheckState.Unchecked)
    assert store[f"trackgroup.{group}.visible"] is False
    window.undo_stack.undo()
    assert store[f"trackgroup.{group}.visible"] is True
    borders = next(i for i in sl._all_items() if i.data(1, Qt.ItemDataRole.UserRole)
                   == "borders.overlay")  # fmt: skip
    borders.setCheckState(0, Qt.CheckState.Checked)
    assert store["borders.overlay"] is True
    # a second map layer appears with its share
    window.set_property("layers.mix", 0.4)
    layers = [i.text(1) for i in sl._all_items() if i.data(0, Qt.ItemDataRole.UserRole)
              == ("layers",)]  # fmt: skip
    assert "B · 40%" in layers
    # headers and rows open their inspector categories
    for key, category in ((("header", "Tracks"), "tracks"), (("layers",), "layers"),
                          (("header", "Map layers"), "layers"),
                          (("camera",), "camera")):  # fmt: skip
        window.inspector.show_category("weather")
        sl.select(key)
        assert window.inspector.category == category, key
    sl.select(("track", 1))
    assert window.inspector.title.text() == "down"
    # collapsed headers stay collapsed when the list is rebuilt; the selection stays
    sl._find(("header", "Photos")).setExpanded(False)
    sl._find(("group", group)).setExpanded(True)
    window.keys.set_key("camera.pose", window.viewport.camera_pose())  # (rebuilds the list)
    assert not sl._find(("header", "Photos")).isExpanded()
    assert sl._find(("group", group)).isExpanded()
    assert sl.current_key() == ("track", 1)
    assert sl._find(("camera",)).text(1) == "1 keys"
    window.undo_stack.setClean()


def test_a_checkbox_does_not_delete_its_own_row(qtbot, tmp_path):
    """Regression (the sporadic access violation in this file): a checkbox's handler rebuilt
    the list at once, deleting the row while QTreeWidgetItem::setData still used it."""
    from PyQt6 import sip

    window = two_track_window(qtbot, tmp_path)
    sl = window.scene_list
    group = window.session.groups[0].name
    for key in (("track", 0), ("group", group)):
        item = sl._find(key)
        item.setCheckState(0, Qt.CheckState.Unchecked)
        assert not sip.isdeleted(item), key
        qtbot.waitUntil(lambda row=item: sip.isdeleted(row))  # (rebuilt right after)
    assert not window.tracks[0].visible and not window.scene.store[f"trackgroup.{group}.visible"]
    assert sl._find(("track", 0)).checkState(0) == Qt.CheckState.Unchecked
    assert sl._find(("group", group)).checkState(0) == Qt.CheckState.Unchecked
    window.undo_stack.setClean()


def test_keys_tick_rename_and_open_the_menu(qtbot, tmp_path, monkeypatch):
    """Space ticks the current row (instead of playing), F2 edits a POI's name, the menu key
    or Shift+F10 opens the row's context menu - all without a mouse."""
    from helpers import menu_choices

    window = two_track_window(qtbot, tmp_path)
    window.show()
    lst, tree = window.scene_list, window.scene_list.tree
    group = next(k for k in keys(lst) if k and k[0] == "group")
    lst.select(group)
    tree.setFocus()
    item = lst._find(group)
    before = item.checkState(0)
    qtbot.keyClick(tree, Qt.Key.Key_Space)
    qtbot.wait(10)
    assert lst._find(group).checkState(0) != before  # (the row was rebuilt meanwhile)
    assert not window.timeline.playing
    lst.select(("poi", 0))
    tree.setFocus()
    qtbot.keyClick(tree, Qt.Key.Key_F2)
    assert window.poi_editor.name.hasFocus() and window.poi_editor.name.text() == "Hut"
    shown = menu_choices(monkeypatch)
    lst.select(("poi", 0))
    tree.setFocus()
    qtbot.keyClick(tree, Qt.Key.Key_Menu)
    qtbot.keyClick(tree, Qt.Key.Key_F10, Qt.KeyboardModifier.ShiftModifier)
    assert [list(menu) for menu in shown] == [["Remove"], ["Remove"]]
