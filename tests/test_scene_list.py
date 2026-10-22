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
