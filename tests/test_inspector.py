from PyQt6.QtCore import QSettings

from earthling.core.properties import PropertyStore
from earthling.render.parameters import build_registry
from earthling.ui.inspector import OTHER, Inspector, category_of, search_terms


def make_inspector(qtbot, tmp_path, store=None):
    store = store or PropertyStore(build_registry())
    settings = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    inspector = Inspector(
        store,
        lambda pid, v, interactive: store.set(pid, v),
        is_animated=lambda pid: pid == "fog.density",
        settings=settings,
    )
    qtbot.addWidget(inspector)
    return inspector, store


def test_every_section_has_a_category():
    for title, _ in PropertyStore(build_registry()).registry.sections():
        assert category_of(title) is not OTHER, title


def test_category_shows_only_its_sections(qtbot, tmp_path):
    inspector, store = make_inspector(qtbot, tmp_path)
    inspector.show_category("weather")
    assert "Height Fog" in inspector.sections and "Tracks" not in inspector.sections
    assert "fog.density" in inspector.editors
    assert inspector.title.text() == "Atmosphere & Weather"
    inspector.show_category("tracks")
    assert "fog.density" not in inspector.editors and "tracks.width" in inspector.editors


def test_search_finds_english_and_german_words(qtbot, tmp_path):
    inspector, store = make_inspector(qtbot, tmp_path)
    assert search_terms("Nebel") == [["nebel", "fog"]]
    inspector.search.setText("nebel")
    inspector.rebuild()
    assert "fog.density" in inspector.editors and "tracks.width" not in inspector.editors
    assert inspector.sections["Height Fog"].flat
    inspector.search.setText("track width")
    inspector.rebuild()
    assert "tracks.width" in inspector.editors and "fog.density" not in inspector.editors
    inspector.editors["tracks.width"].commit(9.0)  # results stay editable
    assert store["tracks.width"] == 9.0
    inspector.jump_to("Tracks")  # the link of a result's section
    assert inspector.search.text() == "" and inspector.category == "tracks"


def test_filters_list_changed_animated_and_pinned(qtbot, tmp_path):
    inspector, store = make_inspector(qtbot, tmp_path)
    store.set("sky.haze", 3.0)
    inspector.set_filter("changed")
    assert set(inspector.editors) == {"sky.haze"}
    assert inspector.chips["changed"].text().endswith("1")
    inspector.set_filter("animated")
    assert set(inspector.editors) == {"fog.density"}
    inspector.set_pinned("labels.size", True)
    inspector.set_filter("pinned")
    assert set(inspector.editors) == {"labels.size"}
    again, _ = make_inspector(qtbot, tmp_path, store)  # pins are remembered
    assert "labels.size" in again.pins
    inspector.set_filter("pinned")  # clicking the active filter again shows all
    assert inspector.filter == "all"


def test_sections_keep_their_state_between_categories(qtbot, tmp_path):
    inspector, store = make_inspector(qtbot, tmp_path)
    inspector.show_category("weather")
    inspector.sections["Sky & Atmosphere"].set_open(False)
    inspector.show_category("camera")
    inspector.show_category("weather")
    assert not inspector.sections["Sky & Atmosphere"].is_open
