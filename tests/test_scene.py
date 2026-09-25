import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from earthling.core.scene import MIGRATIONS, SCENE_VERSION, Scene, SceneError, migrate

DEMO = Path(__file__).parents[1] / "examples" / "alps_demo"


def test_roundtrip(tmp_path):
    scene = Scene()
    scene.store.set("terrain.exaggeration", 2.5)
    scene.camera = {"position": [1, 2, 3], "heading": 10, "pitch": -5}
    assert scene.dirty
    path = scene.save(tmp_path / "s.json")
    assert not scene.dirty
    other = Scene()
    assert other.load(path) == []
    assert other.store["terrain.exaggeration"] == 2.5
    assert other.camera["heading"] == 10
    assert other.path == path and not other.dirty


def test_missing_properties_get_defaults_and_unknown_are_reported(tmp_path):
    scene = Scene()
    scene.store.set("terrain.exaggeration", 3.0)
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": SCENE_VERSION, "properties": {"bogus.x": 1}}))
    problems = scene.load(path)
    assert scene.store["terrain.exaggeration"] == 1.0
    assert problems and "bogus.x" in problems[0]


def test_migration_chain(monkeypatch):
    monkeypatch.setitem(MIGRATIONS, 0, lambda d: {**d, "properties": {"view.fov": 30}})
    doc = migrate({"version": 0})
    assert doc["version"] == SCENE_VERSION and doc["properties"]["view.fov"] == 30
    with pytest.raises(SceneError, match="newer"):
        migrate({"version": SCENE_VERSION + 1})


def test_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SceneError):
        Scene().load(bad)
    bad.write_text("[1, 2]")
    with pytest.raises(SceneError):
        Scene().load(bad)


def test_reset():
    scene = Scene()
    scene.store.set("view.fov", 20)
    scene.reset(Path("x.json"))
    assert scene.store["view.fov"] == 50.0 and not scene.dirty and scene.path == Path("x.json")


def test_window_saves_and_restores_scene(qtbot, tmp_path):
    from earthling.app.main_window import MainWindow

    project = tmp_path / "demo"
    shutil.copytree(DEMO / "gpx", project / "gpx")
    shutil.copy(DEMO / "earthling.toml", project / "earthling.toml")
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(project)
    window.scene.store.set("light.azimuth", 42.0)
    window.viewport.camera.position = np.array([10.0, 20.0, 3000.0])
    window.viewport.camera.heading = 77.0
    window.viewport.mode = window.viewport.FLY
    assert window.windowTitle().startswith("demo – scene.json*")
    assert window.save_scene()
    assert (project / "scene.json").exists()
    assert "*" not in window.windowTitle()

    window2 = MainWindow()
    qtbot.addWidget(window2)
    assert window2.open_project(project)
    assert window2.scene.store["light.azimuth"] == 42.0
    assert window2.viewport.camera.heading == pytest.approx(77.0)
    assert window2.viewport.camera.position == pytest.approx([10.0, 20.0, 3000.0])
