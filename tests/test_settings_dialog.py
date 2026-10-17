import shutil

from earthling.app.settings_dialog import SettingsDialog, check_settings
from helpers import DEMO


def test_check_settings():
    assert check_settings((DEMO / "earthling.toml").read_text(encoding="utf-8")) == ""
    assert "area.border_km" in check_settings("[area]\nborder_km = -1\n")
    assert "invalid TOML" in check_settings("[project\n")
    assert "bogus" in check_settings("[project]\nbogus = 1\n")


def test_settings_dialog_saves_and_the_scene_stays_open(qtbot, tmp_path):
    from earthling.app.main_window import MainWindow

    project = tmp_path / "demo"
    shutil.copytree(DEMO, project)
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.open_project(project)
    other = project / "scenes" / "other.json"
    window.save_scene(other)
    dialog = SettingsDialog(window.project.config_path, window)
    qtbot.addWidget(dialog)
    text = dialog.editor.toPlainText()
    dialog.editor.setPlainText(text + "\n[area\n")
    assert not dialog.check()
    dialog.editor.setPlainText(text.replace('timezone', '# (edited)\ntimezone', 1))
    assert dialog.check()
    dialog._save()
    assert "# (edited)" in window.project.config_path.read_text(encoding="utf-8")
    window._reload_project()
    assert window.scene.path == other
