import pytest

from earthling.app.viewport_overlay import export_frame


def test_export_frame_keeps_the_height():
    ultra = export_frame(3440, 1440, 3840, 2160)  # a 21:9 screen, a 16:9 video
    assert ultra.height() == 1440 and ultra.width() == pytest.approx(2560)
    assert ultra.left() == pytest.approx(440)
    portrait = export_frame(1600, 900, 1080, 1920)
    assert portrait.width() == pytest.approx(506.25)
    assert export_frame(1000, 900, 3840, 2160) is None  # the video is wider than the view


def test_overlay_follows_the_window(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    overlay = window.viewport_overlay
    window.viewport.set_mode(window.viewport.FLY)
    assert overlay.mode_buttons[window.viewport.FLY].isChecked()
    window.set_property("export.resolution", "1080x1920")
    assert overlay.shade.resolution == (1080, 1920)
    assert "1080 × 1920" in overlay.frame_label.text()
    overlay.preview.click()
    assert window.scene.store["view.preview"]
    window.undo_stack.setClean()
    window.scene.mark_dirty(False)
