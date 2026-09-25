import subprocess

import numpy as np
import pytest

from earthling.export.video import PRESETS, VideoWriter, available_presets, export_video, ffmpeg_exe


def probe(path):
    r = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )
    return r.stderr


@pytest.mark.parametrize("preset_id", ["prores4444", "dnxhr444", "h264", "h265"])
def test_presets_encode(tmp_path, preset_id):
    preset = PRESETS[preset_id]
    if preset not in available_presets():
        pytest.skip(f"{preset.encoder} not available")
    path = tmp_path / f"out{preset.extension}"
    # DNxHR needs sizes that are multiples of 16 / known profiles; 256x144 works for all
    writer = VideoWriter(path, 256, 144, 30.0, preset)
    dtype = np.uint16 if preset.bits == 16 else np.uint8
    top = 65535 if preset.bits == 16 else 255
    for i in range(6):
        frame = np.zeros((144, 256, 3), dtype=dtype)
        frame[:, : 40 * (i + 1)] = top
        writer.write(frame)
    writer.close()
    info = probe(path)
    assert "256x144" in info and "Duration: 00:00:00.20" in info


class FakeFrames:
    def __init__(self):
        self.times = []

    def render(self, t, w, h, bits=8):
        self.times.append(t)
        return np.zeros((h, w, 3), dtype=np.uint16 if bits == 16 else np.uint8)


def test_export_video_frame_times_and_cancel(tmp_path):
    frames = FakeFrames()
    progress = []
    n = export_video(frames, tmp_path / "a.mp4", 64, 64, 10.0, PRESETS["h264"], 1.0, 2.0,
                     on_progress=lambda d, t: progress.append((d, t)))  # fmt: skip
    assert (
        n == 10 and frames.times[0] == pytest.approx(1.0) and frames.times[-1] == pytest.approx(1.9)
    )
    assert progress[-1] == (10, 10)
    frames = FakeFrames()
    n = export_video(frames, tmp_path / "b.mp4", 64, 64, 10.0, PRESETS["h264"], 0.0, 5.0,
                     should_cancel=lambda: len(frames.times) >= 3)  # fmt: skip
    assert n == 3


def test_wrong_frame_shape_is_rejected(tmp_path):
    writer = VideoWriter(tmp_path / "c.mp4", 64, 64, 10.0, PRESETS["h264"])
    with pytest.raises(ValueError):
        writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    writer.abort()


def test_export_dialog_writes_video(qtbot, tmp_path):
    from earthling.app.export_dialog import ExportDialog
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.viewport.renderer is not None, timeout=5000)
    dialog = ExportDialog(window)
    qtbot.addWidget(dialog)
    dialog.preset.setCurrentIndex(dialog.preset.findData(PRESETS["h264"]))
    dialog.resolution.setCurrentIndex(0)
    dialog.start.setValue(0.0)
    dialog.end.setValue(0.1)
    out = tmp_path / "gui.mp4"
    dialog.path.setText(str(out))
    dialog.start_export()
    assert dialog.running and window.viewport.suspended
    qtbot.waitUntil(lambda: not dialog.running, timeout=30000)
    assert not window.viewport.suspended
    assert dialog.status.text().startswith("Done")
    fps = window.scene.animation.fps
    assert dialog.count == round(0.1 * fps)
    assert "1280x720" in probe(out)
    # cancelling mid-way leaves the app usable
    dialog.end.setValue(2.0)
    dialog.start_export()
    dialog._cancel_or_close()
    assert not dialog.running and dialog.status.text() == "Cancelled"
    window.undo_stack.setClean()
