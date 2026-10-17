import datetime

import numpy as np
from PIL import Image

from earthling.core.scene import Scene
from earthling.export.qa import H, W, check_video, find_flashes, find_jumps
from earthling.export.stills import apply_overrides, auto_times, contact_sheet, parse_override
from earthling.export.video import PRESETS, VideoWriter


def test_parse_override():
    assert parse_override("terrain.exaggeration=1.5") == ("terrain.exaggeration", 1.5)
    assert parse_override("tracks.color=(1, 0, 0)") == ("tracks.color", (1, 0, 0))
    assert parse_override("sun.datetime=dt:2026-09-05T18:30") == (
        "sun.datetime", datetime.datetime(2026, 9, 5, 18, 30)
    )  # fmt: skip
    assert parse_override("camera.mode=follow") == ("camera.mode", "follow")


def test_overrides_replace_the_animation():
    scene = Scene()
    scene.animation.set_key("terrain.exaggeration", 0.0, 1.0)
    scene.animation.set_key("terrain.exaggeration", 1.0, 2.0)
    apply_overrides(scene, ["terrain.exaggeration=3"])
    assert "terrain.exaggeration" not in scene.animation.curves
    assert scene.store["terrain.exaggeration"] == 3.0


def test_auto_times_span_the_scene():
    assert auto_times(10.0, 3, 50.0) == [0.0, 4.99, 9.98]
    assert auto_times(10.0, 1) == [0.0]


def test_contact_sheet(tmp_path):
    paths = []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (40, 20), (50 * i, 0, 0)).save(path)
        paths.append(path)
    out = contact_sheet(paths, tmp_path / "sheet.jpg", 40, 20)
    with Image.open(out) as sheet:
        assert sheet.size == (120, 40)  # 3 columns, 2 rows


def test_jump_and_flash_scan():
    diff = np.full(60, 1.0)
    diff[30] = 12.0
    assert [j[0] for j in find_jumps(diff, 3.0)] == [31]
    frames = np.full((40, H, W, 3), 100, dtype=np.uint8)
    frames[20, : H // 3, : W // 3] = 160  # a short flash top-left
    frames[5:30, -(H // 3) :, -(W // 3) :] = 30  # a long change bottom-right: not a flash
    flashes = {(f[0], f[2]) for f in find_flashes(frames)}
    assert (20, "top-left") in flashes
    assert not any(where == "bottom-right" and 8 <= i <= 26 for i, where in flashes)


def test_check_video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = VideoWriter(path, 160, 90, 30.0, PRESETS["h264"])
    for i in range(45):
        frame = np.full((90, 160, 3), 90, dtype=np.uint8)
        frame[:, : 2 * i + 4] = 200 if i < 30 else 20  # a steady wipe, then a hard cut
        writer.write(frame)
    writer.close()
    lines = []
    report = check_video(path, tmp_path / "qa", sheet_frames=4, report=lines.append)
    assert report.frames == 45 and report.fps == 30.0
    assert [j[0] for j in report.jumps] == [30]
    assert (tmp_path / "qa" / "clip_sheet.jpg").exists()
    assert (tmp_path / "qa" / "clip_jumps.jpg").exists()
    assert lines and lines[0].startswith("clip: 45 frames")
