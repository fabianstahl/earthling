"""Render a few times of a showcase scene to PNG (preview before the video render).

    python -m uv run python examples/showcase/tools/stills.py s03_stelvio/scenes/s03.json 0 5 12.5
        [--size 960x540] [--set pid=value] [--tag _test]

Stills go to renders/showcase/stills (with a contact sheet when there are several).
"""

from __future__ import annotations

import argparse
import ast
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moderngl  # noqa: E402
from PIL import Image  # noqa: E402
from sc_common import RENDERS, SHOWCASE  # noqa: E402

from earthling.core.config import Project  # noqa: E402
from earthling.core.scene import Scene  # noqa: E402
from earthling.core.session import Session  # noqa: E402
from earthling.export.frames import FrameRenderer, parse_resolution, save_png  # noqa: E402
from earthling.render.renderer import Renderer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="scene file, relative to examples/showcase")
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--samples", default="1")
    ap.add_argument("--set", action="append", default=[], help="pid=python literal (static)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--project", default=None, help="project folder (default: the scene's)")
    args = ap.parse_args()
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = SHOWCASE / scene_path
    project = Path(args.project) if args.project else scene_path.parent.parent
    session = Session(Project.load(project))
    scene = Scene()
    scene.set_dynamic("track_groups", session.track_group_properties())
    for problem in scene.load(scene_path):
        print("warning:", problem)
    scene.store.set("export.samples", args.samples)
    for item in args.set:
        pid, value = item.split("=", 1)
        scene.animation.curves.pop(pid, None)
        if value.startswith("dt:"):  # dt:2026-09-05T18:30
            scene.store.set(pid, datetime.datetime.fromisoformat(value[3:]))
        else:
            try:
                scene.store.set(pid, ast.literal_eval(value))
            except (ValueError, SyntaxError):
                scene.store.set(pid, value)
    width, height = parse_resolution(args.size)
    ctx = moderngl.create_standalone_context(require=430)
    renderer = Renderer(ctx)
    session.set_origin(scene.origin)
    renderer.set_scene(session.frame, session.all_tracks)
    renderer.set_terrain_source(session.terrain_data(), session.terrain_nodes())
    renderer.store = scene.store
    renderer.timezone = session.config.project.timezone
    renderer.set_labels(session.labels, session)
    renderer.pois.set_pois(scene.pois, scene_path.parent)
    renderer.hud.attribution = session.attribution()
    frames = FrameRenderer(renderer, scene.animation)
    out_dir = RENDERS / "stills"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in args.times:
        started = time.monotonic()
        image = frames.render(t, width, height)
        path = out_dir / f"{scene_path.stem}{args.tag}_{t:06.2f}.png"
        save_png(image, path)
        paths.append(path)
        print(f"{path.name}  {time.monotonic() - started:.1f} s")
    renderer.terrain.shutdown()
    if len(paths) > 1:  # contact sheet
        images = [Image.open(p) for p in paths]
        cols = 2 if len(images) <= 4 else 3
        rows = (len(images) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * width, rows * height))
        for i, im in enumerate(images):
            sheet.paste(im, ((i % cols) * width, (i // cols) * height))
        sheet.save(out_dir / f"{scene_path.stem}{args.tag}_sheet.jpg", quality=90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
