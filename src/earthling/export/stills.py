"""Preview stills of a scene: single frames at chosen times and a contact sheet, the quick
check before a long video render (``earthling stills``)."""

from __future__ import annotations

import ast
import datetime
import time as clock
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from earthling.export.frames import FrameRenderer, save_png


def parse_override(text: str) -> tuple[str, object]:
    """``pid=value``: a Python literal (``0.5``, ``(1, 0, 0)``, ``True``), ``dt:<ISO date>``
    for date/time properties, anything else as text."""
    pid, value = text.split("=", 1)
    if value.startswith("dt:"):  # dt:2026-09-05T18:30
        return pid, datetime.datetime.fromisoformat(value[3:])
    try:
        return pid, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return pid, value


def apply_overrides(scene, overrides: Iterable[str]) -> None:
    """Set properties to fixed values for the stills (their animation curves are dropped)."""
    for item in overrides:
        pid, value = parse_override(item)
        scene.animation.curves.pop(pid, None)
        scene.store.set(pid, value)


def auto_times(duration: float, count: int, fps: float = 60.0) -> list[float]:
    """``count`` evenly spaced times from the first to the last frame."""
    last = max(duration - 1.0 / fps, 0.0)
    return [round(float(t), 2) for t in np.linspace(0.0, last, max(count, 1))]


def contact_sheet(paths: list[Path], out: Path, width: int, height: int) -> Path:
    """All stills on one image (2 columns up to 4 stills, else 3), saved as JPEG."""
    from PIL import Image

    cols = 2 if len(paths) <= 4 else 3
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height))
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image, ((i % cols) * width, (i // cols) * height))
    sheet.save(out, quality=90)
    return out


def render_stills(frames: FrameRenderer, times: Iterable[float], width: int, height: int,
                  out_dir: Path, stem: str, sequence_from: float | None = None,
                  report: Callable[[str], None] = print) -> list[Path]:  # fmt: skip
    """Render ``times`` to ``<out_dir>/<stem>_<time>.png``. With ``sequence_from`` every frame
    from that time on is rendered first, as in the video export (for effects that depend on
    the frames before, e.g. labels fading in as the head passes them)."""
    times = sorted(times) if sequence_from is not None else list(times)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sequence_from is not None and times:
        fps = frames.animation.fps
        t = sequence_from
        while t < times[-1] - 0.5 / fps:
            frames.render(t, width, height)
            t += 1.0 / fps
    labels = frames.renderer.labels
    paths = []
    for t in times:
        started = clock.monotonic()
        image = frames.render(t, width, height)
        path = out_dir / f"{stem}_{t:06.2f}.png"
        save_png(image, path)
        paths.append(path)
        report(f"{path.name}  {clock.monotonic() - started:.1f} s  "
               f"labels {labels.last_drawn}/{len(labels.features)}")  # fmt: skip
    return paths
