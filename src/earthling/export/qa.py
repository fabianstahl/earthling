"""Quality check of a rendered video (``earthling qa``): a contact sheet of frames and a scan
for sudden changes.

The whole video is decoded at 480x270. A frame whose mean difference to the previous one is
much larger than its neighbourhood's (pop-in of tiles or symbols, flicker, a jerk of the
camera, a jump of the light) is a *jump*; a short dip or peak of the brightness in a region
of a 3 x 3 grid that the motion scan misses is a *flash*. Writes ``<name>_sheet.jpg`` and
``<name>_jumps.jpg`` (before/after pairs of the largest jumps).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

W, H = 480, 270


@dataclass
class QaReport:
    name: str
    frames: int
    fps: float
    diff: np.ndarray  # mean absolute change per frame step
    jumps: list[tuple[int, float, float]] = field(default_factory=list)  # frame, diff, local
    flashes: list[tuple[int, float, str]] = field(default_factory=list)  # frame, change, where

    def text(self) -> str:
        n, fps, diff = self.frames, self.fps, self.diff
        hold = int(round(fps))
        lines = [f"{self.name}: {n} frames at {fps:g} fps ({n / fps:.2f} s), mean change "
                 f"{diff.mean():.2f}, first second {diff[: hold - 1].mean():.2f}, last second "
                 f"{diff[-(hold - 1) :].mean():.2f}"]  # fmt: skip
        for i, dev, where in self.flashes[:12]:
            lines.append(
                f"  flash at {i / fps:6.2f} s (frame {i}): {dev * 100:4.1f} % in the {where}"
            )
        for i, d, base in self.jumps[:20]:
            lines.append(f"  jump at {i / fps:6.2f} s (frame {i}): {d:5.1f} vs {base:4.1f}")
        return "\n".join(lines)


def decode(path: Path) -> tuple[np.ndarray, float]:
    """All frames at W x H (uint8 RGB) and the frame rate."""
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-v", "error", "-i", str(path), "-vf", f"scale={W}:{H}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W, 3)
    info = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True).stderr
    fps = 60.0
    for part in info.split(","):
        if part.strip().endswith(" fps"):
            fps = float(part.strip().split()[0])
    return frames, fps


def find_jumps(diff: np.ndarray, ratio: float) -> list[tuple[int, float, float]]:
    """(frame, change, local median) where the change exceeds ``ratio`` x the median of the 15
    steps around it (and that median by at least 2.5)."""
    jumps = []
    for i in range(len(diff)):
        lo, hi = max(0, i - 15), min(len(diff), i + 16)
        neighbours = np.concatenate([diff[lo:i], diff[i + 1 : hi]])
        base = float(np.median(neighbours)) if len(neighbours) else 0.0
        if diff[i] > max(ratio * base, base + 2.5):
            jumps.append((i + 1, float(diff[i]), base))
    return jumps


def find_flashes(frames: np.ndarray) -> list[tuple[int, float, str]]:
    """Frames where a region of a 3 x 3 grid is > 6 % darker or brighter than the median of
    its 8 neighbours on each side, in runs of at most 4 frames (a lake or a city drifting
    through a corner changes the brightness for much longer)."""
    n = len(frames)
    grid = frames.reshape(n, 3, H // 3, 3, W // 3, 3).mean(axis=(2, 4, 5))  # (n, 3, 3)
    flashes = []
    for i in range(n):
        lo, hi = max(0, i - 8), min(n, i + 9)
        ref = np.median(np.delete(grid[lo:hi], i - lo, axis=0), axis=0)
        dev = np.abs(grid[i] - ref) / np.maximum(ref, 8.0)
        if dev.max() > 0.06:
            r, c = np.unravel_index(int(np.argmax(dev)), dev.shape)
            flashes.append((i, float(dev.max()), ("top", "middle", "bottom")[r] + "-" +
                            ("left", "centre", "right")[c]))  # fmt: skip
    runs, run = [], []
    for i in sorted({f[0] for f in flashes}):
        if run and i != run[-1] + 1:
            runs.append(run)
            run = []
        run.append(i)
    if run:
        runs.append(run)
    short = {i for r in runs if len(r) <= 4 for i in r}
    return [f for f in flashes if f[0] in short]


def check_video(path: Path, out_dir: Path, sheet_frames: int = 12, ratio: float = 3.0,
                report: Callable[[str], None] = print) -> QaReport:  # fmt: skip
    """Decode ``path``, write the contact sheet (and the jump pairs) to ``out_dir``, report."""
    frames, fps = decode(path)
    n = len(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = path.stem
    picks = np.linspace(0, n - 1, sheet_frames).astype(int)
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H))
    for k, i in enumerate(picks):
        sheet.paste(Image.fromarray(frames[i]), ((k % cols) * W, (k // cols) * H))
    sheet.save(out_dir / f"{name}_sheet.jpg", quality=88)
    diff = np.array([np.abs(frames[i + 1].astype(np.int16) - frames[i]).mean()
                     for i in range(n - 1)])  # fmt: skip
    result = QaReport(name, n, fps, diff, find_jumps(diff, ratio), find_flashes(frames))
    report(result.text())
    if result.jumps:
        top = sorted(result.jumps, key=lambda j: -j[1])[:6]
        pairs = Image.new("RGB", (2 * W, len(top) * H))
        for k, (i, _, _) in enumerate(top):
            pairs.paste(Image.fromarray(frames[i - 1]), (0, k * H))
            pairs.paste(Image.fromarray(frames[i]), (W, k * H))
        pairs.save(out_dir / f"{name}_jumps.jpg", quality=88)
    return result
