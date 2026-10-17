"""Quality check of a rendered video: a contact sheet of frames and a scan for sudden jumps.

    python -m uv run python examples/showcase/tools/qa_video.py renders/showcase/s01_..._1080p60.mp4

The whole video is decoded at 480x270; for every frame the mean absolute difference to the
previous one is computed. A frame whose difference is much larger than its neighbourhood's
(pop-in of tiles or symbols, flicker, a jerk of the camera, a jump of the light) is reported
with its time. Writes renders/showcase/qa/<name>_sheet.jpg and <name>_jumps.jpg (before/after
pairs).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
W, H = 480, 270


def decode(path: Path) -> tuple[np.ndarray, float]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--ratio", type=float, default=3.0, help="jump = diff > ratio x local median")
    args = ap.parse_args()
    path = Path(args.video)
    if not path.is_absolute():
        path = REPO / path
    frames, fps = decode(path)
    n = len(frames)
    out = REPO / "renders" / "showcase" / "qa"
    out.mkdir(parents=True, exist_ok=True)
    name = path.stem
    # contact sheet
    picks = np.linspace(0, n - 1, args.frames).astype(int)
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H))
    for k, i in enumerate(picks):
        sheet.paste(Image.fromarray(frames[i]), ((k % cols) * W, (k // cols) * H))
    sheet.save(out / f"{name}_sheet.jpg", quality=88)
    # jumps
    diff = np.array([np.abs(frames[i + 1].astype(np.int16) - frames[i]).mean()
                     for i in range(n - 1)])  # fmt: skip
    jumps = []
    for i in range(len(diff)):
        lo, hi = max(0, i - 15), min(len(diff), i + 16)
        neighbours = np.concatenate([diff[lo:i], diff[i + 1 : hi]])
        base = float(np.median(neighbours)) if len(neighbours) else 0.0
        if diff[i] > max(args.ratio * base, base + 2.5):
            jumps.append((i + 1, float(diff[i]), base))
    hold = int(round(fps))
    print(
        f"{name}: {n} frames at {fps:g} fps ({n / fps:.2f} s), mean change {diff.mean():.2f}, "
        f"first second {diff[: hold - 1].mean():.2f}, last second {diff[-(hold - 1) :].mean():.2f}"
    )
    for i, d, base in jumps[:20]:
        print(f"  jump at {i / fps:6.2f} s (frame {i}): {d:5.1f} vs {base:4.1f}")
    if jumps:
        top = sorted(jumps, key=lambda j: -j[1])[:6]
        pairs = Image.new("RGB", (2 * W, len(top) * H))
        for k, (i, _, _) in enumerate(top):
            pairs.paste(Image.fromarray(frames[i - 1]), (0, k * H))
            pairs.paste(Image.fromarray(frames[i]), (W, k * H))
        pairs.save(out / f"{name}_jumps.jpg", quality=88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
