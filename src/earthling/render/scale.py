"""Resolution independence: on-screen sizes are given for a 1080p picture."""

from __future__ import annotations

REFERENCE_PX = 1080.0


def pixel_scale(width: float, height: float) -> float:
    """Factor for sizes given in 1080p pixels: the shorter image side over 1080, so a 4K
    export looks like the 1080p preview and a portrait video (1080 x 1920) keeps the sizes of
    a landscape one instead of growing with its height."""
    return max(min(float(width), float(height)), 1.0) / REFERENCE_PX
