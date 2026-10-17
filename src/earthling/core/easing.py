"""Small easing and interpolation helpers shared by the renderer, the data layer and scene
scripts (pure Python / numpy, GLSL-like semantics)."""

from __future__ import annotations

import numpy as np


def smooth01(x: float) -> float:
    """Hermite smoothstep of ``x`` clamped to 0..1."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    """GLSL ``smoothstep`` for scalars (``e1`` must differ from ``e0``)."""
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def smooth01_array(x) -> np.ndarray:
    """:func:`smooth01` for arrays."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smoothstep_array(e0: float, e1: float, x) -> np.ndarray:
    """GLSL ``smoothstep`` for arrays (float64; equal edges give a step)."""
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
