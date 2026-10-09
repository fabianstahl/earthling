"""The colour ramps of shaders/colormaps.glsl on the CPU (e.g. for the stats overlay)."""

from __future__ import annotations

import numpy as np

_HYPSO = np.array([(0.33, 0.55, 0.30), (0.62, 0.72, 0.42), (0.87, 0.80, 0.55),
                   (0.72, 0.55, 0.38), (0.62, 0.58, 0.56), (0.98, 0.98, 1.00)])  # fmt: skip
_VIRIDIS = np.array([(0.2777, 0.0054, 0.3340), (0.1050, 1.4046, 1.3845),
                     (-0.3308, 0.2148, 0.0950), (-4.6342, -5.7991, -19.3324),
                     (6.2282, 14.1799, 56.6905), (4.7763, -13.7451, -65.3530),
                     (-5.4355, 4.6459, 26.3124)])  # fmt: skip
_TURBO = (np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234]),
          np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333]),
          np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771]),
          np.array([-152.94239396, 59.28637943]), np.array([4.27729857, 2.82956604]),
          np.array([-89.90310912, 27.34824973]))  # fmt: skip


def colormap(ramp: str, t) -> np.ndarray:
    """sRGB colours (..., 3) of ramp ``ramp`` ("hypsometric", "viridis", "turbo",
    "grayscale") at ``t`` (clamped to 0..1)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    if ramp == "grayscale":
        return np.stack([t, t, t], axis=-1)
    if ramp == "viridis":
        out = np.zeros(t.shape + (3,))
        for c in _VIRIDIS[::-1]:
            out = out * t[..., None] + c
        return np.clip(out, 0.0, 1.0)
    if ramp == "turbo":
        kr, kg, kb, kr2, kg2, kb2 = _TURBO
        v4 = np.stack([np.ones_like(t), t, t * t, t**3], axis=-1)
        v2 = v4[..., 2:] * v4[..., 2:3]
        rgb = np.stack([v4 @ kr + v2 @ kr2, v4 @ kg + v2 @ kg2, v4 @ kb + v2 @ kb2], axis=-1)
        return np.clip(rgb, 0.0, 1.0)
    x = t * 5.0
    i = np.minimum(np.floor(x), 4).astype(int)
    f = (x - i)[..., None]
    return _HYPSO[i] * (1.0 - f) + _HYPSO[i + 1] * f
