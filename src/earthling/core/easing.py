"""Easing and interpolation helpers shared by the renderer, the data layer and the scene kit
(pure Python / numpy; the smoothsteps have GLSL semantics)."""

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


def smoother(u: float) -> float:
    """Quintic smootherstep of ``u`` clamped to 0..1 (zero speed and acceleration at the ends)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def smoother_soft(u: float) -> float:
    """Mostly linear with eased ends (the head starts and stops softly)."""
    u = min(max(u, 0.0), 1.0)
    return 0.7 * u + 0.3 * u * u * (3.0 - 2.0 * u)


def lerp(a, b, f):
    """Tuples ``a`` to ``b`` at fraction ``f``."""
    return tuple(float(x + (y - x) * f) for x, y in zip(a, b, strict=True))


def gaussian(values, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 (edges padded with the end values)."""
    values = np.asarray(values, dtype=np.float64)
    if sigma_samples <= 0:
        return values
    radius = int(3 * sigma_samples) + 1
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    kernel /= kernel.sum()
    pad = [(radius, radius)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pad, mode="edge")
    out = np.empty_like(values)
    for i in range(len(values)):
        window = padded[i : i + 2 * radius + 1]
        out[i] = np.tensordot(kernel, window, axes=(0, 0))
    return out


def pchip(xs, ys, x: float) -> float:
    """Monotone cubic (Fritsch-Carlson) interpolation: C1 smooth, no overshoot."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.zeros_like(ys)
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # ends: zero slope (the camera eases in and out)
    i = int(np.searchsorted(xs, x) - 1)
    t = (x - xs[i]) / h[i]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return float(h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])


def pchip_array(xs, ys, xq) -> np.ndarray:
    """Vectorised monotone cubic interpolation; ys (n, ...) sampled at xs, queried at xq."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    flat = ys.reshape(len(xs), -1)
    h = np.diff(xs)
    delta = np.diff(flat, axis=0) / h[:, None]
    m = np.zeros_like(flat)
    w1 = (2 * h[1:] + h[:-1])[:, None]
    w2 = (h[1:] + 2 * h[:-1])[:, None]
    same = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    m[1:-1] = np.where(same, harmonic, 0.0)
    xq = np.clip(np.asarray(xq, dtype=np.float64), xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xq) - 1, 0, len(xs) - 2)
    t = ((xq - xs[i]) / h[i])[:, None]
    hi = h[i][:, None]
    out = ((2 * t**3 - 3 * t**2 + 1) * flat[i] + (t**3 - 2 * t**2 + t) * hi * m[i]
           + (-2 * t**3 + 3 * t**2) * flat[i + 1] + (t**3 - t**2) * hi * m[i + 1])  # fmt: skip
    return out.reshape((len(xq),) + ys.shape[1:])
