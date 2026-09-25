"""Atmosphere model shared by the sky shader and the CPU lighting.

The expensive part of single scattering is the optical depth towards the sun from every sample
point. It only depends on the height of the point and the zenith angle of the ray, so it is
precomputed once into a 2-D lookup table (Rayleigh and Mie separately, so the density sliders
can scale them freely) and uploaded as a texture.

Table parameterisation (must match ``atmosphere.glsl``); rows are heights, columns the cosine
``mu`` of the ray's zenith angle:

* row coordinate ``v = sqrt(h / ATMOSPHERE_HEIGHT)``
* column coordinate ``u = 0.5 + 0.5 * sign(mu) * sqrt(|mu|)`` (dense around the horizon)
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

PLANET_RADIUS = 6371e3
ATMOSPHERE_RADIUS = 6471e3
ATMOSPHERE_HEIGHT = ATMOSPHERE_RADIUS - PLANET_RADIUS
RAYLEIGH_SCALE_HEIGHT = 8e3
MIE_SCALE_HEIGHT = 1.2e3
RAYLEIGH_BETA = np.array([5.5e-6, 13.0e-6, 22.4e-6])
MIE_BETA = 21e-6
MIE_EXTINCTION = 1.1  # extinction / scattering ratio for aerosols
# Ozone absorbs orange/green light; without it twilight skies turn grey-green.
OZONE_BETA = np.array([0.65e-6, 1.881e-6, 0.085e-6])
OZONE_CENTER = 25e3
OZONE_HALF_WIDTH = 15e3

LUT_HEIGHTS = 64
LUT_ANGLES = 256
BLOCKED_DEPTH = 1e9  # optical depth stored for rays that hit the planet


def mu_from_coord(u: np.ndarray) -> np.ndarray:
    s = 2.0 * u - 1.0
    return np.sign(s) * s * s


def coord_from_mu(mu: np.ndarray | float) -> np.ndarray:
    mu = np.clip(mu, -1.0, 1.0)
    return 0.5 + 0.5 * np.sign(mu) * np.sqrt(np.abs(mu))


def height_from_coord(v: np.ndarray) -> np.ndarray:
    return v * v * ATMOSPHERE_HEIGHT


def coord_from_height(h: np.ndarray | float) -> np.ndarray:
    return np.sqrt(
        np.clip(np.asarray(h, dtype=np.float64), 0.0, ATMOSPHERE_HEIGHT) / ATMOSPHERE_HEIGHT
    )


@lru_cache(maxsize=1)
def optical_depth_lut(steps: int = 400) -> np.ndarray:
    """float32 array (LUT_HEIGHTS, LUT_ANGLES, 3): Rayleigh, Mie and ozone optical depth in
    metres (density-weighted path length) from a point towards the top of the atmosphere."""
    heights = height_from_coord((np.arange(LUT_HEIGHTS) + 0.5) / LUT_HEIGHTS)
    heights[0] = 0.0
    mus = mu_from_coord((np.arange(LUT_ANGLES) + 0.5) / LUT_ANGLES)
    lut = np.zeros((LUT_HEIGHTS, LUT_ANGLES, 3), dtype=np.float64)
    s = (np.arange(steps) + 0.5) / steps
    for i, h in enumerate(heights):
        r = PLANET_RADIUS + h
        # distance to the top of the atmosphere along each direction
        b = r * mus
        c = r * r - ATMOSPHERE_RADIUS**2
        length = -b + np.sqrt(np.maximum(b * b - c, 0.0))
        # rays hitting the planet
        disc_ground = b * b - (r * r - PLANET_RADIUS**2)
        near = -b - np.sqrt(np.maximum(disc_ground, 0))
        blocked = (mus < 0) & (disc_ground > 0) & (near > -1.0)  # 1 m tolerance at h = 0
        # quadratic spacing: dense near the start, where the air is densest
        t = length[:, None] * s[None, :] ** 2
        dt = length[:, None] * 2.0 * s[None, :] / steps
        # start at (0, r), direction (sqrt(1 - mu^2), mu)
        x = t * np.sqrt(np.maximum(1.0 - mus**2, 0.0))[:, None]
        z = r + t * mus[:, None]
        radius = np.sqrt(x * x + z * z)
        alt = np.maximum(radius - PLANET_RADIUS, 0.0)
        lut[i, :, 0] = (np.exp(-alt / RAYLEIGH_SCALE_HEIGHT) * dt).sum(axis=1)
        lut[i, :, 1] = (np.exp(-alt / MIE_SCALE_HEIGHT) * dt).sum(axis=1)
        lut[i, :, 2] = (ozone_density(alt) * dt).sum(axis=1)
        lut[i, blocked, :] = BLOCKED_DEPTH
    return lut.astype(np.float32)


def ozone_density(h: np.ndarray) -> np.ndarray:
    """Relative ozone density: a tent around 25 km (Bruneton's model)."""
    return np.maximum(0.0, 1.0 - np.abs(h - OZONE_CENTER) / OZONE_HALF_WIDTH)


def lookup_optical_depth(height_m: float, mu: float) -> tuple[float, float, float]:
    """Bilinear lookup of the LUT (same filtering as the GPU texture)."""
    lut = optical_depth_lut()
    v = float(coord_from_height(height_m)) * LUT_HEIGHTS - 0.5
    u = float(coord_from_mu(mu)) * LUT_ANGLES - 0.5
    i0 = int(np.clip(math.floor(v), 0, LUT_HEIGHTS - 2))
    j0 = int(np.clip(math.floor(u), 0, LUT_ANGLES - 2))
    fv = float(np.clip(v - i0, 0.0, 1.0))
    fu = float(np.clip(u - j0, 0.0, 1.0))
    top = lut[i0, j0] * (1 - fu) + lut[i0, j0 + 1] * fu
    bottom = lut[i0 + 1, j0] * (1 - fu) + lut[i0 + 1, j0 + 1] * fu
    od = top * (1 - fv) + bottom * fv
    return float(od[0]), float(od[1]), float(od[2])


def sun_transmittance(
    elevation_deg: float, height_m: float, rayleigh_scale: float = 1.0, mie_scale: float = 1.0
) -> np.ndarray:
    """Transmittance of sunlight at ``height_m`` for a sun at ``elevation_deg``."""
    od_r, od_m, od_o = lookup_optical_depth(height_m, math.sin(math.radians(elevation_deg)))
    tau = (
        RAYLEIGH_BETA * rayleigh_scale * od_r
        + MIE_BETA * MIE_EXTINCTION * mie_scale * od_m
        + OZONE_BETA * rayleigh_scale * od_o
    )
    return np.exp(-np.minimum(tau, 80.0))
