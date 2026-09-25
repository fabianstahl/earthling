"""Geodesy helpers: WGS84, ECEF, local East-North-Up frames and Web Mercator tiles.

Conventions
-----------
* Angles in degrees at API boundaries, arrays are float64.
* ECEF: Earth-centred, Earth-fixed Cartesian coordinates in meters.
* ENU: local tangent frame at an origin. x = east, y = north, z = up. Because points are
  converted through ECEF, Earth curvature is represented exactly (distant points drop below z=0).
* Web Mercator (EPSG:3857) meters; XYZ tiles with y growing southwards (OSM/Google scheme).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

MERCATOR_HALF = math.pi * WGS84_A  # 20037508.342789244
MERCATOR_MAX_LAT = 85.05112877980659


# --- ECEF --------------------------------------------------------------------------------
def geodetic_to_ecef(lat, lon, h=0.0) -> np.ndarray:
    """Returns an array of shape (..., 3)."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bowring's method; sub-millimetre accurate for terrestrial heights."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    lon = np.arctan2(y, x)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


# --- local ENU frame ---------------------------------------------------------------------
@dataclass(frozen=True)
class LocalFrame:
    """East-North-Up tangent frame at a geodetic origin."""

    lat: float
    lon: float
    h: float = 0.0

    @property
    def origin_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat, self.lon, self.h)

    @property
    def rotation(self) -> np.ndarray:
        """3x3 matrix whose rows are the E, N, U unit vectors in ECEF."""
        lat, lon = math.radians(self.lat), math.radians(self.lon)
        sl, cl = math.sin(lat), math.cos(lat)
        so, co = math.sin(lon), math.cos(lon)
        return np.array(
            [
                [-so, co, 0.0],
                [-sl * co, -sl * so, cl],
                [cl * co, cl * so, sl],
            ]
        )

    def ecef_to_enu(self, xyz) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float64) - self.origin_ecef) @ self.rotation.T

    def enu_to_ecef(self, enu) -> np.ndarray:
        return np.asarray(enu, dtype=np.float64) @ self.rotation + self.origin_ecef

    def geodetic_to_enu(self, lat, lon, h=0.0) -> np.ndarray:
        return self.ecef_to_enu(geodetic_to_ecef(lat, lon, h))

    def enu_to_geodetic(self, enu):
        return ecef_to_geodetic(self.enu_to_ecef(enu))

    def up_at(self, enu) -> np.ndarray:
        """Local ellipsoid normal (approx. geocentric up) at an ENU position, in ENU coords."""
        lat, lon, _ = self.enu_to_geodetic(enu)
        up_ecef = geodetic_to_ecef(lat, lon, 1.0) - geodetic_to_ecef(lat, lon, 0.0)
        return up_ecef @ self.rotation.T


def frame_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> LocalFrame:
    return LocalFrame((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0, 0.0)


# --- Web Mercator ------------------------------------------------------------------------
def lonlat_to_mercator(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.clip(np.asarray(lat, dtype=np.float64), -MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
    x = np.radians(lon) * WGS84_A
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * WGS84_A
    return x, y


def mercator_to_lonlat(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = np.degrees(x / WGS84_A)
    lat = np.degrees(2 * np.arctan(np.exp(y / WGS84_A)) - np.pi / 2)
    return lon, lat


def lonlat_to_tile(lon, lat, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional tile coordinates at zoom ``z``."""
    x, y = lonlat_to_mercator(lon, lat)
    n = 2**z
    tx = (x + MERCATOR_HALF) / (2 * MERCATOR_HALF) * n
    ty = (MERCATOR_HALF - y) / (2 * MERCATOR_HALF) * n
    return tx, ty


def tile_to_lonlat(tx, ty, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`lonlat_to_tile` (accepts fractional coordinates)."""
    n = 2**z
    x = np.asarray(tx, dtype=np.float64) / n * 2 * MERCATOR_HALF - MERCATOR_HALF
    y = MERCATOR_HALF - np.asarray(ty, dtype=np.float64) / n * 2 * MERCATOR_HALF
    return mercator_to_lonlat(x, y)


def tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) in Web Mercator meters."""
    size = 2 * MERCATOR_HALF / 2**z
    min_x = -MERCATOR_HALF + x * size
    max_y = MERCATOR_HALF - y * size
    return min_x, max_y - size, min_x + size, max_y


def tile_bounds_lonlat(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) in degrees."""
    min_x, min_y, max_x, max_y = tile_bounds_mercator(z, x, y)
    west, south = mercator_to_lonlat(min_x, min_y)
    east, north = mercator_to_lonlat(max_x, max_y)
    return float(west), float(south), float(east), float(north)


def ground_resolution_m(lat: float, z: int, tile_size: int = 256) -> float:
    """Meters per pixel on the ground for a tile pixel at latitude ``lat``."""
    return math.cos(math.radians(lat)) * 2 * MERCATOR_HALF / (tile_size * 2**z)
