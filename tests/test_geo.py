import numpy as np
import pytest

from earthling.core import geo


def test_ecef_known_point():
    # Equator / prime meridian
    assert geo.geodetic_to_ecef(0, 0, 0) == pytest.approx([geo.WGS84_A, 0, 0])
    # North pole
    assert geo.geodetic_to_ecef(90, 0, 0) == pytest.approx([0, 0, geo.WGS84_B], abs=1e-6)


def test_ecef_roundtrip():
    lat = np.array([45.9237, -33.9, 89.0, 0.0])
    lon = np.array([6.8694, 151.2, -120.0, 180.0])
    h = np.array([1035.0, -20.0, 3000.0, 8848.0])
    lat2, lon2, h2 = geo.ecef_to_geodetic(geo.geodetic_to_ecef(lat, lon, h))
    assert lat2 == pytest.approx(lat, abs=1e-9)
    assert np.cos(np.radians(lon2 - lon)) == pytest.approx(1.0)
    assert h2 == pytest.approx(h, abs=1e-3)


def test_enu_frame_axes():
    frame = geo.LocalFrame(46.0, 7.0, 0.0)
    assert frame.geodetic_to_enu(46.0, 7.0, 0.0) == pytest.approx([0, 0, 0], abs=1e-6)
    up = frame.geodetic_to_enu(46.0, 7.0, 100.0)
    assert up == pytest.approx([0, 0, 100], abs=1e-6)
    north = frame.geodetic_to_enu(46.01, 7.0, 0.0)
    assert north[1] == pytest.approx(1111.9, rel=1e-2) and abs(north[0]) < 1e-6
    east = frame.geodetic_to_enu(46.0, 7.01, 0.0)
    assert east[0] == pytest.approx(772.9, rel=1e-2)


def test_enu_curvature_drop():
    frame = geo.LocalFrame(46.0, 7.0, 0.0)
    far = frame.geodetic_to_enu(46.0 + 200_000 / 111_200, 7.0, 0.0)
    # d^2 / 2R ~ 3.1 km at 200 km
    assert far[2] == pytest.approx(-3140, rel=0.02)


def test_enu_roundtrip():
    frame = geo.LocalFrame(45.5, 6.5, 500.0)
    lat, lon, h = frame.enu_to_geodetic(frame.geodetic_to_enu(46.1, 7.2, 2500.0))
    assert (float(lat), float(lon), float(h)) == pytest.approx((46.1, 7.2, 2500.0), abs=1e-6)


def test_tiles():
    tx, ty = geo.lonlat_to_tile(0.0, 0.0, 1)
    assert (float(tx), float(ty)) == pytest.approx((1.0, 1.0))
    # Chamonix at z12 (slippy map formula: x=2126.0.., y=1458.3..)
    tx, ty = geo.lonlat_to_tile(6.8694, 45.9237, 12)
    assert (int(tx), int(ty)) == (2126, 1458)
    west, south, east, north = geo.tile_bounds_lonlat(12, 2126, 1458)
    assert west <= 6.8694 <= east and south <= 45.9237 <= north
    lon, lat = geo.tile_to_lonlat(tx, ty, 12)
    assert (float(lon), float(lat)) == pytest.approx((6.8694, 45.9237))


def test_mercator_roundtrip_and_resolution():
    x, y = geo.lonlat_to_mercator(6.8694, 45.9237)
    lon, lat = geo.mercator_to_lonlat(x, y)
    assert (float(lon), float(lat)) == pytest.approx((6.8694, 45.9237))
    assert geo.ground_resolution_m(0.0, 0) == pytest.approx(156543.03, rel=1e-6)
