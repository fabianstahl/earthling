from datetime import UTC, datetime, timedelta

import pytest

from earthling.core.sun import local_to_aware, solar_position

LAT, LON = 45.92, 6.87  # Chamonix


def solar_noon(day: datetime):
    """Search the time of maximum elevation (1-minute resolution)."""
    best = max(
        (day + timedelta(minutes=m) for m in range(9 * 60, 15 * 60)),
        key=lambda t: solar_position(t, LAT, LON).elevation,
    )
    return best, solar_position(best, LAT, LON)


def test_solstice_noon_elevation_and_azimuth():
    t, pos = solar_noon(datetime(2026, 6, 21, tzinfo=UTC))
    # 90 - lat + obliquity (23.44) + tiny refraction
    assert pos.elevation == pytest.approx(90 - LAT + 23.44, abs=0.1)
    assert pos.azimuth == pytest.approx(180.0, abs=0.5)
    # solar noon in UTC: 12:00 - lon/15 h - equation of time (~ -1.8 min in late June)
    assert t.hour == 11 and 30 <= t.minute <= 36


def test_equinox_and_winter():
    _, pos = solar_noon(datetime(2026, 3, 20, tzinfo=UTC))
    assert pos.elevation == pytest.approx(90 - LAT, abs=0.5)
    _, pos = solar_noon(datetime(2026, 12, 21, tzinfo=UTC))
    assert pos.elevation == pytest.approx(90 - LAT - 23.44, abs=0.15)


def test_morning_east_evening_west_night_negative():
    tz = "Europe/Paris"
    morning = solar_position(local_to_aware(datetime(2026, 7, 1, 7, 0), tz), LAT, LON)
    evening = solar_position(local_to_aware(datetime(2026, 7, 1, 20, 0), tz), LAT, LON)
    night = solar_position(local_to_aware(datetime(2026, 7, 1, 1, 0), tz), LAT, LON)
    assert 60 < morning.azimuth < 100 and morning.elevation > 0
    assert 270 < evening.azimuth < 310 and evening.elevation > 0
    assert night.elevation < -10


def test_sunset_time_and_direction_vector():
    # Chamonix sunset on July 1st (flat horizon) is around 21:25 CEST
    tz = "Europe/Paris"
    before = solar_position(local_to_aware(datetime(2026, 7, 1, 21, 10), tz), LAT, LON)
    after = solar_position(local_to_aware(datetime(2026, 7, 1, 21, 40), tz), LAT, LON)
    assert before.elevation > 0 > after.elevation
    x, y, z = before.direction_enu()
    assert x < 0 and z > 0 and abs(x * x + y * y + z * z - 1) < 1e-9


def test_compute_lighting_day_and_night():
    from earthling.core.scene import Scene
    from earthling.render.lighting import compute_lighting

    scene = Scene()
    scene.store.set("sun.datetime", datetime(2026, 7, 1, 13, 30))
    day = compute_lighting(scene.store, LAT, LON, "Europe/Paris")
    assert day.sun.elevation > 60 and max(day.sun_radiance) > 1.0
    scene.store.set("sun.datetime", datetime(2026, 7, 1, 23, 30))
    night = compute_lighting(scene.store, LAT, LON, "Europe/Paris")
    assert night.sun_radiance == (0.0, 0.0, 0.0)
    assert max(night.sky_ambient) < 0.1 * max(day.sky_ambient)
    scene.store.set("sun.elevation_offset", 45.0)
    scene.store.set("sun.datetime", datetime(2026, 7, 1, 21, 0))
    lifted = compute_lighting(scene.store, LAT, LON, "Europe/Paris")
    assert lifted.sun.elevation > 40
