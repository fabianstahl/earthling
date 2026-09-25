"""Solar position (NOAA Solar Calculator algorithm, accurate to ~0.01° for 1950-2050)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SunPosition:
    azimuth: float  # degrees clockwise from north
    elevation: float  # degrees above the horizon, refraction corrected

    def direction_enu(self) -> tuple[float, float, float]:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        return (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))


def _julian_day(t: datetime) -> float:
    t = t.astimezone(UTC)
    return t.timestamp() / 86400.0 + 2440587.5


def _refraction(elevation: float) -> float:
    """Atmospheric refraction correction in degrees (NOAA approximation)."""
    if elevation > 85.0:
        return 0.0
    te = math.tan(math.radians(elevation))
    if elevation > 5.0:
        corr = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
    elif elevation > -0.575:
        corr = 1735.0 + elevation * (
            -518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711))
        )
    else:
        corr = -20.772 / te
    return corr / 3600.0


def solar_position(when: datetime, lat: float, lon: float) -> SunPosition:
    """Sun position for an aware datetime (naive datetimes are treated as UTC)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    jd = _julian_day(when)
    jc = (jd - 2451545.0) / 36525.0

    geom_mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccent = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    m = math.radians(geom_mean_anom)
    eq_ctr = (
        math.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * m) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * m) * 0.000289
    )
    true_long = geom_mean_long + eq_ctr
    omega = math.radians(125.04 - 1934.136 * jc)
    app_long = true_long - 0.00569 - 0.00478 * math.sin(omega)
    mean_obliq = (
        23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    )
    obliq = math.radians(mean_obliq + 0.00256 * math.cos(omega))
    declination = math.asin(math.sin(obliq) * math.sin(math.radians(app_long)))

    var_y = math.tan(obliq / 2.0) ** 2
    l0 = math.radians(geom_mean_long)
    eq_time = 4.0 * math.degrees(
        var_y * math.sin(2 * l0)
        - 2 * eccent * math.sin(m)
        + 4 * eccent * var_y * math.sin(m) * math.cos(2 * l0)
        - 0.5 * var_y**2 * math.sin(4 * l0)
        - 1.25 * eccent**2 * math.sin(2 * m)
    )  # minutes

    utc = when.astimezone(UTC)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60.0 + utc.microsecond / 6e7
    true_solar_time = (minutes + eq_time + 4.0 * lon) % 1440.0
    hour_angle = (
        true_solar_time / 4.0 - 180.0 if true_solar_time >= 0 else true_solar_time / 4.0 + 180.0
    )

    phi = math.radians(lat)
    ha = math.radians(hour_angle)
    cos_zenith = math.sin(phi) * math.sin(declination) + math.cos(phi) * math.cos(
        declination
    ) * math.cos(ha)
    zenith = math.acos(max(-1.0, min(1.0, cos_zenith)))
    elevation = 90.0 - math.degrees(zenith)

    denom = math.cos(phi) * math.sin(zenith)
    if abs(denom) < 1e-12:
        azimuth = 180.0 if lat > 0 else 0.0
    else:
        cos_az = (math.sin(phi) * math.cos(zenith) - math.sin(declination)) / denom
        az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        azimuth = (az + 180.0) % 360.0 if hour_angle > 0 else (540.0 - az) % 360.0
    return SunPosition(azimuth, elevation + _refraction(elevation))


def local_to_aware(local: datetime, timezone: str) -> datetime:
    return local.replace(tzinfo=ZoneInfo(timezone))
