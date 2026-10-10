"""Time zone of a position (for the local time of journeys across several time zones)."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo


@lru_cache(maxsize=4096)
def _tz_name(lon: float, lat: float) -> str:
    try:
        import tzfpy
    except ImportError:  # optional: without it every position uses UTC offsets of 0
        return "UTC"
    return tzfpy.get_tz(lon, lat) or "UTC"


def timezone_at(lon: float, lat: float) -> ZoneInfo:
    """The IANA time zone at lon/lat (oceans: the nautical "Etc/GMT±n" zones)."""
    name = _tz_name(round(float(lon), 2), round(float(lat), 2))
    try:
        return ZoneInfo(name)
    except Exception:  # an unknown name in an old tz database
        return ZoneInfo("UTC")


def utc_offset_text(t: datetime) -> str:
    """Offset text like "UTC+7", "UTC-3", "UTC+5:45" for an aware datetime."""
    offset = t.utcoffset()
    minutes = int(round(offset.total_seconds() / 60.0)) if offset is not None else 0
    sign = "+" if minutes >= 0 else "-"
    h, m = divmod(abs(minutes), 60)
    return f"UTC{sign}{h}" + (f":{m:02d}" if m else "")
