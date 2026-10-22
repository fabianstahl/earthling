"""Looks: one click sets a whole category (time of day, weather) to a good starting point.

Each look is a set of property values, some computed for the project: the sun times from
the scene's date and place, fog and snow heights from the elevations of the tracks. A look
is one undo step; animated values are keyed at the playhead (auto-key).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from earthling.core.sun import local_to_aware, solar_position


@dataclass
class Look:
    name: str
    icon: str
    tooltip: str
    values: Callable[[], dict]  # property id -> value
    apply: Callable[[], None] = lambda: None


def sun_time(day: datetime, lat: float, lon: float, timezone: str, elevation: float | None,
             evening: bool = True) -> datetime:  # fmt: skip
    """The local time on ``day`` when the sun is at ``elevation`` degrees (None: solar noon),
    in the evening or the morning; the nearest minute (or the noon / midnight extreme when
    the sun never gets there)."""
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = [start + timedelta(minutes=m) for m in range(0, 24 * 60, 2)]
    heights = [solar_position(local_to_aware(t, timezone), lat, lon).elevation for t in minutes]
    noon = max(range(len(heights)), key=heights.__getitem__)
    if elevation is None:
        return minutes[noon]
    side = range(noon, len(minutes)) if evening else range(noon, -1, -1)
    best = min(side, key=lambda i: abs(heights[i] - elevation))
    return minutes[best]


def make_looks(window) -> dict[str, list[Look]]:
    """The looks per inspector category ("light", "weather")."""

    def place() -> tuple[float, float, str]:
        session = window.session
        if session is None:
            return 46.0, 7.0, "UTC"
        return session.frame.lat, session.frame.lon, session.config.project.timezone

    def ground() -> float:
        """The lowest elevation of the tracks (fog and snow lines start from it)."""
        heights = [t.stats.min_ele_m for t in window.tracks if t.stats.min_ele_m is not None]
        return min(heights) if heights else 800.0

    def at_sun(elevation: float | None, extra: dict) -> Callable[[], dict]:
        def values() -> dict:
            lat, lon, tz = place()
            day = window.scene.store["sun.datetime"]
            when = sun_time(day, lat, lon, tz, elevation)
            return {"sun.follow_track": False, "sun.datetime": when,
                    "sun.elevation_offset": 0.0, **extra}  # fmt: skip

        return values

    clear = {"weather.enabled": False, "fog.enabled": False, "rain.enabled": False,
             "lightning.rate": 0.0, "snow.cover": 0.0}  # fmt: skip

    def weather(build: Callable[[float], dict]) -> Callable[[], dict]:
        return lambda: {**clear, "sky.haze": 1.0, "haze.aerial": 1.0, **build(ground())}

    looks = {
        "light": [
            Look("Noon", "sun", "The sun at its highest on the scene's day",
                 at_sun(None, {"post.exposure": 0.0, "night.lights": 0.0})),
            Look("Golden hour", "sun-horizon", "Low, warm evening sun (6° above the horizon)",
                 at_sun(6.0, {"post.exposure": 0.3, "night.lights": 0.0})),
            Look("Blue hour", "cloud-fog", "Just after sunset (4° below the horizon)",
                 at_sun(-4.0, {"post.exposure": 1.0, "night.lights": 1.0})),
            Look("Night", "moon-stars", "Dark night with stars and city lights",
                 at_sun(-20.0, {"post.exposure": 1.5, "night.lights": 2.0, "sky.stars": 1.5})),
        ],
        "weather": [
            Look("Clear", "sun", "No clouds, fog, rain or snow", weather(lambda g: {})),
            Look("Hazy", "sun-dim", "Summer haze in the distance",
                 weather(lambda g: {"sky.haze": 3.0, "haze.aerial": 2.0})),
            Look("Morning fog", "cloud-fog", "Fog in the valleys below the tracks",
                 weather(lambda g: {
                     "weather.enabled": True, "weather.layer1.enabled": True,
                     "weather.layer1.base": g - 150.0, "weather.layer1.thickness": 350.0,
                     "weather.layer1.coverage": 0.85, "weather.layer2.enabled": False,
                     "weather.layer3.enabled": False, "fog.enabled": True,
                     "fog.base": g + 100.0, "fog.density": 0.6, "sky.haze": 1.5})),
            Look("Storm", "cloud-lightning", "Dark clouds, rain and lightning",
                 weather(lambda g: {
                     "weather.enabled": True, "weather.layer2.enabled": True,
                     "weather.layer2.coverage": 0.9, "rain.enabled": True,
                     "rain.type": "rain", "rain.intensity": 0.8, "lightning.rate": 6.0,
                     "sky.haze": 1.6})),
            Look("Snowfall", "snowflake", "Falling snow and fresh snow on the ground",
                 weather(lambda g: {
                     "weather.enabled": True, "weather.layer2.enabled": True,
                     "weather.layer2.coverage": 0.8, "rain.enabled": True, "rain.type": "snow",
                     "rain.intensity": 0.6, "snow.cover": 0.8, "snow.line": g - 200.0})),
        ],
    }  # fmt: skip
    for items in looks.values():
        for look in items:
            look.apply = lambda lk=look: apply_look(window, lk)
    return looks


def apply_look(window, look: Look) -> None:
    """Set the look's values as one undo step (animated values are keyed at the playhead)."""
    registry = window.scene.store.registry

    def run() -> None:
        for pid, value in look.values().items():
            if pid in registry:
                window._on_panel_edit(pid, registry[pid].coerce(value), False)

    window.undo_batch(f"Look: {look.name}", run)
