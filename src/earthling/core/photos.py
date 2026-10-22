"""Photo overlays: pictures placed on the screen (Qt independent).

A photo belongs to the scene (saved in the scene file). Its static data (the image file) lives
in :class:`Photo`; placement and look are scene properties ``photo.<id>.*`` registered per
photo (see :func:`photo_properties`), so a photo fades in and out with keys on its opacity like
every other property. Positions and sizes are fractions of the frame: the layout is the same
at every output resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from earthling.core.properties import enum, flt, section, shown_unless

APPEAR = [("keys", "By the opacity keys"), ("reached", "When the line reaches where it was taken")]
FADE_S = 0.6  # fade in / out of photos shown when the line reaches them


@dataclass
class Photo:
    id: str  # unique slug, used in property ids
    name: str
    path: str  # image file (absolute or relative to the scene); EXIF orientation is applied
    lat: float | None = None  # where it was taken (from EXIF, a track time or set by hand)
    lon: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Photo:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def photo_properties(photo: Photo) -> list:
    """Keyframeable properties of one photo ("photo.<id>.*")."""
    p = f"photo.{photo.id}."
    return section(
        f"Photo: {photo.name}",
        enum(p + "appear", "Appear", "keys", APPEAR, animatable=False,
             tooltip="Fade in when the drawn line reaches the place of the photo, stay for the "
                     "time below and fade out (instead of keys on the opacity)"),
        flt(p + "hold", "Shown for", 4.0, 0.5, 60.0, step=0.5, decimals=1, unit="s",
            animatable=False, when=shown_unless(p + "appear", "keys"),
            tooltip="With 'When the line reaches…': seconds on screen"),
        flt(p + "opacity", "Opacity", 0.0, 0.0, 1.0,
            tooltip="Keys on the opacity fade the photo in and out (0: not drawn)"),
        flt(p + "x", "Centre x", 0.8, -0.5, 1.5, step=0.005, decimals=3,
            tooltip="Horizontal centre, fraction of the frame width (0 left, 1 right)"),
        flt(p + "y", "Centre y", 0.25, -0.5, 1.5, step=0.005, decimals=3,
            tooltip="Vertical centre, fraction of the frame height (0 top, 1 bottom)"),
        flt(p + "height", "Height", 0.3, 0.02, 1.0, step=0.005, decimals=3,
            tooltip="Height of the picture, fraction of the frame height (width from the image)"),
        flt(p + "scale", "Scale", 1.0, 0.0, 3.0, step=0.01, advanced=True,
            tooltip="Extra size factor (e.g. a grow-in together with the fade)"),
        flt(p + "rotation", "Rotation", 0.0, -45.0, 45.0, step=0.5, decimals=1, unit="°",
            advanced=True),
        flt(p + "border", "Border", 8.0, 0.0, 60.0, step=0.5, decimals=1, unit="px",
            tooltip="White frame around the picture (px at 1080p)"),
        flt(p + "shadow", "Shadow", 0.5, 0.0, 1.0, advanced=True,
            tooltip="Soft drop shadow below the picture"),
        flt(p + "zoom", "Zoom", 1.0, 1.0, 3.0, step=0.01,
            tooltip="Zoom into the picture inside its frame (a slow push-in with keys)"),
    )  # fmt: skip


def all_photo_properties(photos: list[Photo]) -> list:
    return [d for photo in photos for d in photo_properties(photo)]


def reached_alpha(since: float, hold: float) -> float:
    """Opacity of a photo ``since`` seconds after the line reached it: a fade in, ``hold``
    seconds on screen, a fade out."""
    from earthling.core.easing import smooth01

    if since < 0.0:
        return 0.0
    return smooth01(since / FADE_S) * (1.0 - smooth01((since - FADE_S - hold) / FADE_S))


def read_exif(path: Path) -> tuple[float | None, float | None, datetime | None]:
    """(lat, lon, time taken) from a photo's EXIF data (None where missing). The time is
    aware when the file stores its UTC offset, else naive (the camera's clock)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except OSError:
        return None, None, None
    lat = lon = None
    gps = exif.get_ifd(0x8825)
    try:

        def degrees(value, ref) -> float:
            d, m, s = (float(v) for v in value)
            return (-1.0 if ref in ("S", "W") else 1.0) * (d + m / 60.0 + s / 3600.0)

        if gps.get(2) and gps.get(4):
            lat, lon = degrees(gps[2], gps.get(1, "N")), degrees(gps[4], gps.get(3, "E"))
            if lat == 0.0 and lon == 0.0:  # (cameras without a fix write 0, 0)
                lat = lon = None
    except (TypeError, ValueError, ZeroDivisionError):
        lat = lon = None
    taken = None
    info = exif.get_ifd(0x8769)
    text = info.get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
    if text:
        try:
            taken = datetime.strptime(str(text).strip(), "%Y:%m:%d %H:%M:%S")
            offset = info.get(36881)  # OffsetTimeOriginal, e.g. "+02:00"
            if offset:
                sign = -1 if str(offset).startswith("-") else 1
                h, m = (int(v) for v in str(offset).strip("+-").split(":"))
                taken = taken.replace(tzinfo=timezone(sign * timedelta(hours=h, minutes=m)))
        except ValueError:
            taken = None
    return lat, lon, taken
