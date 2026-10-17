"""Photo overlays: pictures placed on the screen (Qt independent).

A photo belongs to the scene (saved in the scene file). Its static data (the image file) lives
in :class:`Photo`; placement and look are scene properties ``photo.<id>.*`` registered per
photo (see :func:`photo_properties`), so a photo fades in and out with keys on its opacity like
every other property. Positions and sizes are fractions of the frame: the layout is the same
at every output resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class Photo:
    id: str  # unique slug, used in property ids
    name: str
    path: str  # image file (absolute or relative to the scene); EXIF orientation is applied

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Photo:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def photo_properties(photo: Photo) -> list:
    """Keyframeable properties of one photo ("photo.<id>.*")."""
    from earthling.render.parameters import flt, section

    p = f"photo.{photo.id}."
    return section(
        f"Photo: {photo.name}",
        flt(p + "opacity", "Opacity", 0.0, 0.0, 1.0,
            tooltip="Keys on the opacity fade the photo in and out (0: not drawn)"),
        flt(p + "x", "Centre x", 0.8, -0.5, 1.5, step=0.005, decimals=3,
            tooltip="Horizontal centre, fraction of the frame width (0 left, 1 right)"),
        flt(p + "y", "Centre y", 0.25, -0.5, 1.5, step=0.005, decimals=3,
            tooltip="Vertical centre, fraction of the frame height (0 top, 1 bottom)"),
        flt(p + "height", "Height", 0.3, 0.02, 1.0, step=0.005, decimals=3,
            tooltip="Height of the picture, fraction of the frame height (width from the image)"),
        flt(p + "scale", "Scale", 1.0, 0.0, 3.0, step=0.01,
            tooltip="Extra size factor (e.g. a grow-in together with the fade)"),
        flt(p + "rotation", "Rotation", 0.0, -45.0, 45.0, step=0.5, decimals=1, unit="°"),
        flt(p + "border", "Border", 8.0, 0.0, 60.0, step=0.5, decimals=1, unit="px",
            tooltip="White frame around the picture (px at 1080p)"),
        flt(p + "shadow", "Shadow", 0.5, 0.0, 1.0,
            tooltip="Soft drop shadow below the picture"),
        flt(p + "zoom", "Zoom", 1.0, 1.0, 3.0, step=0.01,
            tooltip="Zoom into the picture inside its frame (a slow push-in with keys)"),
    )  # fmt: skip


def all_photo_properties(photos: list[Photo]) -> list:
    return [d for photo in photos for d in photo_properties(photo)]
