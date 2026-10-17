"""The POI library: generic kinds of points of interest to place on a map (a peak, a town, a
water point, a "10 km" badge ...) and the user's own images. Qt independent.

A kind gives a new POI its icon, name and caption; everything else is the usual POI.
Icons: ``builtin:<name>`` (src/earthling/assets/icons), ``badge`` (the caption drawn inside a
pill), ``none`` (the caption alone) or an image file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PoiKind:
    category: str
    name: str  # also the name of a new POI
    icon: str
    caption: str = ""
    size_px: float = 64.0


def _kinds(category: str, *items: tuple) -> list[PoiKind]:
    return [PoiKind(category, *item) for item in items]


KINDS: list[PoiKind] = [
    *_kinds("Route",
            ("Pin", "builtin:pin"), ("Start", "builtin:flag"), ("Finish", "builtin:finish"),
            ("Turnaround", "builtin:turnaround"), ("Signpost", "builtin:summit"),
            ("Warning", "builtin:warning"), ("Text only", "none", "Text")),
    *_kinds("Distance",
            *((f"{km} km", "badge", f"{km} km", 48.0) for km in range(10, 101, 10)),
            ("Badge", "badge", "Text", 48.0)),
    *_kinds("Nature",
            ("Peak", "builtin:peak"), ("Viewpoint", "builtin:viewpoint"),
            ("Pass", "builtin:pass"), ("Waterfall", "builtin:waterfall"),
            ("Lake / swimming", "builtin:swimming"), ("Beach", "builtin:beach"),
            ("Photo spot", "builtin:camera")),
    *_kinds("Places",
            ("Town", "builtin:town"), ("Highlight", "builtin:attraction"),
            ("Church", "builtin:church"), ("Castle / ruin", "builtin:castle"),
            ("Bridge", "builtin:bridge"), ("Information", "builtin:info")),
    *_kinds("Sleep",
            ("Tent", "builtin:tent"), ("Mountain hut", "builtin:hut"), ("Hotel", "builtin:hotel")),
    *_kinds("Food & supplies",
            ("Coffee", "builtin:coffee"), ("Restaurant", "builtin:restaurant"),
            ("Meal", "builtin:couscous"), ("Picnic", "builtin:picnic"),
            ("Supermarket", "builtin:supermarket"), ("Drinking water", "builtin:water"),
            ("Aid station", "builtin:aid"), ("Party", "builtin:beerpong")),
    *_kinds("Services",
            ("Toilet", "builtin:toilet"), ("First aid", "builtin:firstaid"),
            ("Charging", "builtin:power"), ("Parking", "builtin:parking"),
            ("Bike repair", "builtin:bikerepair")),
    *_kinds("Weather",
            ("Bad weather", "builtin:storm"), ("Cold", "builtin:cold")),
    *_kinds("Transport",
            ("Airport", "builtin:airport"), ("Train station", "builtin:train"),
            ("Bus stop", "builtin:bus"), ("Ferry / pier", "builtin:ferry"),
            ("Taxi", "builtin:taxi"), ("Cable car", "builtin:cablecar")),
]  # fmt: skip

MY_IMAGES = "My images"


def load_user_images(path: Path) -> list[PoiKind]:
    """The user's own icon images (a JSON list of file paths), missing files left out."""
    try:
        files = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [PoiKind(MY_IMAGES, Path(f).stem, str(f)) for f in files if Path(f).exists()]


def add_user_image(path: Path, image: Path) -> PoiKind:
    """Remember ``image`` in the user's library file ``path``; returns its kind."""
    image = Path(image).resolve()
    files = [k.icon for k in load_user_images(path)]
    if str(image) not in files:
        files.append(str(image))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(files, indent=1), encoding="utf-8")
    return PoiKind(MY_IMAGES, image.stem, str(image))
