"""Tile data providers.

A provider describes where tiles come from, their format, zoom range and license. Imagery
providers serve XYZ tiles directly. DEM providers (see :mod:`earthling.data.dem`) download
source rasters and bake them into the same XYZ scheme.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

USER_AGENT = "earthling/0.1 (personal, non-commercial flyover renderer)"


@dataclass(frozen=True)
class License:
    name: str
    attribution: str
    url: str = ""
    commercial_use: bool = False


@dataclass(frozen=True)
class TileProvider:
    id: str
    name: str
    kind: str  # "imagery" or "dem"
    url_template: str = ""  # {z} {x} {y}
    ext: str = "jpg"
    min_zoom: int = 0
    max_zoom: int = 19
    tile_size: int = 256
    license: License = License("unknown", "")
    max_requests_per_second: float = 20.0
    concurrency: int = 8
    headers: dict[str, str] = field(default_factory=dict)
    # MD5 hashes of "no data" placeholder images that must be treated as missing tiles.
    placeholder_md5: frozenset[str] = frozenset()
    # Regional providers: coverage name (see earthling.data.coverage), None = worldwide.
    coverage: str | None = None
    feather_m: float = 300.0  # blend width at the coverage edge

    @property
    def coverage_area(self):
        from earthling.data.coverage import get_coverage

        return get_coverage(self.coverage, self.feather_m)

    def tile_url(self, z: int, x: int, y: int) -> str:
        return self.url_template.format(z=z, x=x, y=y)

    def is_placeholder(self, data: bytes) -> bool:
        return bool(self.placeholder_md5) and hashlib.md5(data).hexdigest() in self.placeholder_md5


ESRI_WORLD_IMAGERY = TileProvider(
    id="esri_world_imagery",
    name="Esri World Imagery",
    kind="imagery",
    url_template=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    ext="jpg",
    max_zoom=19,
    license=License(
        "Esri Master License Agreement (non-commercial use with attribution)",
        "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "https://www.esri.com/en-us/legal/terms/full-master-agreement",
    ),
    max_requests_per_second=15.0,
    placeholder_md5=frozenset({"f27d9de7f80c13501f470595e327aa6d"}),
)

EOX_S2CLOUDLESS = TileProvider(
    id="eox_s2cloudless_2024",
    name="EOX Sentinel-2 cloudless 2024",
    kind="imagery",
    url_template=(
        "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/"
        "GoogleMapsCompatible/{z}/{y}/{x}.jpg"
    ),
    ext="jpg",
    max_zoom=15,
    license=License(
        "CC BY-NC-SA 4.0",
        "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH "
        "(Contains modified Copernicus Sentinel data 2024)",
        "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    ),
    max_requests_per_second=10.0,
)

OPENTOPOMAP = TileProvider(
    id="opentopomap",
    name="OpenTopoMap",
    kind="topo",
    url_template="https://tile.opentopomap.org/{z}/{x}/{y}.png",
    ext="png",
    max_zoom=17,
    license=License(
        "CC BY-SA 3.0 (map style), ODbL (OpenStreetMap data)",
        "Map data: (c) OpenStreetMap contributors, SRTM | Map style: (c) OpenTopoMap (CC-BY-SA)",
        "https://opentopomap.org/about",
    ),
    # OpenTopoMap is run by volunteers: keep the load low
    max_requests_per_second=2.0,
    concurrency=2,
)

# --- France: IGN Géoplateforme --------------------------------------------------------------
IGN_BDORTHO = TileProvider(
    id="ign_bdortho",
    name="IGN BD ORTHO (France)",
    kind="imagery",
    url_template=(
        "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
        "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&TILEMATRIXSET=PM"
        "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"
    ),
    ext="jpg",
    max_zoom=19,  # 20 cm orthophotos
    license=License(
        "Licence Ouverte / Open Licence 2.0 (Etalab)",
        "Orthophotos: © IGN – BD ORTHO®",
        "https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
        commercial_use=True,
    ),
    max_requests_per_second=10.0,
    # all-white tile served inside the service area where there is no imagery
    placeholder_md5=frozenset({"cb33c7debb71738617c005e666ba5db4"}),
    coverage="france",
)

PROVIDERS: dict[str, TileProvider] = {
    p.id: p for p in (ESRI_WORLD_IMAGERY, EOX_S2CLOUDLESS, OPENTOPOMAP, IGN_BDORTHO)
}


def register(provider: TileProvider) -> None:
    PROVIDERS[provider.id] = provider


def get_provider(provider_id: str) -> TileProvider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise KeyError(f"unknown provider '{provider_id}' (known: {known})") from None
