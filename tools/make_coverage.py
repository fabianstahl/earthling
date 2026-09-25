"""Generate the provider coverage polygons in src/earthling/data/coverage/.

Coverages come from Natural Earth 1:10m country polygons, simplified and shrunk a little so
that the feathered blend at the edge always lies within the provider's real data.

    python tools/make_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import shapely
from pyproj import Transformer

URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_10m_admin_0_countries.geojson"
)
OUT = Path(__file__).resolve().parents[1] / "src" / "earthling" / "data" / "coverage"
SHRINK_M = 250.0  # stay clear of the border (Natural Earth accuracy, data edges)
SIMPLIFY_M = 150.0

# coverage name -> (ADM0_A3 codes, optional lon/lat clip box)
REGIONS = {
    "france": (["FRA"], (-5.5, 41.0, 10.0, 51.5)),  # metropolitan France incl. Corsica
    "switzerland": (["CHE", "LIE"], None),  # swisstopo covers Liechtenstein too
    "austria": (["AUT"], None),
}


def load_countries(cache: Path) -> dict:
    if not cache.exists():
        print(f"downloading {URL}")
        cache.write_bytes(httpx.get(URL, timeout=120, follow_redirects=True).content)
    return json.loads(cache.read_text(encoding="utf-8"))


def make(name: str, codes: list[str], clip, features: list[dict]) -> shapely.Geometry:
    parts = [
        shapely.from_geojson(json.dumps(f["geometry"]))
        for f in features
        if f["properties"].get("ADM0_A3") in codes
    ]
    if not parts:
        raise SystemExit(f"{name}: no country {codes}")
    geom = shapely.union_all(parts)
    if clip is not None:
        geom = geom.intersection(shapely.box(*clip))
    # metric processing in ETRS89 / LAEA Europe
    to_laea = Transformer.from_crs(4326, 3035, always_xy=True)
    to_ll = Transformer.from_crs(3035, 4326, always_xy=True)

    def fwd(c):
        return np.column_stack(to_laea.transform(c[:, 0], c[:, 1]))

    def inv(c):
        return np.column_stack(to_ll.transform(c[:, 0], c[:, 1]))

    metric = shapely.transform(geom, fwd)
    metric = metric.buffer(-SHRINK_M).simplify(SIMPLIFY_M, preserve_topology=True)
    metric = shapely.make_valid(metric)
    return shapely.set_precision(shapely.transform(metric, inv), 1e-5)


def main() -> int:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "ne_10m_countries.geojson"
    features = load_countries(cache)["features"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (codes, clip) in REGIONS.items():
        geom = make(name, codes, clip, features)
        path = OUT / f"{name}.geojson"
        path.write_text(shapely.to_geojson(geom), encoding="utf-8")
        n = shapely.get_num_coordinates(geom)
        print(
            f"{path.name}: {n} vertices, {path.stat().st_size / 1024:.0f} KB, bounds {geom.bounds}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
