# Regional data sources

Earthling works everywhere with its worldwide sources (Esri World Imagery, Copernicus DEM
GLO-30). National mapping agencies often publish much better data for their country: 10–20 cm
orthophotos and 0.5–2 m terrain models. This guide explains how such sources are combined,
which ones are built in, and how to add a new one.

## How sources are combined

`[sources]` in `earthling.toml` lists imagery and elevation sources in **priority order**:

```toml
[sources]
imagery = ["ign_bdortho", "swisstopo_swissimage", "esri_world_imagery"]
dem     = ["ign_rgealti", "swisstopo_alti3d", "copernicus_glo30"]
```

- A **regional** source has a *coverage*: a polygon (lon/lat) of where its data is valid.
  Worldwide sources have none.
- For every terrain tile, the highest-priority source covering it wins. At a coverage edge,
  the next source shows through with a soft transition. The source's weight goes from 0 at the
  edge to 1 at `feather_m` (default 300 m) inside. The weights come from world-space distances,
  so they match across all levels of detail.
- Missing data falls back to the next source: imagery tiles that don't exist or are
  placeholders, and holes in a DEM.
- Elevation blends per heightmap sample, because every source is baked onto the same grid.
- `earthling plan <project>` lists the tiles and download size per source. Downloads only
  fetch the tiles each source is actually needed for. A tile fully inside a higher-priority
  coverage is never requested from the sources below it.

Seasons and colours differ between sources: an orthophoto taken in spring shows more snow
than a summer mosaic next to it. Blending hides the seam, not the difference in content.

## Built-in regional sources

| id | kind | region | data |
| --- | --- | --- | --- |
| `ign_bdortho` | imagery | France | BD ORTHO 20 cm, Géoplateforme WMTS, up to z19 |
| `ign_rgealti` | DEM | France | RGE ALTI 1–5 m, Géoplateforme WMS (float32), fetched per tile |
| `swisstopo_swissimage` | imagery | Switzerland | SWISSIMAGE 10 cm, WMTS (EPSG:3857), up to z19 |
| `swisstopo_alti3d` | DEM | Switzerland | swissALTI3D 2 m, 1 km sheets from the STAC API (LV95, reprojected) |

Licences: IGN data is under the Licence Ouverte 2.0 (Etalab). swisstopo data falls under the
terms of use for free geodata. Both require attribution, and every source carries its
attribution text (`earthling check-source` prints it).

The regional DEMs are *terrain* models (bare ground), while Copernicus is a *surface* model
(it includes trees and buildings). The difference, typically 5–20 m in forests, is hidden by
the feathered transition. Use a higher `dem_zoom` (14–15) in the inner zones to benefit from
the detail.

## Adding a source without code

Projects can define their own sources in `earthling.toml` and then use their ids in
`[sources]`. A project source with the id of a built-in one replaces it.

### Imagery (XYZ / WMTS tiles in Web Mercator)

Tested example: Austrian orthophotos from basemap.at (note the `{z}/{y}/{x}` order):

```toml
[sources]
imagery = ["basemap_at_ortho", "esri_world_imagery"]

[[providers]]
id          = "basemap_at_ortho"
kind        = "imagery"            # or "topo" for map layers
name        = "basemap.at Orthofoto (Austria)"
url         = "https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg"
ext         = "jpg"
max_zoom    = 19
coverage    = "austria"            # shipped name or a GeoJSON file (relative to the project)
attribution = "Orthofoto: basemap.at"
license     = "CC BY 4.0"
license_url = "https://basemap.at"
requests_per_second = 5
```

WMTS services work if they offer a Web Mercator tile matrix set ("GoogleMapsCompatible",
"PM", "3857", …). Write the tile URL with `{z}`, `{x}` (column) and `{y}` (row) filled into the
right places, as in the IGN example in `src/earthling/data/providers.py`.

A service may return a uniform image instead of an error where it has no data, for example
IGN's white tile. Put its MD5 hash into `placeholder_md5 = ["…"]`. `check-source` prints the
hash when it sees a uniform tile.

### Elevation from a WMS

This works for any WMS that can return raw 32-bit floats (`image/x-bil;bits=32`) in EPSG:3857.
Earthling requests each heightmap tile directly at 2× resolution and averages it onto the exact
sample grid. The averaging matters: many servers snap to their internal grid, and without it
neighbouring zoom levels would disagree by several metres.

```toml
[[providers]]
id       = "my_dtm"
kind     = "dem"
type     = "wms"
url      = "https://example.org/wms"
layer    = "DTM_1M"
coverage = "my_region.geojson"
native_resolution_m = 1.0
oversample   = 2           # 1–4
nodata_below = -1000.0     # values below are "no data" (e.g. -99999)
requests_per_second = 4
concurrency  = 4
```

### Elevation from a STAC catalogue (GeoTIFF sheets)

For agencies that publish their terrain model as tiled GeoTIFFs listed in a STAC API. Any CRS
works: rasters are reprojected while baking.

```toml
[[providers]]
id           = "my_stac_dtm"
kind         = "dem"
type         = "stac"
url          = "https://example.org/api/stac/v1/collections/dtm/items"
asset_suffix = "_2m.tif"   # which asset of each item to download
coverage     = "my_region.geojson"
native_resolution_m = 2.0
```

Items are expected to be named `<prefix>_<year>_<sheet>`, as swisstopo does. When several
editions of a sheet exist, the newest one is used. Downloads keep a local index
(`<cache>/_sources/<id>/index.json`), so baking needs no network.

### Coverage polygons

- The shipped coverages are `france`, `switzerland` and `austria`. They come from Natural Earth
  1:10m and are shrunk by 250 m. `tools/make_coverage.py` generates them; add an entry to
  `REGIONS` there for other countries.
- Any GeoJSON geometry (Polygon / MultiPolygon, lon/lat) works as a file. It should lie
  **inside** the real data: the soft transition ends at the coverage edge, so data must exist up
  to there. Shrink an administrative boundary by a few hundred metres.
- Without a coverage, a source counts as worldwide and hides every source below it.

## Checking a source

```powershell
uv run earthling check-source <project> <id> [--at lon,lat] [--out folder]
```

Without `--at`, the check uses the track point deepest inside the coverage.

- **Imagery:** fetches the tile at the location on four zoom levels. It reports HTTP status,
  content type, size, latency and tile size, detects placeholders and uniform tiles, and checks
  the coverage. `--out` saves the tiles as a contact sheet.
- **Elevation:** bakes the location's tiles on two or three zoom levels and reports:
  - data fraction and elevation range;
  - the shift against Copernicus GLO-30, in samples, from slope correlation;
  - the mean difference;
  - LOD consistency: a tile against its four children.

  Good sources show shifts around ±0.5 samples, which is the precision of this comparison;
  shifts of 1 sample or more are flagged. The LOD mean difference should be well below 1 m,
  otherwise the terrain visibly pops when the level of detail changes.

The exit code is non-zero if there are problems, so the check can run in scripts.

## Sources that need code

Other protocols need a small Python class, for example ZIP archives, other tiling schemes,
authentication, or file names computed from coordinates:

- **Imagery:** create a `TileProvider` (`src/earthling/data/providers.py`) and add it to
  `PROVIDERS`. For anything that isn't XYZ, put the URL building into a subclass overriding
  `tile_url(z, x, y)`.
- **Elevation from files:** subclass `DemSource` (`src/earthling/data/dem.py`). Implement
  `files_for_bounds(bounds, client=None)` returning `SourceFile(url, filename, bounds_lonlat)`
  entries; see `CopernicusGlo30`. Set `listed_remotely = True` if the list needs the network.
  Downloading, reprojection and baking are then automatic.
- **Elevation per tile:** set `direct = True` and implement `fetch_tile(z, x, y, client)`
  returning the 259×259 heightmap samples (`sample_positions_mercator` gives their exact
  positions) or `None`; see `WmsDemSource`.
- Register DEM sources in `DEM_SOURCES`, then write tests with `httpx.MockTransport`
  (`tests/test_wms_dem.py` and `tests/test_swisstopo.py` show how). Finally run
  `check-source` against the real service.

Candidates for the Alps: Austria (basemap.at imagery above; ALS terrain models from the
federal states), Bavaria (DOP orthophotos and DGM1 as open data), South Tyrol (Geoportal
orthophotos and DTM) and Italy (regional geoportals). Check each service's terms before bulk
downloads, and keep the request rate low.
