# Project configuration (`earthling.toml`)

A project is a folder with an `earthling.toml`, the GPX files and the scenes (`scenes/*.json`
or `scene.json`). Paths in the file are relative to the project folder; `~` is expanded.
Unknown keys are errors, so typos show up when the project is opened. The model lives in
`src/earthling/core/config.py`.

## `[project]`

| Key | Default | Meaning |
|---|---|---|
| `gpx_dir` | `"gpx"` | folder with the GPX files (used when there are no `[[tracks]]` groups) |
| `cache_dir` | `"~/earthling-cache"` | tile cache, shared between projects |
| `timezone` | `"UTC"` | IANA time zone of the tracks (sun position, clock in the stats) |

## `[area]`

The downloaded area is the union of all tracks, buffered by `border_km`. Zones give the
resolution by distance to the tracks; the first zone that contains a point wins, and the
outermost zone must reach at least `border_km`.

```toml
[area]
border_km = 8.0
zones = [
  { within_km = 1.5, imagery_zoom = 16, dem_zoom = 13 },
  { within_km = 4.0, imagery_zoom = 15, dem_zoom = 12 },
  { within_km = 8.0, imagery_zoom = 13, dem_zoom = 11 },
]
```

`imagery_zoom` is 0-22, `dem_zoom` 0-18. `earthling plan <project>` shows the tile counts and
the download size per zone and source.

## `[world]` (optional)

A coarse pyramid of the whole globe, so views from space have no holes outside the tracks'
area: `imagery_zoom` (default 4, at most 6) and `dem_zoom` (default 3). Zoom 4 imagery is 341
tiles, a few MB.

## `[sources]`

Data sources in priority order. A regional source is used where it has data (blended into the
next one over `feather_m` at its edge), the worldwide ones everywhere else.

| Key | Default | Meaning |
|---|---|---|
| `imagery` | `["esri_world_imagery"]` | satellite imagery |
| `dem` | `["copernicus_glo30"]` | elevation |
| `topo` | `["opentopomap"]` | topographic map layer (downloaded on demand) |
| `colour_reference` | none | e.g. `"eox_s2cloudless_2024"`: the imagery takes its large-scale colours from this consistent mosaic (no colour rectangles between zoom levels) |
| `colour_reference_max_zoom` | 12 | finest zoom of the reference |
| `colour_reference_strength` | 1.0 | 0 = off, 1 = full correction |

Built-in ids: imagery `esri_world_imagery`, `eox_s2cloudless_2024`, `ign_bdortho` (France),
`swisstopo_swissimage` (Switzerland); elevation `copernicus_glo30`, `aws_terrarium`,
`ign_rgealti` (France), `swisstopo_alti3d` (Switzerland); topo `opentopomap`.
`earthling check-source <project> <id>` tests a source at a location.

## `[[tracks]]`: track groups

Without `[[tracks]]` all GPX files of `gpx_dir` form one walked group. With groups, every
group has its own files, role, detail zones and keyframeable style (`trackgroup.<name>.*`
in the Parameters panel).

```toml
[[tracks]]
name  = "walked"             # lower case, digits and _
gpx   = "gpx/walked/"        # a folder, a file or a glob pattern
label = "Walked"

[[tracks]]
name  = "official"
gpx   = "gpx/official/*.gpx"
role  = "planned"            # drawn in full; progress, marker and stats ignore it
spacing_m = 50.0
[tracks.area]                # its own (coarser) zones
border_km = 3.0
zones = [{ within_km = 3.0, imagery_zoom = 13, dem_zoom = 11 }]
```

| Key | Default | Meaning |
|---|---|---|
| `name` | required | id used in property names |
| `gpx` | required | folder, file or glob pattern |
| `label` | name | shown in the UI |
| `role` | `"walked"` | `walked`: the journey (progress, marker, stats, labels follow it); `planned`: e.g. an official route, always drawn in full |
| `area` | the project's `[area]` | the group's own zones |
| `spacing_m` | by role | resampling of the line geometry |
| `elevation` | Tracks setting | `"gpx"` keeps the file's heights (a flight arc in the air), `"dem"` puts the line on the terrain |
| `labels` | walked groups | whether the label area includes the group |
| `water` | false | split the line by the water mask: the parts on water get their own style (`trackgroup.<name>.water_*`, e.g. paddling and portages) |
| `distance_marks` | none | known distances (see below) |
| `distance_end_km` | none | the true length of each track |

Several walked groups (hikes, bus rides, ferries, flights) form one journey in time order.

### True distances

GPS watches often count a few kilometres too many. Marks correct the distance shown in the
stats and the profile (piecewise linear; the drawn line is unchanged). A mark is placed by
position (the nearest track point after the previous mark) or by time (an ISO time with
offset, e.g. a photo without GPS):

```toml
[[tracks]]
name = "walk"
gpx  = "gpx/race.gpx"
distance_end_km = 100.0
distance_marks = [
  { km = 10, lat = 50.1377, lon = 8.5581 },
  { km = 30, time = 2026-10-10T17:57:06+02:00 },
]
```

## `[[providers]]`: your own data sources

Imagery or topo tiles from any XYZ/WMTS URL, elevation from a WMS (float32 BIL) or from
GeoTIFF sheets listed by a STAC collection, each with a coverage area. See
[adding-a-region.md](adding-a-region.md) for the keys and a worked example.
