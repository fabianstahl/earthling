# Earthling – Roadmap

Earthling is a desktop tool for making cinematic 3D flyover videos of GPS tracks (hikes and similar).
It loads GPX tracks, downloads satellite imagery and elevation data for the area around them, and
builds 3D terrain rendered with OpenGL shaders. Shots are animated on a keyframe timeline and exported
as 4K60 video for DaVinci Resolve.

The first use case is a month-long hike through the French Alps with a few short crossings into
Switzerland. The tool must work for any other region too.

---

## 1. Ground rules for implementation

- **Each numbered step below is one commit**, and after every step the app starts and works.
- Steps run in order. A step builds only on earlier steps.
- Every commit passes `uv run pytest` and `uv run ruff check`.
- Tests never touch the network. Downloaders are tested with mocked HTTP responses and small
  fixture files in `tests/fixtures/`.
- While executing the roadmap I don't ask questions. The only exception is a blocking problem
  with no reasonable workaround. Smaller decisions are made sensibly and noted in the commit message.
- Commit messages follow the pattern `step X.Y: <summary>`.

## 2. Decisions made during planning

| Topic | Decision |
|---|---|
| Language / UI / GL | Python 3.12, PyQt6, `moderngl` inside a `QOpenGLWidget`, OpenGL 4.3 core |
| Tooling | `uv` + `pyproject.toml`, `ruff`, `pytest` (+ `pytest-qt`) |
| Target platform | Windows first, no intentional platform lock-in |
| Data strategy | Global sources first, through a pluggable provider system. National high-res sources (IGN France, swisstopo) come later as extra providers |
| Scale | Download resolution set in zones by distance to the track; quadtree LOD terrain with async tile streaming |
| Licensing | Non-commercial use. Every provider declares its license and attribution text, and an attribution overlay is optional |
| Sun | Physically computed from lat/lon + date + time, with manual azimuth/elevation override offsets |
| Sky | Real-time single-scattering atmosphere (Rayleigh + Mie + ozone, optical-depth LUT) + sun disc + night/stars. Chosen over Preetham during 5.2 because it handles sunsets/twilight. Full multiple-scattering LUTs (Bruneton) remain an optional late upgrade |
| Effects | Height and distance fog, terrain shadows. Bloom only as needed for track glow. No volumetric clouds, no general post FX suite |
| Track | 3D tube/ribbon with glow, animated progress, moving hiker marker |
| Texture layers | Generic layer registry. Two slots (A, B) with a crossfade factor. Choosing the layer for each slot can be keyframed |
| Interpolation | Step, linear, ease in / out / in-out, cubic Bézier with handles. Camera: Catmull-Rom position splines + quaternion squad rotation |
| Camera | Free-fly + "snapshot keyframe", look-at target, follow track |
| Timeline UI | Dope sheet first, graph editor (Bézier curve editing) later |
| Export | ffmpeg: ProRes 4444 / DNxHR 444 `.mov` (visually lossless, Resolve-friendly) and H.264/H.265 MP4. Default 3840×2160 @ 60 fps |
| Persistence | `earthling.toml` (data config, edited by hand) + `scene.json` (parameters, keyframes; saved by the app). Undo/redo via `QUndoStack` |
| Extras | Peak/place labels, stats overlay (day/distance/elevation + profile), preview-quality toggle |
| Out of scope (for now) | Volumetric clouds, DOF/motion blur/supersampling, audio |

## 3. Architecture overview

```
src/earthling/
  app/        main window, docks, menus, actions (PyQt6)
  core/       config, geo math (WGS84, WebMercator, ECEF, ENU), GPX, AOI
  data/       providers, tile cache, downloader, DEM baking, OSM/vector data
  render/     GL context, terrain LOD, streaming, layers, sky, lighting, track, passes
    shaders/  GLSL files (hot-reloadable in dev mode)
  anim/       properties, keyframes, interpolation, timeline model, camera animation
  ui/         reusable widgets: property panels, timeline/dope sheet, graph editor
  export/     offscreen rendering, ffmpeg pipe
  cli.py      `earthling [gui|plan|fetch|render]`
tests/
```

Key concepts:

- **Everything is an XYZ tile pyramid (Web Mercator).** Imagery providers deliver tiles directly.
  DEM providers (GeoTIFF/COG in any CRS) are *baked* into float32 heightmap tiles in the same
  scheme (256×256 plus a 1-px border, so normals are seamless). Renderer and layers then only need
  to know about one tile scheme.
- **Cache layout:** `cache/<provider-id>/<z>/<x>/<y>.<ext>`. Shared across projects, and resumable.
- **World space:** a local East-North-Up frame centered at the project origin (centroid of all
  tracks). Vertices are placed via ECEF, so Earth curvature is correct (at 200 km distance the
  surface drops by about 3 km). Each tile keeps its own origin in float64, and vertex offsets are
  float32 relative to it (camera-relative rendering), so there's no precision jitter.
- **Terrain LOD:** chunked quadtree. Each node is one tile with a fixed grid mesh plus skirts; the
  vertex shader displaces it by sampling the heightmap. Nodes are chosen by screen-space error and
  capped by the max zoom of the zone they fall in. Imagery can use a higher zoom than geometry.
- **Property system (central):** every adjustable value (sun time, fog density, layer choice,
  camera, track progress, …) is a declared `Property` with type, range, default, UI section and
  target uniform. From that one declaration you get the UI widget, the shader binding, the keyframe
  button, serialization and undo. Adding a new shader parameter = one declaration.
- **Animation:** each animated property has a curve of keyframes. At time *t* the timeline
  evaluates all curves, pushes the values into the property store, and the renderer draws. Viewport
  preview and export use exactly the same evaluation path.
- **Determinism for export:** the export renderer waits until every tile needed for a frame is
  loaded at full LOD before it draws that frame, so there's no pop-in.

### Data sources

| Purpose | Global (first) | National high-res (later) |
|---|---|---|
| Imagery | Esri World Imagery (XYZ, up to ~z19), EOX Sentinel-2 cloudless (10 m) | IGN BD ORTHO (FR, 20 cm, Géoplateforme WMTS), swisstopo SWISSIMAGE (CH, 10 cm, WMTS) |
| DEM | Copernicus GLO-30 (30 m, COG on AWS) | IGN RGE ALTI (FR, 1–5 m), swissALTI3D (CH, 0.5–2 m, STAC GeoTIFF) |
| Topo map | OpenTopoMap (XYZ) | IGN Plan / SCAN, swisstopo Landeskarte |
| Borders | Natural Earth 10 m admin-0/1 | – |
| Labels | OpenStreetMap via Overpass (peaks, places, huts) | – |

Details of the national source endpoints (layer names, formats) get checked when those steps are
implemented.

### Config file (`earthling.toml`, example)

```toml
[project]
gpx_dir   = "gpx/"
cache_dir = "~/earthling-cache"
timezone  = "Europe/Paris"

[area]
border_km = 15.0
# resolution zones by distance to the track (first match wins)
zones = [
  { within_km = 2.0,  imagery_zoom = 18, dem_zoom = 14 },
  { within_km = 6.0,  imagery_zoom = 16, dem_zoom = 13 },
  { within_km = 15.0, imagery_zoom = 13, dem_zoom = 11 },
]

[sources]
imagery = ["esri_world_imagery"]   # priority order; later: ["ign_ortho", "swissimage", "esri_world_imagery"]
dem     = ["copernicus_glo30"]
```

### Main dependencies

PyQt6, moderngl, numpy, PyGLM, gpxpy, shapely, pyproj, rasterio, httpx, Pillow, pydantic,
imageio-ffmpeg; dev: pytest, pytest-qt, ruff.

---

## 4. Roadmap

For each step: **what gets built → what works afterwards**.

### Phase 0 – Foundation

**0.1 Project skeleton**
uv project, `pyproject.toml`, package layout, ruff + pytest config, `.gitignore`, README,
entry point `earthling` opening an empty main window with menu bar and status bar.
→ `uv run earthling` shows a window; smoke test passes.

**0.2 OpenGL viewport**
`QOpenGLWidget` hosting a moderngl context (4.3 core, bound to the widget's default FBO). Shader
loader reads GLSL from `render/shaders/` with hot reload in dev mode. Renders a test triangle,
with resize handling and an FPS counter in the status bar.
→ The window shows a GL-rendered triangle, and editing the shader file updates it live.

**0.3 Config & project folder**
Pydantic model for `earthling.toml` (schema above) with validation and useful error messages.
*File → Open Project Folder*, recent projects list. Sample config written for new projects.
→ Opening a folder loads and validates its config; errors are shown in a dialog. Tests for parsing.

### Phase 1 – Tracks & geography

**1.1 GPX loading**
Load all GPX files in `gpx_dir` (tracks, segments, timestamps, elevation) with gpxpy. Per-track
stats: distance, ascent/descent, duration, bounding box. A "Tracks" dock lists the tracks and
their stats.
→ The project's tracks are listed with correct stats. Tests with fixture GPX files.

**1.2 Geo core + first 3D view**
Conversions WGS84 ↔ WebMercator ↔ tile XYZ, WGS84 ↔ ECEF ↔ local ENU, project origin.
A basic orbit camera. Tracks rendered as plain GL lines in 3D ENU space.
→ Tracks appear as 3D lines you can orbit around. Geo math is tested against reference values.

**1.3 Area of interest & tile planning**
Buffer the union of all tracks by `border_km` (in a metric projection), apply the zones, and
compute the set of tiles for each zoom and data type. Size estimate. AOI outline drawn in the
viewport. CLI `earthling plan <project>` prints tile counts and estimated GB.
→ You can see how large the download will be before starting it, and tune the zones.

### Phase 2 – Data acquisition

**2.1 Provider system, tile cache, downloader, first imagery source**
Provider interface (id, tile URL/fetch, format, max zoom, license, attribution). Disk cache,
concurrent downloader (httpx, thread pool), per-provider rate limit, retries, resume, correct
User-Agent. Providers: Esri World Imagery, EOX Sentinel-2 cloudless. UI: *Data → Download*
with progress/cancel (worker thread). CLI `earthling fetch`.
→ Imagery tiles for the AOI get downloaded and cached; interrupted downloads resume.

**2.2 DEM provider & heightmap baking**
Copernicus GLO-30: find and download the 1°×1° COGs covering the AOI, then bake them with
rasterio into float32 heightmap tiles (Web Mercator, 256² + border) at the dem zooms of the zones.
Parent levels are built by downsampling.
→ The cache contains a seamless heightmap pyramid. Tests check tile-edge continuity and
bilinear sampling.

### Phase 3 – Terrain

**3.1 Static terrain mesh**
Render a fixed zoom level covering the AOI: grid mesh per tile, displaced by the heightmap in the
vertex shader, placed via ECEF→ENU with per-tile origins. Colored by elevation. Vertical
exaggeration setting.
→ The Alps show up as 3D terrain with the tracks on top.

**3.2 Satellite texturing & basic shading**
Imagery tiles mapped onto terrain tiles (imagery zoom ≥ geometry zoom, via UV sub-rects).
Normals from the heightmap in the fragment shader, fixed directional light (Lambert + ambient).
→ Textured, shaded terrain.

**3.3 Quadtree LOD**
Chunked quadtree with screen-space-error selection, zone max-zoom caps, skirts against cracks,
LOD bias setting. Debug view that colors tiles by zoom level.
→ The whole hike area renders, sharp near the camera and coarse far away.

**3.4 Async tile streaming**
Background loader (disk → decode → staging), GPU texture pool (texture arrays) with LRU eviction,
parent tile used as a fallback while a child loads, and a memory budget setting. Optional on-demand
download of missing tiles inside the AOI. API `wait_until_complete(view)` (used later by export).
→ Moving through the scene is smooth; tiles refine progressively; memory stays bounded.

**3.5 Free-fly camera**
Camera modes: orbit and free-fly (WASD/QE + mouse look, speed scaled by height above ground,
shift/ctrl modifiers). Minimum height above ground. HUD showing lat/lon/altitude/heading.
"Frame track" and "Go to" actions.
→ You can fly freely through the terrain like in Google Earth.

### Phase 4 – Property system & persistence

**4.1 Property system & auto-generated panels**
`Property` declarations (float, int, bool, enum, color, vec3, datetime) grouped into sections.
The property store drives the uniforms. The "Parameters" dock builds one collapsible section per
shader/system, with sliders, spinboxes, color pickers, combo boxes and checkboxes. Existing settings
(exaggeration, LOD bias, light direction) move to it.
→ All rendering parameters can be adjusted in UI panels generated from the declarations.

**4.2 Scene file (save/load)**
`scene.json` holds property values, camera and UI state, with a schema version and a migration hook.
*File → Save / Save As / Revert*, and a dirty indicator in the title bar.
→ The session state survives restarts. Round-trip tests.

**4.3 Undo/redo**
`QUndoStack`. Every property change is a command, and slider drags merge into one command.
Edit menu + Ctrl+Z / Ctrl+Y. History dock (optional).
→ Every parameter change can be undone.

### Phase 5 – Lighting, sky, fog

**5.1 Sun position & terrain lighting**
NOAA solar position algorithm (lat/lon of the scene center, date, time, timezone), plus azimuth/
elevation override offsets. Properties: date, time of day, sun intensity/color. Lighting: Lambert +
hemispherical sky ambient, normals from finer DEM data than the geometry.
→ Moving the time-of-day slider moves the sun realistically. Solar math tested against reference data.

**5.2 HDR pipeline & analytic sky**
HDR render target, exposure + ACES tone mapping. Sky from real-time single scattering (Rayleigh,
Mie, ozone; optical depths precomputed into a small LUT shared with the CPU sun color), sun disc.
Twilight reddening, night transition with a procedural star field and a dim bluish night
ambient. Sky color feeds terrain ambient.
→ Moving the time from noon to midnight turns the sky red, then dark, with stars.

**5.3 Height & distance fog**
Exponential height fog (base height, falloff, density, color) + distance-based aerial perspective
tinted by sky/sun color (in-scattering toward the sun).
→ Valley mist and atmospheric haze, adjustable in their own panel section.

**5.4 Terrain shadows**
Cascaded shadow maps from the sun (3–4 cascades fitted to the view frustum), PCF soft shadows,
bias settings, shadow strength property.
→ Mountains cast shadows, most visible at sunrise and sunset.

### Phase 6 – Texture layers

**6.1 Layer framework & DEM-derived layers**
Generic layer registry: a layer is either a *tile-source layer* (imagery pyramid) or a
*procedural layer* (a GLSL function over height/normal/position), each with its own properties.
Two slots A and B (enum properties) + crossfade factor. Layers: satellite, elevation (selectable
color ramp + range), slope, aspect, hillshade. The terrain shader is assembled from the layers
registered.
→ You can pick any two layers and crossfade between them, e.g. satellite → slope map.

**6.2 Topo map & contours**
OpenTopoMap tile-source layer. Procedural contour layer (interval, major-line interval, line
width, color), and contours can be overlaid on any other layer.
→ Topographic layers are available for blending.

**6.3 Country & region borders**
Natural Earth 10 m admin-0 (+ optional admin-1), converted into distance-field tiles so lines stay
sharp at any zoom. Border layer + optional overlay on top of both slots (width, color, glow, dashed).
→ Country borders (e.g. France/Switzerland) can be shown and blended in.

### Phase 7 – Track visualization

**7.1 3D track tube/ribbon**
Replace the GL lines with tube or flat-ribbon geometry, snapped to the DEM with a height offset
(or GPX elevation, selectable), simplified/smoothed. Width in meters or constant screen pixels.
Color per track (or per day). Per-track visibility.
→ The track is shown as good-looking 3D geometry that follows the terrain.

**7.2 Glow**
Emissive track material + a bloom pass limited to emissive objects (threshold, intensity, radius).
→ The track glows, which looks strong in dusk and night scenes.

**7.3 Progress & hiker marker**
Per-vertex cumulative-distance attribute. Properties: `progress` (0–1, by distance or mapped to
GPX timestamps), visible range start/end (so only a *part* of the track is shown), fade-in length
at the head. Hiker marker at the progress point (glowing sphere/billboard with a pulse).
→ Dragging the progress slider draws the hike with a moving marker.

### Phase 8 – Timeline & keyframes

**8.1 Animation core (logic only)**
Keyframes and curves for every property type. Interpolation modes: step, linear, ease in / out /
in-out, cubic Bézier (with handle data; handles get default values until the graph editor exists).
Type-aware blending: colors in a perceptual space (OKLab), enums/bools always step, datetimes as
continuous time. Time in seconds, snapped to project fps (default 60).
→ Fully unit-tested animation evaluation; nothing is visible yet.

**8.2 Timeline dock & playback**
Transport bar (play/pause, stop, loop, jump to start/end, frame stepping), duration and fps
settings, time ruler with scrubbing. Real-time preview playback (frames are dropped if slow).
→ You can scrub and play the timeline (nothing is animated yet).

**8.3 Keyframing properties + dope sheet**
A ◆ button next to every property widget adds/updates/removes a key at the playhead, and the widget
shows whether it is keyed or animated. Dope sheet: one row per animated property, grouped by section;
select/move/delete/copy/paste keys, set the interpolation per key (context menu). Undo + saving
included.
→ Any parameter can be animated, e.g. time of day 12:00 → 22:00 over 20 s.

**8.4 Camera keyframes & path**
Camera as an animated property (position + orientation). "Add camera keyframe" takes a snapshot of
the current free-fly view. Position via centripetal Catmull-Rom (or Bézier), rotation via quaternion
squad; option for constant-speed reparametrization. 3D camera path spline + keyframe frustum gizmos
in the viewport. Toggle between "animated camera" and "free edit camera".
→ **Example scenario works:** a flight through 5 camera keys while the day turns to night.

**8.5 Look-at & follow-track camera modes**
Look-at target (click on terrain to pick it; the target can be keyframed) and follow track (the
camera follows the hiker marker with offset, lag/smoothing, and look-ahead). The mode is keyframed
per timeline segment; mode switches are blended smoothly over a short transition.
→ Orbit shots around peaks and chase shots along the track.

**8.6 Layer switching on the timeline (acceptance)**
Layer choice for slots A/B uses step keys, and the crossfade uses any interpolation. Built-in
checks that there's no visual jump when a hidden slot switches (a warning if you switch a slot
that is visible). Sample scene in `examples/` covering every animated feature.
→ Sequences such as "satellite → slope → elevation" can be fully keyframed.

### Phase 9 – Export

**9.1 Offscreen rendering & quality presets**
Render to an offscreen FBO at any resolution, separate from the viewport. Deterministic frame
stepping with `wait_until_complete` per frame (full LOD, no pop-in). Preview-quality toggle for the
viewport (resolution scale, LOD bias, shadow resolution) vs export quality.
→ "Render current frame" saves a full-quality 4K still.

**9.2 Video export via ffmpeg**
ffmpeg (bundled via imageio-ffmpeg) is fed raw frames through a pipe, with asynchronous readback
(PBO) and 16-bit readback for 10-bit codecs. Presets: **ProRes 4444 (.mov)**, **DNxHR 444 (.mov)**,
H.264/H.265 MP4 (NVENC when available). In/out range, resolution, fps, progress/ETA dialog, cancel.
CLI `earthling render <project> --preset prores4444`.
→ **First complete pipeline:** a 4K60 video that imports directly into DaVinci Resolve.

### Phase 10 – Graph editor

**10.1 Graph editor**
Value-over-time curve view for the selected properties: Bézier handles (auto/aligned/free/vector),
drag keys and handles, zoom/pan, frame all/selected, normalized view for comparing curves, undo.
Sits in the timeline dock next to the dope sheet (tabbed).
→ Precise control over the shape of every transition.

### Phase 11 – High-resolution national data

**11.1 Coverage-aware provider compositing**
Providers declare coverage polygons. For every tile the highest-priority provider that covers it
is used, with feathered blending at coverage edges (imagery) and seam-free fallback (DEM).
`plan` shows the size per provider.
→ Sources can be mixed per region without visible seams.

**11.2 France: IGN (Géoplateforme)**
BD ORTHO imagery (WMTS, Web Mercator) + RGE ALTI DEM (1–5 m) baked into heightmap tiles, with
license/attribution (Etalab Licence Ouverte).
→ Much sharper imagery and terrain for the French Alps.

**11.3 Switzerland: swisstopo**
SWISSIMAGE (WMTS 3857) + swissALTI3D (STAC API, GeoTIFF in LV95, reprojected during baking).
→ High-res Swiss sections.

**11.4 Adding a new region (documentation + template)**
A provider template and a guide for adding more national sources (e.g. Austria basemap.at + ALS
DGM, Bavaria DOP, South Tyrol), with a provider test harness.
→ New regions can be added quickly.

### Phase 12 – Overlays

**12.1 Peak / place / hut labels**
Fetch OSM features via Overpass for the AOI (cached), with filters (type, minimum elevation,
prominence, distance to the track). 3D-anchored text rendered with an SDF font atlas, leader lines,
depth-based occlusion, decluttering, fading with distance. Visibility and filters can be keyframed.
→ Named peaks and places appear in the scene and in the export.

**12.2 Stats overlay**
2D HUD layer in viewport and export: day number, date, distance walked, current elevation, total
ascent, and an elevation profile with a moving marker. All values come from the track progress.
Layout, style and visibility properties, all keyframeable. Optional attribution text.
→ Informational overlays are baked into the video.

### Phase 13 – Optional upgrades

**13.1 Physically-based atmosphere** – precomputed atmospheric scattering (Bruneton-style LUTs)
replacing the analytic sky, with real aerial perspective.
**13.2 Terrain rendering quality** – geomorphing between LOD levels, anisotropic filtering, detail
normal maps at close range.
**13.3 Export quality** – supersampling / temporal AA for export, optional motion blur.


### Phase 14 – Real projects: track groups & time

**14.1 Track groups**
A project can load several groups of GPX files (`[[tracks]]` in `earthling.toml`), e.g. the
planned route of a long-distance trail and the tracks actually walked. Every group has its own
resolution zones (only the relevant part in high detail, the rest as a coarse overview), a style
(colour per track or single colour, width, dashed line, glow) and a role: progress, the hiker
marker, the stats and the labels follow the walked tracks, while planned routes are drawn in full.
Per-group properties (visibility, opacity, colour, highlight of one track with dimming of the
others) are keyframeable.
→ The whole trail as context, the walked part in detail.

**14.2 Sun from the GPX time**
Optionally, the sun follows the local time at the head of the drawn track (with an offset), so
a day's route animates from its real start to its real end with the matching light.
→ A day tour from sunrise to sunset without manual sun keys.

### Phase 15 – Points of interest

**15.1 POI layer**
Points of interest in the scene: position (on the map or snapped to the track), icon (PNG,
animated GIF/APNG or sprite sheet), size, caption. Built-in effects (pop-in, bounce, pulse, wobble,
spin) and frame animation speed; visibility, opacity, scale and effects are keyframeable per POI.
Anchored in 3D, hidden behind terrain, drawn crisp after tone mapping.
→ Coffee stops, closed paths, camps … appear and move in the video.

**15.2 POI editing**
A POI dock: add a POI at the hiker position or by clicking the terrain, choose an icon, edit name
and caption; every POI gets its own section in the parameter panel (keys, dope sheet, graph
editor). Placeholder icons (animated coffee, turnaround sign, tent, camera) are included.
→ POIs are created and animated without editing files.

### Phase 16 – Weather

**16.1 Fog and cloud layers**
Volumetric cloud and fog layers (three independent layers: base height, thickness, coverage,
density, noise scale, wind), raymarched against the scene depth and lit by sun and sky, with
moving cloud shadows on the terrain.
→ Drifting fog banks between the ridges, clouds casting shadows.

**16.2 Rain**
A rain cloud deck at a defined height, rain streaks below it (depth-aware, wind-driven),
darkened wet ground and reduced visibility.
→ Rainy descents look rainy.

**16.3 Thunderstorms**
Lightning strikes at a keyframeable rate (deterministic schedule), branching bolts from the
cloud base to the ground with glow, and flashes that light up clouds, sky and terrain.
→ A night storm with lightning over the mountains.

### Phase 17 – Presentation polish

**17.1 Readable tracks**
A soft, dark casing around the track lines (width, colour, opacity) so they stand out on busy
imagery; casings are drawn below all lines. Tracks no longer write depth: overlapping tracks
blend instead of hiding each other.
→ Coloured routes stay readable over forests, snow and farmland.

**17.2 German on-screen text**
A language setting for the overlays (stats labels, date and number formats): English or German.
→ Videos for a German audience without English words in the picture.

**17.3 Map styles**
Blend the imagery into a dark map (land and sea tones from the DEM, a hillshade) with glowing
country borders, keyframeable; imagery brightness and saturation controls.
→ A country overview that starts as a clean map and turns into satellite imagery.

**17.4 Smooth terrain detail**
No holes where a detail source has no data (the parent stays drawn, coarse tiles come from the
next DEM source), and refined nodes fade in from the parent's imagery along with the geomorph
instead of switching sharpness and colour at once.
→ Zooms from space to the valley without missing tiles or popping shading.

**17.5 More POI icons**
Built-in icons for power/charging, bad weather and a supermarket; icons without captions;
multi-line captions (e.g. stage markers with name, length and ascent).
→ Symbols tell the story without text.

### Phase 18 – Journeys (trips with flights, buses and ferries)

**18.1 Continental views**
An optional `[world]` section adds a coarse pyramid of the whole globe (imagery z0–z4, DEM
z0–z3, a few MB), so views from space never show holes outside the tracks' areas. A scene can
carry its own frame origin (`"origin": [lat, lon]`): far from the project origin the ENU "up"
tilts away from the local vertical (5° at 600 km), which tilts fog and cloud layers. The far
plane grows with the camera height up to the horizon seen from space.
→ One project spans a whole country (and the flight there); every scene is level where it plays.

**18.2 Travel icons**
Built-in icons for airports, ferries, buses, hotels, taxis, sights, a finish flag and a summit.
→ A journey's stops are told with symbols.

**18.3 Views from space**
Tile bounds of the coarse world tiles sample the whole tile (a zoom 0 tile's 3×3 points lie on
one meridian plane and culled the globe). Exact near/far culling planes and depth clamping
(every pass writes its own logarithmic depth; with 0.5 m to 16 000 km the float32 projection
cut holes into distant terrain). Sun light, sky, aerial perspective, height fog and cloud
heights use the camera's own local vertical and location instead of the frame origin's.
→ The Earth as a globe with its atmosphere rim, down to the streets of a city in one shot.

**18.4 Travel lines**
Several walked groups form one journey in time order (e.g. hikes, bus rides, ferries, flights:
progress, the sun from the GPX time and the drawn part follow the timestamps across groups).
Per group `elevation = "gpx"` keeps the file's heights (a flight arc in the air), and a
"Dotted" style draws round dots that neither crawl nor pop while zooming.
→ Flights and bus rides appear as dotted lines between the hikes, at the right time.

**18.5 Resolution-independent sizes**
Track widths, casing, outline, dashes and dots, the hiker marker, border and contour lines are
given for 1080p and scale with the image height (like labels, POIs and the HUD already did);
the glow leaves out its finest levels at higher resolutions, so it keeps its size relative to
the picture.
→ A 4K export looks like the 1080p preview, only sharper.

**18.6 Colour harmonisation**
`[sources] colour_reference = "eox_s2cloudless_2024"`: the imagery keeps its detail but takes
its large-scale colours from one consistent mosaic (`imagery * blur(reference) / blur(imagery)`
on a block grid that terrain levels 7-12 share; block means come from the imagery each node
shows). Esri's per-zoom mosaics (colour rectangles, sun glint around islands, clouds over the
sea) no longer show.
→ Flights over a whole country without patchwork.

**18.7 Rain seen from above**
The rain haze (reduced visibility) only applies to the part of a view ray below the rain
clouds: from above, the rain deck shows as dark clouds with the ground between them instead of a
grey veil over the whole picture. The height fog integral is evaluated with bounded exponents
(seen from hundreds of kilometres up it multiplied exp(-1000) by exp(+1000), giving frames full
of fog colour).
→ Flights out of the rain and over fog look right.

### Phase 19 – Showcase features (ten very different scenes)

Ten synthetic showcase scenes (`examples/showcase`: Patagonia from space, the Hurtigruten,
the Stelvio by bike, the Grand Canyon, Kilimanjaro, Iceland, the Berlin marathon, Everest, the
Boundary Waters, the Trans-Siberian) each push a different part of the app. These steps add
what they need.

**19.1 Wind over time and gusts**
The clouds and the rain deck drift by the integral of the wind velocity over the timeline
(cached tables), not speed x time: a keyed wind speed or direction changes how fast the
clouds move, never where they are. `weather.gusts` / `weather.gust_period` let the wind of
all layers (and the slant of the rain) speed up and calm down in gusts.
→ A storm that builds up, with clouds racing in gusts, without jumps.

**19.2 Banking camera**
`follow.bank`: in follow mode the camera rolls into the turns of the track (the change of
direction over 250 m before and after the head), like a pilot or a cyclist.
→ First-person flights that lean into the valley curves.

**19.3 Local time on journeys**
`stats.timezone = "position"`: the clock in the stats overlay shows the time zone where the
head is (tzfpy), optionally with the UTC offset (`stats.show_utc_offset`), so a train across
Siberia or a flight to Asia shows the local time of the place it passes.
→ The clock jumps with every time zone the journey crosses.

**19.4 Portrait and square videos**
Export resolutions 1080 x 1920, 2160 x 3840 (9:16) and 1080 x 1080. All on-screen sizes
(tracks, marker, labels, POIs, stats, borders, lightning, glow) scale with the shorter image
side instead of the height, so a portrait video keeps the proportions of a landscape one.
→ Clips for phones and social media from the same scene.

**19.5 Speed, gradient and heart rate**
The GPX reader keeps the heart rate (Garmin / Strava `TrackPointExtension`). Every track point
gets its elevation, gradient (over +- 60 m), speed (over +- 15 s) and heart rate as vertex
values: `tracks.color_by` colours the walked tracks by one of them with a colour ramp
(range and ramp keyframeable), and the elevation profile of the stats overlay takes the same
colours. The stats overlay can show the speed (km/h, pace in min/km or knots), the gradient
and the heart rate, and an extra line of text (`stats.text`, a new keyframeable text property
type that switches at its keys, e.g. a hairpin counter or chapter titles).
→ A bike climb with its steep ramps in red, the speed and the pulse at the head.

**19.6 Snow and rainbows**
`rain.type = "snow"`: slow, tumbling flakes (round on screen at any viewing angle, drifting
at most ~50 degrees in the wind) instead of rain streaks, and no wet ground. `snow.cover`
whitens the imagery above a snow line with a noisy, patchy edge, keeps steep rock and dark
outcrops free and a little of the imagery's detail in the snow. `rain.rainbow`: the primary
and secondary bow (with the dark band between them and the brighter sky inside) at 42 and 51
degrees around the antisolar point, wherever sunlit rain is between the camera and the scene
(the terrain cuts the bow, nothing above the rain clouds).
→ Fresh snow and a blizzard at Everest Base Camp; a rainbow over Iceland after a shower.

**19.7 Day and night side, city lights**
Seen from far up (blended in from 20 to 200 km camera height) every point of the terrain is
lit by its own sun elevation: the sunlight through the atmosphere's transmittance LUT (red at
the terminator) and the ambient light from its day and night parts (as a ratio of the
camera's, so the weather's dimming and lightning stay in), instead of the light at the
camera everywhere. `night.lights` adds the city lights of NASA's Black Marble (VIIRS 2016, a
new "night" tile source, zoom 8, downloaded on demand) where it is night, switching on in the
dusk.
→ The day-night line sweeping over Siberia, the cities lighting up behind it.

**19.8 Vehicle markers**
Per track group a head marker: the glowing dot or a vehicle icon (ship, plane, train, car,
canoe seen from above and turned to the direction of travel on screen; bike, runner, hiker
from the side, mirrored when the track runs to the left), with its own size. A journey's head
turns into a ship on the ferry and into a plane on the flight. Built-in icons in
`assets/icons/vehicles` (drawn by `tools/make_poi_icons.py`).
→ The Hurtigruten ship sailing up the coast, the cyclist on the Stelvio hairpins.

**19.9 Several heads at once**
`progress.mode = "clock"`: every walked track is drawn up to the same clock time, each with
its own marker (in a light version of its line colour, or its group's vehicle). A time shift
per group (`trackgroup.<name>.time_offset`) lines up tracks of different days, e.g. last
year's run with this year's. The first group of the project leads: the follow camera, the sun
and the stats (of that track alone) follow it, and `stats.show_gap` shows how far (in time)
the second track is behind.
→ A ghost race through Berlin: 2025 against 2024, with the gap in the overlay.

**19.10 Water surface**
A water mask per LOD node from the OpenFreeMap vector tiles (OpenMapTiles `water` layer: the
OSM coastline, lakes and river areas, up to zoom 14, a few metres accurate; a small MVT
decoder, tiles downloaded on demand), rasterised with supersampling plus a blurred "open
water" channel. Where the DEM is the only source the sea level is the fallback. On the water
the imagery stays the colour of the water body and gets a Fresnel reflection of the sky
(sky-view LUT), the sun's glint after Cox & Munk (slope variance from the wind) and moving
waves close up: a sum of deep-water waves around the wind direction, of which only those the
pixel resolves are drawn; the slopes of the others turn into roughness, so nothing shimmers
and the glint widens with the distance. Waves and glint calm down along the shore. Properties
`water.*` (reflection, waves, wind, direction).
→ Fjords glittering in the midnight sun, the sun's glint on the Gulf of Thailand from space.

**19.11 Tracks on water**
`water = true` in a `[[tracks]]` group splits its tracks by the water mask (19.10): every
point knows whether it is on water, and the parts on water take the group's water style
(`trackgroup.<name>.water_color`, dashes or dots), the rest the normal one.
→ A canoe trip: blue lines across the lakes, dotted portages between them.

**19.12 The whole globe, smoother gradients**
Found on the Patagonia dive: the terrain normals were turned into the world with the tangent
frame of the node centre, which coarse nodes (a quarter of the Earth at zoom 1) get wrong
towards their borders - the shading jumped along meridians seen from space. Every fragment now
builds its frame from its own ellipsoid normal and the Earth's axis (the heightmap grid runs
east-west everywhere). The polar caps beyond the Web Mercator tiles (85.05 degrees) were a
dark blue disc; the sky pass now draws them as lit ice. The per-point values (19.5) take the
gradient from the file's smoothed elevations where it has them (a road draped on a 30 m DEM
goes up and down at every hairpin), and the stats profile uses the same values as the line.
→ A clean globe from 14 000 km, Antarctica included.

**19.13 Fixes from the showcase scenes**
Labels: names in a script the label font cannot draw (Devanagari, Chinese, Thai ...) showed
as boxes around Everest; they now fall back to `name:en` / `int_name` (or are left out). By
clock (19.9), the lead group's line is drawn on top of the others on a shared course (the
ghost run lay over this year's line and turned its colour pale).
Clouds seen at a grazing angle (a deck towards the horizon) showed fine stripes along their
shapes: the raymarch's last steps on those long segments were over a kilometre long. They now
get up to 2.5 times the steps, the light samples towards the sun are dithered, and every
anti-aliasing sample uses its own dither pattern.
Rain streaks seen steeply from above converged into concentric arcs around the nadir (the
lanes of all layers meet there, spiralling with the wind): they fade out towards steep views,
where the haze still shows the rain.
Seen from far up, the aerial perspective volume (32 slices up to 300 km from the camera) drew
rings around the camera: all the haze lies in the last kilometres above the ground, within one
slice. From 12 km camera height up it is integrated per pixel along the ray instead (32 steps,
dense near the ground; the volume itself now integrates its slices cumulatively), which also
gives views from space their blue veil. On the water the waves fade out earlier (20 pixels per
wavelength) and the sun's glint is capped lower.
Seen from space, the whole day side changed colour in one frame when the sun rose at the
camera: the sky light's tint came from the sunlight's transmittance at the camera, which
underflows below the horizon (white) and turns red at sunrise; it now uses the sun on the
horizon there, and the bounce light from the ground takes the sun's visibility at each point.
→ "Mount Everest", "Lhotse", "Ama Dablam" instead of boxes; this year's line stays orange;
a smooth monsoon sky over the Grand Canyon.

**19.14 Aerial perspective per pixel**
The aerial perspective is integrated along every view ray (24 steps, dense near the ground for
cameras above 3 km) instead of being looked up in the 32-slice froxel volume, whose slices
showed as rings around the camera sweeping over the terrain while zooming. (The volume stays
as a fallback, `u_ap_direct` < 1.)
→ Zooms without haze rings.

**19.15 Stats panel with a fixed width**
The panel's width comes from the widest each value can get (digits as "8", the longest month
name, "888.8 km/h" ...), so it no longer grows and shrinks while the numbers change.
→ A steady overlay box.

**19.16 Less uniform clouds**
`weather.layerN.variation`: the base and the top of the clouds vary across a layer (two
coverage-map lookups), so they no longer all start at one flat height; a slow domain warp of
the pattern breaks up rows of similar round blobs, and the vertical profile is softer.
→ Cloud fields that look grown, not stamped.

**19.17 Waves into the distance**
The wave spectrum reaches from 300 m swell down to 1.3 m ripples (22 components), and every
component fades over a wide range of pixel footprints: the water keeps its pattern into the
distance instead of turning flat a few hundred metres from the camera.
→ No visible edge where the waves stop.

**19.18 Coloured vehicles with a caption**
Per track group `marker_tint` multiplies the vehicle icon and `marker_label` writes a caption
above it that moves with it (in a light version of the tint).
→ A race of a gold "2025" runner against a silver "2024" one.

**19.19 No labels in the clouds**
The cloud pass also writes its transmittance into a second target (the "veil"); the labels'
visibility test reads it at the anchor, so a peak hidden in or behind clouds loses its label
instead of leaving it floating on the cloud.
→ "Mount Everest" only where Everest is seen.

**19.20 Mini-map**
`minimap.*`: the whole journey on a small north-up map in a corner of the picture: land and
water from the OpenStreetMap water polygons (the vector tiles of 19.10, clipped per tile,
simplified and triangulated), the country borders, the route faint and its drawn part in the
track colours, a dot at the head (by clock one per track). The geometry is built once per
path and only scaled per frame.
→ Viewers see where on the map the ship or the camper is.

**19.21 Country and city labels**
`labels.countries`, `labels.cities`, `labels.min_population`: country names (in capitals,
larger, centred on Natural Earth's label point, English or German by `stats.language`) and the
large cities (Natural Earth populated places) at any scale. Very large areas skip the Overpass
query (it timed out for Moscow – Beijing, and the timeout reply was cached as an empty area: such
replies are no longer cached). The label fade-out distance now reaches 5000 km. The snow cover's
noise fades to its mean once a pixel covers its period (no speckles on distant slopes).
→ From a satellite's height viewers see which country and which city the train passes.

**19.22 Livelier clouds, even open water**
The cloud coverage varies at two scales (fields of cloud with wide clear areas between them),
a coarser noise octave mixes big and small clouds, and a multiple-scattering term keeps the
inside and the shadowed parts of clouds from turning dark grey (a closed deck above still
shades them). From high up the layers stay visible up to three times the camera's height
(before: 150 km), so they can fade in softly during a descent. `water.uniform` /
`water.color`: open water away from the shore takes one even colour instead of the imagery,
whose sea differed from tile to tile.
→ Clouds look less like a uniform pattern; seen from 100 km the sea has no seams.

**19.23 Cloud fields that keep their coverage; a clearer mini-map route**
The cloud clusters of 19.22 shifted the local coverage up and down and clamped it at zero,
which raised the mean: 5 % coverage looked like 30 %. The local coverage is now up to three
times the set one inside the cloud fields and zero in the gaps (mirrored for a mostly covered
sky), so the mean stays the set coverage and the fields stay clustered at any coverage. The
drawn route on the mini-map is thicker, with a dark outline. A second sample of the cloud noise, turned by 37° and
scaled by 1.31, hides the noise texture's period: sparse clouds seen from 100 km no longer line
up in a regular grid.
→ Sparse fair-weather cumulus seen from 100 km really is sparse; the route on the mini-map
reads on any land colour.

---

## 5. Risks & notes

- **Download volume:** high zoom levels grow 4× per level. The zones and `earthling plan` step exist
  to keep this under control; the defaults are conservative.
- **Tile provider terms:** Esri/OpenTopoMap have usage policies (rate limits, no bulk scraping at
  scale, attribution). The downloader respects rate limits and sends an identifying User-Agent.
  Non-commercial use only, as decided.
- **Copernicus GLO-30 is a surface model** (it includes forests and buildings). That's acceptable for
  flyovers; national DTMs (Phase 11) are bare-earth.
- **Python performance:** per-frame Python work stays small (property evaluation, LOD selection).
  Heavy work goes to numpy, background threads and the GPU.
- **moderngl + QOpenGLWidget:** the context must render into the widget's default FBO, which is
  handled once in step 0.2.
