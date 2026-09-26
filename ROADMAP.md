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
Load finer tiles ahead of the camera (and a margin beyond the frame), no holes where a detail
source has no data, and fade between detail levels instead of switching.
→ Zooms from space to the valley without missing tiles or popping shading.

**17.5 More POI icons**
Built-in icons for power/charging, bad weather and a supermarket; icons without captions;
multi-line captions (e.g. stage markers with name, length and ascent).
→ Symbols tell the story without text.

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
