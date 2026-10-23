# Earthling – Roadmap

Earthling is a desktop tool for making cinematic 3D flyover videos of GPS tracks (hikes and
similar). It loads GPX tracks, downloads satellite imagery and elevation data for the area
around them, and builds 3D terrain rendered with OpenGL shaders. Shots are animated on a
keyframe timeline and exported as 4K60 video for DaVinci Resolve.

The first use case was a month-long hike through the French Alps with a few short crossings
into Switzerland; the tool works for any other region too.

- How it is built: [docs/architecture.md](docs/architecture.md)
- `earthling.toml`: [docs/configuration.md](docs/configuration.md)
- New data sources: [docs/adding-a-region.md](docs/adding-a-region.md)

---

## 1. Ground rules for implementation

- **Each numbered step is one commit**, and after every step the app starts and works.
  Commit messages follow `step X.Y: <summary>` (clean-ups: `cleanup N: <summary>`).
- Steps run in order. A step builds only on earlier steps.
- Every commit passes `uv run pytest` and `uv run ruff check`.
- Tests never touch the network. Downloaders are tested with mocked HTTP responses and small
  fixture files in `tests/fixtures/`.
- While executing the roadmap I don't ask questions. The only exception is a blocking problem
  with no reasonable workaround. Smaller decisions are made sensibly and noted in the commit
  message.

## 2. Decisions made during planning

| Topic | Decision |
|---|---|
| Language / UI / GL | Python 3.12, PyQt6, `moderngl` inside a `QOpenGLWidget`, OpenGL 4.3 core |
| Tooling | `uv` + `pyproject.toml`, `ruff`, `pytest` (+ `pytest-qt`) |
| Target platform | Windows first, no intentional platform lock-in |
| Data strategy | Global sources through a pluggable provider system; national high-res sources (IGN France, swisstopo) as extra providers inside their coverage |
| Scale | Download resolution set in zones by distance to the track; quadtree LOD terrain with async tile streaming |
| Licensing | Non-commercial use. Every provider declares its license and attribution text; the attribution overlay is optional |
| Sun | Physically computed from lat/lon + date + time (or the GPX time at the head), with manual offsets |
| Sky | Precomputed atmospheric scattering LUTs (Hillaire 2020, step 13.1) with per-pixel aerial perspective, sun disc, night sky with stars |
| Effects | Height and distance fog, cascaded terrain shadows, bloom for glowing tracks, volumetric cloud layers, rain, snow and lightning (phase 16, 19), anti-aliasing and motion blur for exports |
| Track | Screen-space ribbon with glow and casing, animated progress, a marker or a vehicle at the head |
| Texture layers | Generic layer registry. Two slots (A, B) with a crossfade factor; the layer of each slot can be keyframed |
| Interpolation | Step, linear, ease in / out / in-out, cubic Bézier with handles. Camera: Catmull-Rom position splines + quaternion squad rotation |
| Camera | Free-fly + "snapshot keyframe", look-at target, follow track; scene scripts plan smooth cameras with the scene kit |
| Timeline UI | Dope sheet and graph editor (Bézier curve editing) |
| Interface | One dark theme from colour tokens, Inter and the Phosphor icons; an inspector with one category at a time, search, a command palette and a scene list (phase 21) |
| Export | ffmpeg: ProRes 4444 / DNxHR 444 `.mov` (visually lossless, Resolve-friendly) and H.264/H.265 MP4. Default 3840×2160 @ 60 fps; portrait and square formats |
| Persistence | `earthling.toml` (data config, edited by hand) + scene JSON files (parameters, keyframes; saved by the app or written by scene scripts). Undo/redo via `QUndoStack` |
| Out of scope (for now) | Depth of field, audio |

## 3. Changelog

All phases below are implemented. The full text of every step is in its commit message
(`git log --grep "step 12."`).

| Phase | Steps | What it added |
|---|---|---|
| 0 Foundation | 0.1 project skeleton, 0.2 OpenGL viewport, 0.3 config & project folder | the app window with a moderngl viewport (shader hot reload), validated `earthling.toml`, recent projects |
| 1 Tracks & geography | 1.1 GPX loading, 1.2 geo core + first 3D view, 1.3 area of interest & tile planning | tracks with stats, WGS84/ECEF/ENU/Web Mercator maths, `earthling plan` |
| 2 Data acquisition | 2.1 providers, cache, downloader, 2.2 DEM baking | resumable, rate-limited tile downloads (`earthling fetch`), a seamless heightmap pyramid from Copernicus GLO-30 |
| 3 Terrain | 3.1 static mesh, 3.2 imagery & shading, 3.3 quadtree LOD, 3.4 async streaming, 3.5 free-fly camera | textured LOD terrain streamed in the background within a memory budget, orbit and fly cameras |
| 4 Properties & persistence | 4.1 property system & panels, 4.2 scene file, 4.3 undo/redo | one declaration per value gives widget, uniform, keys, JSON and undo |
| 5 Lighting, sky, fog | 5.1 sun & lighting, 5.2 HDR & sky, 5.3 fog, 5.4 shadows | NOAA sun, HDR with tone mapping, twilight and stars, height fog, cascaded shadows |
| 6 Texture layers | 6.1 layer framework, 6.2 topo map & contours, 6.3 borders | crossfading between satellite, elevation, slope, aspect, hillshade, topo, contours and borders |
| 7 Tracks | 7.1 ribbon, 7.2 glow, 7.3 progress & marker | glowing tracks that draw themselves with a marker at the head |
| 8 Timeline & keyframes | 8.1 animation core, 8.2 timeline & playback, 8.3 keyframing + dope sheet, 8.4 camera keys & path, 8.5 look-at & follow modes, 8.6 layer switching | everything keyframeable, camera paths with smooth splines |
| 9 Export | 9.1 offscreen rendering, 9.2 video via ffmpeg | deterministic frames without pop-in, ProRes/DNxHR/H.264/H.265 (`earthling render`) |
| 10 Graph editor | 10.1 | Bézier curve editing of every animated property |
| 11 National data | 11.1 coverage-aware compositing, 11.2 IGN France, 11.3 swisstopo, 11.4 adding a region | regional imagery and DEMs feathered into the worldwide ones, project-defined `[[providers]]`, `earthling check-source` |
| 12 Overlays | 12.1 labels, 12.2 stats overlay | OSM peak/pass/place/hut labels with occlusion and decluttering, a stats panel with profile |
| 13 Upgrades | 13.1 precomputed atmosphere, 13.2 geomorphing & detail normals, 13.3 anti-aliasing & motion blur | Hillaire sky LUTs, smooth LOD transitions, supersampled exports |
| 14 Real projects | 14.1 track groups, 14.2 sun from the GPX time | walked and planned groups with their own zones and styles, light that follows the hike's clock |
| 15 Points of interest | 15.1 POI layer, 15.2 POI editing | animated icons with captions and effects, a POI dock |
| 16 Weather | 16.1 fog & cloud layers, 16.2 rain, 16.3 thunderstorms | volumetric clouds with shadows, rain with wet ground, lightning |
| 17 Presentation | 17.1 readable tracks, 17.2 German text, 17.3 map styles, 17.4 smooth terrain detail, 17.5 more POI icons | casings, an English/German overlay, a dark map style with glowing borders, zooms without holes |
| 18 Journeys | 18.1 continental views, 18.2 travel icons, 18.3 views from space, 18.4 travel lines, 18.5 resolution-independent sizes, 18.6 colour harmonisation, 18.7 rain from above | trips with flights, buses and ferries across a country, the globe from space, scene origins, `colour_reference` |
| 19 Showcase features | 19.1 wind & gusts, 19.2 banking camera, 19.3 local time, 19.4 portrait/square, 19.5 speed/gradient/HR, 19.6 snow & rainbows, 19.7 day/night side & city lights, 19.8 vehicles, 19.9 several heads, 19.10 water surface, 19.11 tracks on water, 19.12 the whole globe, 19.13 showcase fixes, 19.14 per-pixel aerial perspective, 19.15 fixed-width stats, 19.16 less uniform clouds, 19.17 waves into the distance, 19.18 coloured vehicles with captions, 19.19 no labels in clouds, 19.20 mini-map, 19.21 country & city labels, 19.22 livelier clouds, even open water, 19.23 cloud fields keep coverage, 19.24 bus marker, 19.25 lines end at their points, 19.26 still cameras stay still, 19.27 no dark flashes at the memory limit, 19.28 steady dotted lines, 19.29 stats of the current track, 19.30 photo overlays, 19.31 true distances, 19.32 aid & beer pong icons, 19.33 clock without date, 19.34 labels when passed, 19.35 colours linearised once, 19.36 no crash from concurrent coverage tests, 19.37 round snowflakes in wind along the view, 19.38 labels that do not blink, a smooth vehicle heading, 19.39 clouds hide only the labels behind them, 19.40 glow from just outside the frame | ten synthetic showcase scenes (`examples/showcase`) and everything they needed |
| Cleanup (Oct 2026) | cleanup 1–11 | dead code removed, shared helpers instead of copies, core independent of data/render, `earthling.session`, split render modules, `gl_util`, `earthling stills` / `earthling qa`, the scene kit (`earthling.scenekit`), GUI widget helpers, test helpers, these docs |
| 20 Everything in the UI | 20.1 POI library, text badges, placing POIs, 20.2 POIs appear when reached, 20.3 Photos dock, 20.4 Animation menu generators, 20.5 scene frame, 20.6 project settings, 20.7 new lines, 20.8 test a source, 20.9 journey numbers | what the scene scripts of the private projects did is now possible in the application |
| 21 A new interface | 21.1 dark theme, Inter and Phosphor icons, 21.2 compact editors, conditional rows, section switches and tabs, 21.3 inspector with categories, search and filters, 21.4 scene list, 21.5 command palette, 21.6 looks, 21.7 timeline and viewport controls, export frame, 21.8 start screen, 21.9 fitting forms, 21.10 README | one category at a time instead of a list of 274 settings, search in English or German, the scene in one list, Ctrl K for everything, a preview that matches the export on any screen |

## 4. Open points

- Rendering is deterministic up to small run-to-run differences (a fraction of a colour level
  on average, single pixels more): label heights depend on which heightmaps happen to be
  loaded when the labels are enriched.
- From the showcase reviews: the twilight model is too dark in high-latitude summer nights;
  cloud layers have no vertical profile (no cumulonimbus); a flat far cloud band at the
  horizon; a pale band at the top edge beyond the loaded terrain; colour matching of mixed
  land/sea imagery blocks.

## 5. Risks & notes

- **Download volume:** high zoom levels grow 4× per level. The zones and `earthling plan`
  keep this under control; the defaults are conservative.
- **Tile provider terms:** Esri/OpenTopoMap have usage policies (rate limits, no bulk scraping
  at scale, attribution). The downloader respects rate limits and sends an identifying
  User-Agent. Non-commercial use only.
- **Copernicus GLO-30 is a surface model** (it includes forests and buildings). That's
  acceptable for flyovers; national DTMs (phase 11) are bare-earth.
- **Python performance:** per-frame Python work stays small (property evaluation, LOD
  selection). Heavy work goes to numpy, background threads and the GPU.
- **moderngl + QOpenGLWidget:** the context renders into the widget's default FBO (step 0.2).
