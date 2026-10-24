# Architecture

Earthling turns GPX tracks into cinematic 3D flyover videos. A **project** (a folder with GPX
files and an `earthling.toml`) is loaded into a **session**, a **scene** (a JSON document of
property values and keyframes) describes one video, and the **renderer** draws a scene at any
time of its timeline: interactively in the GUI or offscreen for stills and video export.

## Packages and layers

```
src/earthling/
  core/       model, no GL and no Qt: config, geo math, GPX, AOI, properties, animation,
              scene document, POIs, photos, sun, easing
  data/       data sources: providers, tile cache, downloader, DEM baking, imagery
              compositing, colour harmonisation, labels, borders, water, download jobs;
              every download goes through data/net.py (retries, offline pause per host)
  session.py  Session: a loaded project = core model + data sources (+ the renderer's
              track-group properties); used by GUI, CLI, scene scripts and tests
  render/     moderngl renderer: terrain LOD, sky, lighting, shadows, weather, tracks,
              labels, POIs, HUD, mini-map, photos; shaders/ holds the GLSL
  export/     offscreen frames, stills, video encoding (ffmpeg), video QA
  scenekit/   helpers for scene scripts (Python files that build scenes)
  app/        PyQt6 application: main window (main_window + window_* mixins: actions,
              palette, project, items), viewport (+ its overlay), scene list, item editors
              (POIs, photos), looks, start screen, dialogs, controllers
  ui/         reusable Qt widgets: theme (colour tokens, Inter, Phosphor icons), property
              editors and sections, inspector, command palette, timeline, dope sheet,
              graph editor
  cli.py      `earthling gui | plan | fetch | render | stills | qa | check-source`
```

Imports point downwards only:

```
app, ui, cli  ->  scenekit, export  ->  render  ->  session  ->  data  ->  core
```

with two deliberate exceptions: `core.scene.Scene` builds its property registry from
`render.parameters` (the scene's schema is the renderer's property declarations), and
`session` declares the per-track-group properties via `render.parameters`. `core.session`,
`render.tracks`, `render.hud` and `render.labels` re-export names that moved, so existing
scene scripts keep working.

## A frame, step by step

1. `Session(Project.load(folder))` loads the GPX groups, computes the area of interest and the
   tile plan, and resolves the data sources in priority order.
2. `session.load_scene(path)` returns a `Scene`: a `PropertyStore` (every adjustable value),
   an `Animation` (a curve of keys per animated property), POIs, photos and an optional frame
   origin.
3. `export.frames.scene_renderer(ctx, session, scene)` sets up a `render.renderer.Renderer`
   (tracks, terrain source, labels, POIs, photos, credits). The GUI's viewport does the same
   for its widget context.
4. `FrameRenderer.render(t, w, h)` applies the animation at `t` (values go into the store),
   asks the `CameraRig` for the pose (keyed path, look-at or follow mode), waits until every
   terrain tile of the view is loaded, renders and reads the image back. With anti-aliasing
   or motion blur it averages jittered sub-frames.
5. `export.video` pipes the frames to ffmpeg; `export.stills` saves PNGs and a contact sheet;
   `export.qa` checks a finished video for jumps and flashes.

`Renderer.render` draws, in order: shadow cascades, sky, terrain (layers, water, aerial
perspective), tracks and the marker, lightning, clouds and rain, the glow (bloom) and tone
mapping, then the screen-space overlays in output pixels: labels, POIs and vehicles, photos
and the HUD (stats panel and mini-map).

## Key concepts

- **Properties.** Every adjustable value is declared once as a `PropertyDef` (type, range,
  default, UI section, optional shader uniform), mostly in `render/parameters.py` (plus
  `layers.py`, `weather.py`, `lightning.py`, `core/pois.py`, `core/photos.py` and
  `render.parameters.track_group_properties`). From that declaration come the UI widget, the
  shader binding (`bind_uniforms`), keyframing, serialisation and undo. A new shader
  parameter is one declaration with `uniform="u_name"`.
- **The inspector.** `ui/inspector.py` maps section titles to the rail's categories (a new
  section lands in "More" until a category lists it) and builds only the shown sections.
  The declarations steer it without affecting rendering: `advanced=True` puts a setting
  behind "More settings", `when=shown_if(...)` / `shown_unless(...)` hides it while it has
  no effect, `switch=True` makes a boolean its section's on/off switch, `keywords=` helps the
  search, and labels "<tab>: <setting>" share tabs (the cloud layers). Editor slots are
  bound methods: PyQt disconnects them when a rebuilt view is destroyed, not lambdas.
- **The export frame.** The viewport keeps the vertical field of view of the export, so a
  wider view shows more at the sides; `Renderer.overlay_rect` places the screen overlays
  (stats, mini-map, photos) inside the frame the video will show.
- **Animation.** At time *t* the animation evaluates all curves and pushes the values into the
  store; the viewport and the export use exactly the same path.
- **Tiles.** Everything is an XYZ Web Mercator pyramid. Imagery comes as tiles; DEM sources
  (GeoTIFF, COG, WMS, STAC) are baked into 16-bit PNG heightmap tiles in the same scheme.
  Cache layout `<cache>/<provider-id>/<z>/<x>/<y>.<ext>`, shared between projects.
  Regional providers are used inside their coverage and feathered into the worldwide ones.
- **World space.** A local East-North-Up frame (metres) at the project origin, or at the
  scene's own `origin`. Points go through ECEF, so the Earth's curvature is exact; rendering is
  camera-relative (float64 on the CPU, float32 offsets on the GPU) with logarithmic depth from
  0.5 m to thousands of kilometres.
- **Terrain LOD.** A quadtree of tiles selected by screen-space error, capped by the zone
  zoom; nodes stream in on background threads, morph from their parent's surface, and are
  evicted under a GPU memory budget (`render/lod.py`, `render/terrain.py`,
  `data/terrain_data.py`).
- **The journey axis.** The walked tracks in time order form one path
  (`render.progress.ProgressPath`): global distance <-> position <-> time. Progress (the drawn
  part of the line), the marker, the stats, labels that appear when passed, the follow camera
  and the sun from the GPX time all use it. `render.track_geometry.build_tracks` builds the
  geometry without a GPU.
- **Determinism.** The export waits for all terrain of a frame before drawing it; weather,
  lightning and noise are functions of the timeline time, never of the wall clock. Two renders
  of a scene match up to a fraction of a colour level (label heights depend on which
  heightmaps happen to be loaded when the labels are enriched).
- **Resolution independence.** Sizes on screen are given for 1080p and scale with the output
  height (`render/scale.py`).

## Scene scripts and the scene kit

Long videos are built by Python scripts rather than by hand: `earthling.scenekit` loads a
project's journey (`load_journey`, no GPU needed), plans smooth cameras (`follow_shots`,
`map_shots`, `plan_camera`, `key_shots`), draws the head along tracks (`head_keys_for`,
`travel_scene`), adds POIs, light and still seconds at both ends (`add_holds`) and saves
`scenes/<name>.json`. A project's own helper module adds what is specific to it (named
places, track numbering, the look of a series) and usually starts with
`from earthling.scenekit import *`; see `examples/showcase/tools/sc_common.py`.

The application offers the same generators (`app/generators.py`, the Animation menu: camera,
drawing the tracks, the sun, still seconds) as undoable keys, and the scene list with the POI
and photo editors places items at coordinates, the hiker, a click or a track point (`app/item_editor.py`,
`app/place_dialog.py`).

The workflow for a video: run the scene script, check stills
(`earthling stills <project> --scene scenes/x.json 0 5 12.5`), render
(`earthling render <project> --scene scenes/x.json --preset h264`), check the video
(`earthling qa <video>`).

## Checking that a refactor changes nothing

The scene scripts of the showcase (and of private projects) are deterministic: regenerate
their scenes before and after a change and compare the JSON files byte for byte; render a few
stills per scene with `earthling stills` before (twice: that is the run-to-run noise) and
after, and check that the differences stay within that noise. Unit tests
(`uv run pytest`, about 80 s) never touch the network; GL tests skip without OpenGL 4.3. The
slow suite (`uv run pytest -m slow`, about 10 minutes, also offline) chains whole workflows
(GPX to video), checks that large journeys and the most expensive settings stay within time
budgets, and measures how long the interface freezes (`helpers.longest_freeze`).
`tests/test_monkey.py` takes 1500 random steps through the interface (menus, buttons, keys,
the scene list, the inspector, the palette, value drags, undo) from a fixed seed: a failure
repeats, and its message lists the steps that led there.
