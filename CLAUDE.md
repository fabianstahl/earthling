# Earthling – notes for coding agents

Cinematic 3D flyover videos of GPS tracks: Python 3.12, PyQt6, moderngl (OpenGL 4.3), numpy.
Start with [docs/architecture.md](docs/architecture.md) (packages, layers, how a frame is
made) and [docs/configuration.md](docs/configuration.md) (`earthling.toml`).

## Commands

```powershell
uv run pytest                      # ~70 s, no network; GL tests skip without OpenGL 4.3
uv run ruff check; uv run ruff format
uv run earthling gui examples/alps_demo
uv run earthling render|stills|qa|plan|fetch|check-source ...   # see `earthling -h`
```

Where `uv` is not on PATH (e.g. Git Bash), use `.venv/Scripts/python.exe -m pytest`,
`.venv/Scripts/ruff.exe` and `.venv/Scripts/earthling.exe`.

## Where things go

- A new adjustable value: one `PropertyDef` in `render/parameters.py` (`flt`, `enum`, ... from
  `core.properties`); `uniform="u_name"` binds it to shaders automatically.
- GL boilerplate: `render/gl_util.py` (`set_uniform(s)`, `DynamicBuffer`, `draw_overlay`).
- Easing / interpolation: `core/easing.py`. Geodesy: `core/geo.py`.
- Helpers for scene scripts: `earthling.scenekit`; project-specific bits stay in the
  project's own helper module. The application's Animation menu (`app/generators.py`) calls
  the same kit functions, so a new generator belongs in the kit plus a small dialog.
- POI kinds: `core/poi_library.py`; built-in icons: `tools/make_poi_icons.py` (writes
  `assets/icons/`). Editors of placed items (POIs, photos) derive from `app/item_editor.py`;
  the scene list (`app/scene_list.py`) selects them and the inspector shows their form.
- `core` imports neither `data` nor `render` (except `core.scene` for the property registry);
  `session.py` sits above `data`.

## Conventions

- One roadmap step per commit, message `step X.Y: <summary>` (clean-ups `cleanup N: ...`);
  every commit passes pytest and ruff. Update README/ROADMAP for user-visible features.
- Line length 100; hand-aligned tables and argument lists keep `# fmt: skip`.
- LF line endings (`.gitattributes`); write files with `newline="\n"` / bytes on Windows.
- Comments: short, explain why. On-screen text is English or German (`stats.language`).
- Tests never touch the network (mock HTTP, fixtures in `tests/fixtures/`, shared helpers in
  `tests/helpers.py`).

## Compatibility

Private video projects outside the repo have scene scripts that use the package API:
`earthling.scenekit`, `earthling.core.session` (alias of `earthling.session`), names
re-exported by `render.tracks`, `render.hud` and `render.labels`,
`Session.track_group_properties()`, `TerrainData.height_at/heights_at`, `ProgressPath`
methods. Keep these working (or keep an alias) when moving things. Never commit personal GPX
files, photos or project scripts; `renders/` is git-ignored.

## Verifying "no visible change"

Scene scripts are deterministic: regenerate the showcase scenes
(`python examples/showcase/tools/sNN_*.py`) before and after a change and compare the JSON
files; render stills (`earthling stills ... --out DIR`) before and after and compare pixels
(twice before, to know the run-to-run noise).

## Known pitfalls

- Colour properties reach the shaders as sRGB (`bind_uniforms`); a shader linearises them
  once. Colours set in Python (`c**2.2`) are passed linear.
- shapely/GEOS predicates on one prepared geometry are not thread-safe: `Coverage` serialises
  them (label enrichment and the terrain workers share coverage geometries).
- Two renders of a scene can differ by a fraction of a colour level (label heights depend on
  which heightmaps are loaded); compare stills against the run-to-run noise, not bit by bit.
- The viewport defers GL work until `initializeGL`; setters store data and `_upload()` it.
