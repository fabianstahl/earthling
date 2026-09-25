# Earthling

Cinematic 3D flyover videos of GPS tracks: load GPX files, download satellite imagery and
elevation data, render the terrain with OpenGL shaders, animate everything on a keyframe
timeline and export 4K video.

See [ROADMAP.md](ROADMAP.md) for the plan.

## Setup

```powershell
python -m pip install --user uv   # if uv is not installed
uv sync
uv run earthling
```

## Development

```powershell
uv run pytest
uv run ruff check
```
