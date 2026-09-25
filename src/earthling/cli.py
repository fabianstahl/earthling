"""Command line entry point."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earthling", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="start the graphical application (default)")
    gui.add_argument("project", nargs="?", help="project folder to open")
    gui.add_argument("--dev", action="store_true", help="enable shader hot reload")
    plan = sub.add_parser("plan", help="show which tiles a project needs and how large they are")
    plan.add_argument("project", help="project folder")
    fetch = sub.add_parser("fetch", help="download the tiles of a project into the cache")
    fetch.add_argument("project", help="project folder")
    fetch.add_argument("--kind", choices=["imagery", "dem", "topo", "labels", "all"], default="all")
    fetch.add_argument("--max-zoom", type=int, default=None, help="limit the finest zoom level")
    render = sub.add_parser("render", help="render the animation of a scene to a video file")
    render.add_argument("project", help="project folder")
    render.add_argument("--scene", default="scene.json", help="scene file (in the project folder)")
    render.add_argument("--out", default=None, help="output file (default: <scene>.<ext>)")
    render.add_argument("--preset", default="prores4444", help="video preset, see --list-presets")
    render.add_argument("--resolution", default=None, help="e.g. 3840x2160 (default: scene)")
    render.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    render.add_argument("--end", type=float, default=None, help="end time (default: duration)")
    render.add_argument("--list-presets", action="store_true", help="list presets and exit")
    check = sub.add_parser(
        "check-source", help="test an imagery provider or DEM source at a location"
    )
    check.add_argument("project", help="project folder (for its own providers and the cache)")
    check.add_argument("source", help="provider / DEM source id, e.g. ign_bdortho")
    check.add_argument("--at", default=None, help="lon,lat (default: first track point or the "
                       "middle of the coverage)")  # fmt: skip
    check.add_argument("--out", default=None, help="folder for a contact sheet of the tiles")
    return parser


def cmd_check_source(args) -> int:
    from pathlib import Path

    from earthling.core.config import ConfigError, Project
    from earthling.core.session import Session
    from earthling.data.check import check_dem, check_imagery
    from earthling.data.dem import DEM_SOURCES
    from earthling.data.providers import PROVIDERS, TileProvider

    try:
        session = Session(Project.load(args.project))
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    source = PROVIDERS.get(args.source) or DEM_SOURCES.get(args.source)
    if source is None:
        known = ", ".join(sorted([*PROVIDERS, *DEM_SOURCES]))
        print(f"unknown source '{args.source}' (known: {known})", file=sys.stderr)
        return 2
    if args.at:
        lon, lat = (float(v) for v in args.at.split(","))
    else:
        lon, lat = _default_location(session, source)
    if isinstance(source, TileProvider):
        out = Path(args.out) if args.out else None
        report = check_imagery(source, lon, lat, out_dir=out)
    else:
        report = check_dem(source, lon, lat, session.cache)
    print(report.text())
    return 0 if report.ok else 1


def _default_location(session, source) -> tuple[float, float]:
    """The track point farthest inside the source's coverage (or the first track point)."""
    import numpy as np
    import shapely

    coverage = source.coverage_area
    points = [np.column_stack([s.lon, s.lat]) for t in session.tracks for s in t.segments]
    points = np.concatenate(points) if points else np.zeros((0, 2))
    if coverage is None:
        return (float(points[0, 0]), float(points[0, 1])) if len(points) else (7.0, 46.0)
    points = points[:: max(1, len(points) // 5000)]
    inside = points[shapely.contains_xy(coverage.lonlat, points[:, 0], points[:, 1])]
    if len(inside):
        depth = shapely.distance(coverage.lonlat.boundary, shapely.points(inside))
        lon, lat = inside[int(np.argmax(depth))]
        return float(lon), float(lat)
    point = coverage.lonlat.representative_point()
    return point.x, point.y


def cmd_render(args) -> int:
    import time
    from pathlib import Path

    import moderngl

    from earthling.core.scene import Scene
    from earthling.export.frames import FrameRenderer, parse_resolution
    from earthling.export.video import PRESETS, available_presets, export_video
    from earthling.render.renderer import Renderer

    if args.list_presets:
        for preset in available_presets():
            print(f"{preset.id:12s} {preset.label}")
        return 0
    if args.preset not in PRESETS:
        print(f"unknown preset '{args.preset}'", file=sys.stderr)
        return 2
    session = _load_session(args.project)
    if session is None:
        return 1
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = session.project.folder / scene_path
    scene = Scene()
    for problem in scene.load(scene_path):
        print(f"warning: {problem}", file=sys.stderr)
    preset = PRESETS[args.preset]
    width, height = parse_resolution(args.resolution or scene.store["export.resolution"])
    anim = scene.animation
    end = anim.duration if args.end is None else args.end
    out = Path(args.out) if args.out else scene_path.with_suffix(preset.extension)
    ctx = moderngl.create_standalone_context(require=430)
    renderer = Renderer(ctx)
    renderer.set_scene(session.frame, session.tracks)
    renderer.set_terrain_source(session.terrain_data(), session.terrain_nodes())
    renderer.store = scene.store
    renderer.timezone = session.config.project.timezone
    renderer.set_labels(session.labels, session)
    frames = FrameRenderer(renderer, anim)
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        elapsed = time.monotonic() - started
        eta = elapsed / done * (total - done)
        print(f"\r  frame {done}/{total}  {done / elapsed:5.2f} fps  ETA {eta / 60:5.1f} min",
              end="", flush=True)  # fmt: skip

    print(f"rendering {scene_path.name} {args.start:g}-{end:g} s at {width}x{height} "
          f"{anim.fps:g} fps -> {out} ({preset.label})")  # fmt: skip
    try:
        count = export_video(
            frames, out, width, height, anim.fps, preset, args.start, end, progress
        )
    except KeyboardInterrupt:
        print("\ncancelled")
        return 130
    finally:
        renderer.terrain.shutdown()
    print(f"\nwrote {count} frames to {out}")
    return 0


def _load_session(project_dir: str):
    from earthling.core.config import ConfigError, Project
    from earthling.core.session import Session

    try:
        session = Session(Project.load(project_dir))
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return None
    for err in session.load_errors:
        print(f"warning: {err}", file=sys.stderr)
    if session.aoi is None:
        print("no tracks found", file=sys.stderr)
        return None
    return session


def _progress_printer():
    import time

    last = [0.0]

    def report(p) -> None:
        now = time.monotonic()
        if now - last[0] > 0.5 or p.finished:
            last[0] = now
            print(
                f"\r  {p.done}/{p.total} tiles, {p.downloaded} new, {p.missing} unavailable, "
                f"{p.failed} failed, {p.bytes / 1e6:.1f} MB",
                end="",
                flush=True,
            )

    return report


def cmd_fetch(project_dir: str, kind: str, max_zoom: int | None) -> int:
    from earthling.data.jobs import download_jobs

    session = _load_session(project_dir)
    if session is None:
        return 1
    kinds = {"imagery", "dem", "labels"} if kind == "all" else {kind}
    status = 0
    print(f"cache: {session.cache.root}")
    for job in download_jobs(session, kinds, max_zoom):
        print(job.name)
        try:
            result = job.run(_progress_printer())
        except KeyboardInterrupt:
            job.cancel()
            print("\ncancelled")
            return 130
        print()
        for err in result.errors[:10]:
            print(f"  error: {err}", file=sys.stderr)
        status = 1 if result.failed else status
    return status


def cmd_plan(project_dir: str) -> int:
    from earthling.core.aoi import plan_report

    session = _load_session(project_dir)
    if session is None:
        return 1
    bounds = tuple(round(v, 4) for v in session.aoi.bounds)
    print(f"{len(session.tracks)} track(s), AOI bounds {bounds}")
    print(plan_report(session.plan))
    print(session.provider_report())
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # e.g. cp1252 consoles: never crash on output
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    command = args.command or "gui"
    if command == "gui":
        from earthling.app.main import run_gui

        return run_gui(getattr(args, "project", None), dev_mode=getattr(args, "dev", False))
    if command == "plan":
        return cmd_plan(args.project)
    if command == "fetch":
        return cmd_fetch(args.project, args.kind, args.max_zoom)
    if command == "render":
        return cmd_render(args)
    if command == "check-source":
        return cmd_check_source(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
