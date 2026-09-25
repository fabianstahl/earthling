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
    return parser


def cmd_plan(project_dir: str) -> int:
    from earthling.core.aoi import compute_aoi, plan_report, plan_tiles
    from earthling.core.config import ConfigError, Project
    from earthling.core.gpx import load_gpx_folder

    try:
        project = Project.load(project_dir)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    tracks, errors = load_gpx_folder(project.gpx_dir)
    for err in errors:
        print(f"warning: {err}", file=sys.stderr)
    aoi = compute_aoi(tracks, project.config.area)
    if aoi is None:
        print("no tracks found", file=sys.stderr)
        return 1
    print(f"{len(tracks)} track(s), AOI bounds {tuple(round(v, 4) for v in aoi.bounds)}")
    print(plan_report(plan_tiles(aoi, project.config.area)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "gui"
    if command == "gui":
        from earthling.app.main import run_gui

        return run_gui(getattr(args, "project", None), dev_mode=getattr(args, "dev", False))
    if command == "plan":
        return cmd_plan(args.project)
    return 1


if __name__ == "__main__":
    sys.exit(main())
