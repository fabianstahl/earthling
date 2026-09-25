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
    fetch.add_argument("--kind", choices=["imagery", "dem", "all"], default="all")
    fetch.add_argument("--max-zoom", type=int, default=None, help="limit the finest zoom level")
    return parser


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
    kinds = {"imagery", "dem"} if kind == "all" else {kind}
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
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "gui"
    if command == "gui":
        from earthling.app.main import run_gui

        return run_gui(getattr(args, "project", None), dev_mode=getattr(args, "dev", False))
    if command == "plan":
        return cmd_plan(args.project)
    if command == "fetch":
        return cmd_fetch(args.project, args.kind, args.max_zoom)
    return 1


if __name__ == "__main__":
    sys.exit(main())
