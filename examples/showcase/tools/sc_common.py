"""Showcase specifics on top of :mod:`earthling.scenekit` (the camera, scene and journey helpers
the scene scripts use): where the scene projects live, English stats.

Run the scene scripts with ``python -m uv run python examples/showcase/tools/<script>.py``
from the repository; each writes ``<project>/scenes/<name>.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from earthling import scenekit
from earthling.scenekit import *  # noqa: F403 (the kit is the scene scripts' vocabulary)
from earthling.scenekit import Journey

TOOLS = Path(__file__).resolve().parent
SHOWCASE = TOOLS.parent
REPO = SHOWCASE.parent.parent
RENDERS = REPO / "renders" / "showcase"
sys.path.insert(0, str(TOOLS))

Hike = Journey


def load_hike(project: str, origin: tuple[float, float] | None = None) -> Journey:
    """Scene project ``project`` (folder name in examples/showcase), with the local frame centred
    on ``origin`` (lat, lon) when given (the scene gets the same origin: :func:`new_scene`)."""
    return scenekit.load_journey(SHOWCASE / project, origin)


def new_scene(hike: Journey, duration: float, fps: float = 60.0):
    scene = scenekit.new_scene(hike, duration, fps)
    scene.store.set("stats.language", "en")
    return scene
