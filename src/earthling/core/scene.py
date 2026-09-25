"""The editable scene document: property values, camera (and later keyframes, ...).

Scenes are stored as JSON (``scene.json`` in the project folder by default)::

    {"version": 1, "properties": {...}, "camera": {...}, "ui": {...}}

``MIGRATIONS`` upgrade older files step by step to :data:`SCENE_VERSION`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from earthling.core.properties import PropertyStore

SCENE_VERSION = 1
DEFAULT_SCENE_NAME = "scene.json"

# version -> function upgrading a document of that version to version + 1
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


class SceneError(Exception):
    pass


def migrate(doc: dict[str, Any]) -> dict[str, Any]:
    version = int(doc.get("version", 0))
    if version > SCENE_VERSION:
        raise SceneError(f"scene file version {version} is newer than supported ({SCENE_VERSION})")
    while version < SCENE_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise SceneError(f"no migration from scene version {version}")
        doc = step(doc)
        version += 1
        doc["version"] = version
    return doc


class Scene:
    def __init__(self) -> None:
        from earthling.render.parameters import build_registry

        self.registry = build_registry()
        self.store = PropertyStore(self.registry)
        self.camera: dict[str, Any] = {}  # serialised camera state (owned by the viewport)
        self.ui: dict[str, Any] = {}  # window/dock layout (owned by the main window)
        self.path: Path | None = None
        self.dirty = False
        self._dirty_listeners: list[Callable[[bool], None]] = []
        self.store.subscribe(lambda pid, value: self.mark_dirty())

    # --- dirty state ------------------------------------------------------------------
    def mark_dirty(self, dirty: bool = True) -> None:
        if dirty != self.dirty:
            self.dirty = dirty
            for listener in list(self._dirty_listeners):
                listener(dirty)

    def on_dirty_changed(self, listener: Callable[[bool], None]) -> None:
        self._dirty_listeners.append(listener)

    # --- (de)serialisation ------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "version": SCENE_VERSION,
            "properties": self.store.to_json(),
            "camera": self.camera,
            "ui": self.ui,
        }

    def load_json(self, doc: dict[str, Any]) -> list[str]:
        """Replace the scene contents. Returns non-fatal problems (unknown properties...)."""
        doc = migrate(dict(doc))
        for d in self.registry:  # properties missing in the file get their defaults
            self.store.set(d.id, d.default)
        problems = self.store.load_json(doc.get("properties", {}))
        self.camera = dict(doc.get("camera", {}))
        self.ui = dict(doc.get("ui", {}))
        return problems

    def reset(self, path: Path | None = None) -> None:
        """Back to defaults (a new, unsaved scene that will be saved to ``path``)."""
        self.load_json({"version": SCENE_VERSION})
        self.path = Path(path) if path is not None else None
        self.mark_dirty(False)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.path or DEFAULT_SCENE_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        tmp.replace(path)
        self.path = path
        self.mark_dirty(False)
        return path

    def load(self, path: Path) -> list[str]:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneError(f"cannot read scene {path}: {exc}") from exc
        if not isinstance(doc, dict):
            raise SceneError(f"{path} is not a scene file")
        problems = self.load_json(doc)
        self.path = Path(path)
        self.mark_dirty(False)
        return problems
