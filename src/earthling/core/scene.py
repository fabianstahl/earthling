"""The editable scene document: property values (and later keyframes, camera paths, ...)."""

from __future__ import annotations

from earthling.core.properties import PropertyStore


class Scene:
    def __init__(self) -> None:
        from earthling.render.parameters import build_registry

        self.registry = build_registry()
        self.store = PropertyStore(self.registry)
