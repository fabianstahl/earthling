"""Loads GLSL programs from disk, resolves ``#include`` and supports hot reload.

A program named ``foo`` consists of ``foo.vert`` and ``foo.frag`` (optionally ``foo.geom``)
inside the shader directory. Sources may contain ``#include "file.glsl"`` lines which are
resolved relative to the shader directory. Every file a program depends on is tracked, so
that a change to any of them triggers a recompile of that program.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import moderngl

log = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).parent / "shaders"
GLSL_VERSION = "#version 430 core"

_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)


class ShaderError(RuntimeError):
    pass


def preprocess(
    source: str,
    base_dir: Path,
    defines: dict[str, object] | None = None,
    deps: set[Path] | None = None,
    _stack: tuple[Path, ...] = (),
    virtual: dict[str, str] | None = None,
) -> str:
    """Resolve includes recursively and prepend the version line and defines.

    ``virtual`` maps include names to generated sources (checked before the file system).
    """

    def resolve(text: str, stack: tuple[Path, ...]) -> str:
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if virtual and name in virtual:
                marker = Path(f"<virtual>/{name}")
                if marker in stack:
                    raise ShaderError(f"recursive include of {name}")
                return resolve(virtual[name], (*stack, marker))
            path = (base_dir / name).resolve()
            if path in stack:
                raise ShaderError(f"recursive include of {path.name}")
            if not path.exists():
                raise ShaderError(f"include not found: {match.group(1)}")
            if deps is not None:
                deps.add(path)
            return resolve(path.read_text(encoding="utf-8"), (*stack, path))

        return _INCLUDE_RE.sub(repl, text)

    body = resolve(source, _stack)
    header = [GLSL_VERSION]
    for key, value in (defines or {}).items():
        header.append(f"#define {key} {value}")
    return "\n".join(header) + "\n#line 1\n" + body


@dataclass
class _Entry:
    name: str
    defines: dict[str, object]
    vertex: str | None = None  # use <vertex>.vert instead of <name>.vert
    program: moderngl.Program | None = None
    deps: set[Path] = field(default_factory=set)
    listeners: list[Callable[[moderngl.Program], None]] = field(default_factory=list)


class ShaderLibrary:
    def __init__(self, ctx: moderngl.Context, shader_dir: Path = SHADER_DIR) -> None:
        self.ctx = ctx
        self.shader_dir = shader_dir
        self._entries: dict[tuple, _Entry] = {}
        self.virtual: dict[str, str] = {}

    def register_virtual(self, name: str, source: str) -> None:
        """Provide a generated include file (e.g. layer dispatch code)."""
        self.virtual[name] = source

    def get(
        self,
        name: str,
        defines: dict[str, object] | None = None,
        on_reload: Callable[[moderngl.Program], None] | None = None,
        vertex: str | None = None,
    ) -> moderngl.Program:
        """``vertex``: share another program's vertex shader (e.g. "fullscreen")."""
        defines = dict(defines or {})
        key = (name, vertex, tuple(sorted(defines.items())))
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(name, defines, vertex)
            entry.program = self._compile(entry)
            self._entries[key] = entry
        if on_reload is not None:
            entry.listeners.append(on_reload)
        assert entry.program is not None
        return entry.program

    def compute(self, name: str) -> moderngl.ComputeShader:
        """A compute shader from ``<name>.comp`` (compiled once)."""
        cache = self.__dict__.setdefault("_compute", {})
        shader = cache.get(name)
        if shader is None:
            path = self.shader_dir / f"{name}.comp"
            source = preprocess(
                path.read_text(encoding="utf-8"), self.shader_dir, None, set(), virtual=self.virtual
            )
            try:
                shader = self.ctx.compute_shader(source)
            except moderngl.Error as exc:
                raise ShaderError(f"{name}: {exc}") from exc
            cache[name] = shader
        return shader

    def _compile(self, entry: _Entry) -> moderngl.Program:
        deps: set[Path] = set()
        stages: dict[str, str] = {}
        for stage, ext in (
            ("vertex_shader", "vert"),
            ("fragment_shader", "frag"),
            ("geometry_shader", "geom"),
        ):
            base = entry.vertex if (ext == "vert" and entry.vertex) else entry.name
            path = self.shader_dir / f"{base}.{ext}"
            if not path.exists():
                continue
            deps.add(path.resolve())
            stages[stage] = preprocess(
                path.read_text(encoding="utf-8"),
                self.shader_dir,
                entry.defines,
                deps,
                virtual=self.virtual,
            )
        if "vertex_shader" not in stages:
            raise ShaderError(f"shader program '{entry.name}' has no vertex shader")
        try:
            program = self.ctx.program(**stages)
        except moderngl.Error as exc:
            raise ShaderError(f"{entry.name}: {exc}") from exc
        entry.deps = deps
        return program

    def watched_files(self) -> set[Path]:
        files: set[Path] = set()
        for entry in self._entries.values():
            files |= entry.deps
        return files

    def reload_changed(self, changed: Path) -> list[str]:
        """Recompile all programs depending on ``changed``. Returns error messages."""
        changed = changed.resolve()
        errors: list[str] = []
        for entry in self._entries.values():
            if changed not in entry.deps:
                continue
            try:
                program = self._compile(entry)
            except ShaderError as exc:
                log.error("shader reload failed: %s", exc)
                errors.append(str(exc))
                continue
            old = entry.program
            entry.program = program
            for listener in entry.listeners:
                listener(program)
            if old is not None:
                old.release()
            log.info("reloaded shader %s", entry.name)
        return errors
