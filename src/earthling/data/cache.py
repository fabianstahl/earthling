"""On-disk tile cache: ``<root>/<provider-id>/<z>/<x>/<y>.<ext>``.

Tiles known to be unavailable (HTTP 404, placeholder images) are recorded as empty
``<y>.missing`` marker files so they are not requested again.
"""

from __future__ import annotations

import os
from pathlib import Path


class TileCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, provider_id: str, z: int, x: int, y: int, ext: str) -> Path:
        return self.root / provider_id / str(z) / str(x) / f"{y}.{ext}"

    def missing_path(self, provider_id: str, z: int, x: int, y: int) -> Path:
        return self.root / provider_id / str(z) / str(x) / f"{y}.missing"

    def has(self, provider_id: str, z: int, x: int, y: int, ext: str) -> bool:
        return self.path(provider_id, z, x, y, ext).exists()

    def is_missing(self, provider_id: str, z: int, x: int, y: int) -> bool:
        return self.missing_path(provider_id, z, x, y).exists()

    def is_known(self, provider_id: str, z: int, x: int, y: int, ext: str) -> bool:
        """True if the tile is cached or known to be unavailable."""
        return self.has(provider_id, z, x, y, ext) or self.is_missing(provider_id, z, x, y)

    def read(self, provider_id: str, z: int, x: int, y: int, ext: str) -> bytes | None:
        path = self.path(provider_id, z, x, y, ext)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def write(self, provider_id: str, z: int, x: int, y: int, ext: str, data: bytes) -> Path:
        path = self.path(provider_id, z, x, y, ext)
        write_atomic(path, data)
        return path

    def mark_missing(self, provider_id: str, z: int, x: int, y: int) -> None:
        path = self.missing_path(provider_id, z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file so interrupted downloads never leave partial tiles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
