"""Composing imagery textures for terrain tiles from cached XYZ tiles."""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image

from earthling.data.cache import TileCache
from earthling.data.providers import TileProvider

MAX_FALLBACK_LEVELS = 8


def decode_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


def load_tile_rgb(
    cache: TileCache, provider: TileProvider, z: int, x: int, y: int
) -> tuple[np.ndarray, int] | None:
    """Tile (z, x, y) as RGB array, falling back to (a crop of) cached ancestors.

    Returns the image and how many levels had to be climbed (0 = exact tile).
    """
    size = provider.tile_size
    for up in range(MAX_FALLBACK_LEVELS + 1):
        pz = z - up
        if pz < 0:
            break
        px, py = x >> up, y >> up
        data = cache.read(provider.id, pz, px, py, provider.ext)
        if data is None:
            continue
        img = decode_image(data)
        if up == 0:
            return img, 0
        n = 1 << up
        sub = size // n
        if sub < 1:
            break
        ox = (x - (px << up)) * sub
        oy = (y - (py << up)) * sub
        crop = Image.fromarray(img[oy : oy + sub, ox : ox + sub])
        return np.asarray(crop.resize((size, size), Image.Resampling.BICUBIC)), up
    return None


def compose_tile_texture(
    cache: TileCache,
    provider: TileProvider,
    z: int,
    x: int,
    y: int,
    image_zoom: int,
    ensure: Callable[[int, int, int], object] | None = None,
) -> np.ndarray | None:
    """RGB texture covering terrain tile (z, x, y) built from imagery tiles at ``image_zoom``.

    The result has (tile_size * 2**(image_zoom - z))^2 pixels. Returns None if no imagery at all
    (including ancestors) is available. ``ensure(z, x, y)`` is called for every wanted imagery
    tile first, e.g. to download missing tiles on demand.
    """
    image_zoom = min(max(image_zoom, z), provider.max_zoom)
    d = image_zoom - z
    if d < 0:
        # Terrain finer than imagery: crop from the ancestor.
        found = load_tile_rgb(cache, provider, z, x, y)
        return None if found is None else found[0]
    n = 1 << d
    size = provider.tile_size
    out = np.zeros((size * n, size * n, 3), dtype=np.uint8)
    any_found = False
    for j in range(n):
        for i in range(n):
            if ensure is not None:
                ensure(image_zoom, (x << d) + i, (y << d) + j)
            found = load_tile_rgb(cache, provider, image_zoom, (x << d) + i, (y << d) + j)
            if found is None:
                continue
            any_found = True
            out[j * size : (j + 1) * size, i * size : (i + 1) * size] = found[0]
    return out if any_found else None
