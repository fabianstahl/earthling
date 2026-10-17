"""Colour harmonisation of imagery against a consistent reference mosaic.

Worldwide imagery services often use a different mosaic per zoom range (other satellites,
seasons, sun glint on the sea). Where the terrain mixes levels, neighbouring tiles then differ
in colour and show as rectangles. With a colour reference (for example Sentinel-2 cloudless,
one mosaic for all zooms), every imagery texture keeps its own detail but takes its large-scale
colour from the reference::

    harmonised = imagery * blur(reference) / blur(imagery)

The gain lives on a grid of 16 x 16 blocks per terrain node. The block means of a node come
from the imagery that node actually shows (whatever its imagery zoom), and the blur uses the
neighbouring nodes as context, so neighbours of one level compute the same values at their
common border: no seams where the imagery zoom of the nodes changes (the edge of a download
zone). Means of tiles and nodes are cached.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image

from earthling.core.geo import TileKey
from earthling.data.cache import LruCache, TileCache
from earthling.data.imagery import decode_image, load_tile_rgb
from earthling.data.providers import TileProvider

BLOCKS = 4  # block means per 256 px imagery tile edge
NODE_BLOCKS_ZOOM = 4  # gain grid: 2^4 = 16 blocks per terrain node edge
CONTEXT = 5  # blocks of the neighbours around a node
SIGMA = 1.5  # blur radius in blocks
EPS = 3.0  # (0-255) keeps the gain stable in very dark areas
GAIN_RANGE = (0.15, 5.0)  # glinting sea (white ~200) down to ~30, a dark sea up to turquoise


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return k / k.sum()


def _blur(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised (weighted) separable Gaussian blur of an (h, w, c) grid."""
    k = _gaussian_kernel(sigma)

    def conv(a: np.ndarray, axis: int) -> np.ndarray:
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, a)

    num = conv(conv(values * weights[..., None], 0), 1)
    den = conv(conv(weights, 0), 1)[..., None]
    return num / np.maximum(den, 1e-6)


def _decode_small(data: bytes, size: int) -> np.ndarray:
    """Decode an image at about ``size`` pixels (JPEG: scaled while decoding, much faster)."""
    with Image.open(io.BytesIO(data)) as img:
        img.draft("RGB", (size, size))
        return np.asarray(img.convert("RGB"))


def _block_means(rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(rgb))
    return np.asarray(img.resize((BLOCKS, BLOCKS), Image.Resampling.BOX), dtype=np.float32)


class ColourReference:
    """Harmonises textures of ``source`` (the imagery) with the ``reference`` provider, whose
    tiles are used up to ``max_zoom`` (deeper tiles are crops of those)."""

    def __init__(
        self,
        cache: TileCache,
        source: TileProvider,
        reference: TileProvider,
        max_zoom: int = 12,
        strength: float = 1.0,
        ensure_reference: Callable[[int, int, int], object] | None = None,
        max_entries: int = 200_000,
    ) -> None:
        self.cache = cache
        self.source = source
        self.reference = reference
        self.max_zoom = min(max_zoom, reference.max_zoom)
        self.strength = strength
        self.ensure_reference = ensure_reference
        self._means = LruCache(max_entries)
        self._decoded = LruCache(4096)

    # --- block means ---------------------------------------------------------------------
    def source_means(self, z: int, x: int, y: int) -> np.ndarray | None:
        key = ("s", z, x, y)
        hit, value = self._means.lookup(key)
        if hit:
            return value
        data = self.cache.read(self.source.id, z, x, y, self.source.ext)
        if data is not None:  # means only: a small decode is enough
            return self._means.put(key, _block_means(_decode_small(data, 32)))
        found = load_tile_rgb(self.cache, self.source, z, x, y)  # an ancestor's crop
        return self._means.put(key, None if found is None else _block_means(found[0]))

    def _reference_tile(self, z: int, x: int, y: int, size: int) -> np.ndarray | None:
        """The reference tile decoded at about ``size`` pixels (enough for the means)."""
        key = (z, x, y, size)
        hit, value = self._decoded.lookup(key)
        if hit:
            return value
        if self.ensure_reference is not None:
            self.ensure_reference(z, x, y)
        image = None
        for up in range(0, 9):  # the tile itself or a crop of the nearest cached ancestor
            if z - up < 0:
                break
            data = self.cache.read(self.reference.id, z - up, x >> up, y >> up,
                                   self.reference.ext)  # fmt: skip
            if data is None:
                continue
            need = min(256, size << up)
            tile = _decode_small(data, need) if need < 256 else decode_image(data)
            n = 1 << up
            sub = max(1, tile.shape[0] // n)
            ox, oy = (x - ((x >> up) << up)) * sub, (y - ((y >> up) << up)) * sub
            image = tile[oy : oy + sub, ox : ox + sub]
            break
        return self._decoded.put(key, image)

    def reference_means(self, z: int, x: int, y: int) -> np.ndarray | None:
        key = ("r", z, x, y)
        hit, value = self._means.lookup(key)
        if hit:
            return value
        up = max(0, z - self.max_zoom)
        tile = self._reference_tile(z - up, x >> up, y >> up, min(256, 32 << up))
        if tile is None:
            return self._means.put(key, None)
        n = 1 << up
        size = tile.shape[0]
        sub = max(1, size // n)
        ox, oy = (x - ((x >> up) << up)) * size // n, (y - ((y >> up) << up)) * size // n
        crop = tile[oy : oy + sub, ox : ox + sub]
        return self._means.put(key, _block_means(crop))

    # --- per node blocks --------------------------------------------------------------
    def node_blocks(self, which: str, key: TileKey, block_zoom: int, image_zoom: int = 0):
        """(values (nb, nb, 3), weights (nb, nb)) of node ``key`` on the block grid of
        ``block_zoom`` (nb = 2^(block_zoom - z)): the source imagery as the node shows it (at
        ``image_zoom``, "s") or the reference ("r")."""
        cache_key = (which, *key, block_zoom, image_zoom if which == "s" else -1)
        hit, value = self._means.lookup(cache_key)
        if hit:
            return value
        z, x, y = key
        nb = 1 << (block_zoom - z)
        if which == "s":
            zoom, means_of = max(image_zoom, z), self.source_means
        else:  # tile means 4 x 4: tiles two levels above the blocks (never finer than needed)
            zoom, means_of = max(z, min(block_zoom - 2, z + 4)), self.reference_means
        d = zoom - z
        n = 1 << d
        grid = np.zeros((n * BLOCKS, n * BLOCKS, 3), dtype=np.float32)
        weight = np.zeros((n * BLOCKS, n * BLOCKS), dtype=np.float32)
        for j in range(n):
            for i in range(n):
                m = means_of(zoom, (x << d) + i, (y << d) + j)
                if m is None:
                    continue
                rows = slice(j * BLOCKS, (j + 1) * BLOCKS)
                cols = slice(i * BLOCKS, (i + 1) * BLOCKS)
                grid[rows, cols] = m
                weight[rows, cols] = 1.0
        k = grid.shape[0]
        if k > nb:  # average down
            f = k // nb
            w = weight.reshape(nb, f, nb, f).sum(axis=(1, 3))
            v = (grid * weight[..., None]).reshape(nb, f, nb, f, 3).sum(axis=(1, 3))
            grid, weight = v / np.maximum(w, 1e-6)[..., None], (w > 0).astype(np.float32)
        elif k < nb:  # repeat
            f = nb // k
            grid = np.repeat(np.repeat(grid, f, axis=0), f, axis=1)
            weight = np.repeat(np.repeat(weight, f, axis=0), f, axis=1)
        return self._means.put(cache_key, (grid, weight))

    # --- harmonisation -------------------------------------------------------------------
    def gain(self, z: int, x: int, y: int, zoom_of: Callable[[TileKey], int]):
        """(gain (h, w, 3) on the block grid of block_zoom(z), blocks per node edge): the
        node's blocks with CONTEXT blocks of its neighbours (same level) around them."""
        zb = block_zoom(z)
        nb = 1 << (zb - z)
        size = nb + 2 * CONTEXT
        grid_s = np.zeros((size, size, 3), dtype=np.float32)
        grid_r = np.zeros_like(grid_s)
        weight = np.zeros((size, size), dtype=np.float32)
        tiles = 1 << z
        b0x, b0y = x * nb - CONTEXT, y * nb - CONTEXT  # first block of the grid
        reach = (CONTEXT + nb - 1) // nb  # neighbour nodes needed on each side
        for oy in range(y - reach, y + reach + 1):
            if not 0 <= oy < tiles:
                continue
            for ox in range(x - reach, x + reach + 1):
                # the part of this owner's blocks inside the grid
                gx0, gy0 = ox * nb - b0x, oy * nb - b0y
                x_lo, x_hi = max(0, gx0), min(size, gx0 + nb)
                y_lo, y_hi = max(0, gy0), min(size, gy0 + nb)
                if x_lo >= x_hi or y_lo >= y_hi:
                    continue
                owner = (z, ox % tiles, oy)
                s, ws = self.node_blocks("s", owner, zb, zoom_of(owner))
                r, wr = self.node_blocks("r", owner, zb)
                src = (slice(y_lo - gy0, y_hi - gy0), slice(x_lo - gx0, x_hi - gx0))
                dst = (slice(y_lo, y_hi), slice(x_lo, x_hi))
                grid_s[dst] = s[src]
                grid_r[dst] = r[src]
                weight[dst] = ws[src] * wr[src]
        if not weight[CONTEXT:-CONTEXT, CONTEXT:-CONTEXT].any():
            return None
        blur_s = _blur(grid_s, weight, SIGMA)
        blur_r = _blur(grid_r, weight, SIGMA)
        gain = (blur_r + EPS) / (blur_s + EPS)
        return np.clip(gain, *GAIN_RANGE).astype(np.float32), nb

    def apply(self, image: np.ndarray, z: int, x: int, y: int,
              zoom_of: Callable[[TileKey], int]) -> np.ndarray:  # fmt: skip
        """``image``: the texture of node (z, x, y) (uint8 RGB, any size); ``zoom_of(key)``:
        the imagery zoom a node's texture is made of."""
        if self.strength <= 0.0:
            return image
        found = self.gain(z, x, y, zoom_of)
        if found is None:
            return image
        gain, nb = found
        h, w = image.shape[:2]
        # bilinear between block centres (PIL maps pixel centres the same way), one block of
        # the neighbours around the node, then the node's part
        g = gain[CONTEXT - 1 : CONTEXT + nb + 1, CONTEXT - 1 : CONTEXT + nb + 1]
        bx, by = w / nb, h / nb
        size = (round((nb + 2) * bx), round((nb + 2) * by))
        ox, oy = round(bx), round(by)
        field = np.dstack([
            np.asarray(Image.fromarray(np.ascontiguousarray(g[..., c])).resize(
                size, Image.Resampling.BILINEAR))[oy : oy + h, ox : ox + w]
            for c in range(3)
        ])  # fmt: skip
        if self.strength != 1.0:
            field = 1.0 + (field - 1.0) * self.strength
        out = image.astype(np.float32)
        out *= field
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)


def block_zoom(z: int) -> int:
    """Zoom of the gain grid for terrain nodes of level z (16 blocks per node edge).

    (A grid shared by several levels was tried: it removed no seams between levels of detail
    and coarsened the correction of the nearest nodes.)"""
    return z + NODE_BLOCKS_ZOOM
