"""CPU-side terrain data access for the LOD renderer (heightmaps and imagery per node)."""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from earthling.core.aoi import TilePlan
from earthling.data.cache import TileCache
from earthling.data.dem import GRID_INTERVALS, read_heightmap
from earthling.data.imagery import compose_tile_texture
from earthling.data.providers import TileProvider

TileKey = tuple[int, int, int]
TEXEL_ZOOM_OFFSET = 2  # node imagery = 4x4 tiles at z + 2 (1024 px)


@dataclass
class HeightmapRef:
    """A node's view into (an ancestor's) heightmap.

    Sample index ``s`` of the node (0..256 across the node) maps to sample index
    ``offset + s * scale / 256`` of ``source``'s heightmap.
    """

    source: TileKey
    heights: np.ndarray
    scale: float  # in source sample units across the node (256 / 2**levels_up)
    offset: tuple[float, float]


class TerrainData:
    def __init__(
        self,
        cache: TileCache,
        dem_id: str,
        imagery: TileProvider,
        plan: TilePlan,
        max_heightmaps: int = 512,
        downloader=None,
    ) -> None:
        """``downloader``: optional TileDownloader for on-demand fetching of missing imagery."""
        self.cache = cache
        self.dem_id = dem_id
        self.imagery = imagery
        self.plan = plan
        self._imagery_sets = {
            z: {(int(x), int(y)) for x, y in t} for z, t in plan.levels.get("imagery", {}).items()
        }
        self._heightmaps: OrderedDict[TileKey, np.ndarray | None] = OrderedDict()
        self._max = max_heightmaps
        self._lock = threading.Lock()
        self.downloader = downloader
        self.on_demand = downloader is not None

    def _heightmap(self, key: TileKey) -> np.ndarray | None:
        with self._lock:
            if key in self._heightmaps:
                self._heightmaps.move_to_end(key)
                return self._heightmaps[key]
        z, x, y = key
        heights = None
        if not self.cache.is_missing(self.dem_id, z, x, y):
            heights = read_heightmap(self.cache, self.dem_id, z, x, y)
        with self._lock:
            self._heightmaps[key] = heights
            while len(self._heightmaps) > self._max:
                self._heightmaps.popitem(last=False)
        return heights

    def heightmap_for(self, key: TileKey) -> HeightmapRef | None:
        z, x, y = key
        for up in range(z + 1):
            src = (z - up, x >> up, y >> up)
            heights = self._heightmap(src)
            if heights is None:
                continue
            n = 1 << up
            scale = GRID_INTERVALS / n
            offset = ((x - (src[1] << up)) * scale, (y - (src[2] << up)) * scale)
            return HeightmapRef(src, heights, scale, offset)
        return None

    def imagery_zoom_for(self, key: TileKey) -> int:
        z, x, y = key
        for d in range(TEXEL_ZOOM_OFFSET, -1, -1):
            zi = z + d
            if zi > self.imagery.max_zoom:
                continue
            tiles = self._imagery_sets.get(zi)
            if tiles and ((x << d), (y << d)) in tiles:
                return zi
        return min(z + TEXEL_ZOOM_OFFSET, self.imagery.max_zoom)

    def _ensure_imagery(self, z: int, x: int, y: int) -> None:
        if (
            self.on_demand
            and self.downloader is not None
            and (x, y) in self._imagery_sets.get(z, ())
        ):
            # network problems must never break rendering
            with contextlib.suppress(Exception):
                self.downloader.fetch_one(z, x, y)

    def imagery_for(self, key: TileKey) -> np.ndarray | None:
        z, x, y = key
        return compose_tile_texture(
            self.cache, self.imagery, z, x, y, self.imagery_zoom_for(key), self._ensure_imagery
        )

    # --- height queries (CPU) ------------------------------------------------------------
    def height_at(self, lon: float, lat: float, z: int | None = None) -> float | None:
        """Terrain height (m) at lon/lat from the finest available heightmap, or None."""
        from earthling.core.geo import lonlat_to_tile

        if z is None:
            dem_levels = self.plan.levels.get("dem", {})
            z = max(dem_levels) if dem_levels else 12
        tx, ty = lonlat_to_tile(lon, lat, z)
        x, y = int(np.floor(tx)), int(np.floor(ty))
        ref = self.heightmap_for((z, x, y))
        if ref is None:
            return None
        # position in source sample units
        sx = ref.offset[0] + (float(tx) - x) * ref.scale + 1.0
        sy = ref.offset[1] + (float(ty) - y) * ref.scale + 1.0
        h = ref.heights
        x0 = int(np.clip(np.floor(sx), 0, h.shape[1] - 2))
        y0 = int(np.clip(np.floor(sy), 0, h.shape[0] - 2))
        fx, fy = sx - x0, sy - y0
        top = h[y0, x0] * (1 - fx) + h[y0, x0 + 1] * fx
        bottom = h[y0 + 1, x0] * (1 - fx) + h[y0 + 1, x0 + 1] * fx
        value = float(top * (1 - fy) + bottom * fy)
        return value if np.isfinite(value) else None
