"""CPU-side terrain data access for the LOD renderer (heightmaps and imagery per node)."""

from __future__ import annotations

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
    ) -> None:
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

    def imagery_for(self, key: TileKey) -> np.ndarray | None:
        z, x, y = key
        return compose_tile_texture(self.cache, self.imagery, z, x, y, self.imagery_zoom_for(key))
