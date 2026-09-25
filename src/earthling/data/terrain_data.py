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
from earthling.data.downloader import TileDownloader
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
    """Heightmaps and textures for LOD nodes.

    ``sources`` maps tile-source names used by texture layers ("imagery", "topo", ...) to
    providers. All texture sources use the imagery zones of the tile plan (capped at each
    provider's maximum zoom); missing tiles can be downloaded on demand from worker threads.
    """

    def __init__(
        self,
        cache: TileCache,
        dem_id: str,
        sources: dict[str, TileProvider] | TileProvider,
        plan: TilePlan,
        max_heightmaps: int = 512,
        on_demand: bool = False,
        borders=None,
    ) -> None:
        """``borders``: optional BorderData providing the "borders" texture source."""
        if isinstance(sources, TileProvider):
            sources = {"imagery": sources}
        self.cache = cache
        self.dem_id = dem_id
        self.sources = dict(sources)
        self.plan = plan
        self._imagery_sets = {
            z: {(int(x), int(y)) for x, y in t} for z, t in plan.levels.get("imagery", {}).items()
        }
        self._heightmaps: OrderedDict[TileKey, np.ndarray | None] = OrderedDict()
        self._max = max_heightmaps
        self._lock = threading.Lock()
        self.downloaders = (
            {name: TileDownloader(p, cache) for name, p in self.sources.items()}
            if on_demand
            else {}
        )
        self.on_demand = on_demand
        self.borders = borders

    @property
    def imagery(self) -> TileProvider:
        return self.sources["imagery"]

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

    def texture_zoom_for(self, key: TileKey, provider: TileProvider) -> int:
        z, x, y = key
        for d in range(TEXEL_ZOOM_OFFSET, -1, -1):
            zi = z + d
            if zi > provider.max_zoom:
                continue
            tiles = self._imagery_sets.get(zi)
            if tiles and ((x << d), (y << d)) in tiles:
                return zi
        return min(z + TEXEL_ZOOM_OFFSET, provider.max_zoom)

    def imagery_zoom_for(self, key: TileKey) -> int:
        return self.texture_zoom_for(key, self.imagery)

    def close(self) -> None:
        """Stop on-demand downloads so pending worker tasks finish quickly."""
        self.on_demand = False
        for downloader in self.downloaders.values():
            downloader.cancel()

    def _ensurer(self, source: str):
        downloader = self.downloaders.get(source)

        def ensure(z: int, x: int, y: int) -> None:
            if (
                self.on_demand
                and downloader is not None
                and (x, y) in self._imagery_sets.get(z, ())
            ):
                # network problems must never break rendering
                with contextlib.suppress(Exception):
                    downloader.fetch_one(z, x, y)

        return ensure

    def source_ready(self, source: str) -> bool:
        """False while a texture source is still initialising (then nodes skip it for now)."""
        if source == "borders":
            if self.borders is None:
                return True
            self.borders.allow_download = self.on_demand
            return self.borders.ready
        return True

    def texture_for(self, key: TileKey, source: str = "imagery") -> np.ndarray | None:
        if source == "borders":
            if self.borders is None:
                return None
            self.borders.allow_download = self.on_demand
            return self.borders.distance_field(*key)
        provider = self.sources.get(source)
        if provider is None:
            return None
        z, x, y = key
        zoom = self.texture_zoom_for(key, provider)
        return compose_tile_texture(self.cache, provider, z, x, y, zoom, self._ensurer(source))

    def imagery_for(self, key: TileKey) -> np.ndarray | None:
        return self.texture_for(key, "imagery")

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
