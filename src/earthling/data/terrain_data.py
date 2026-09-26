"""CPU-side terrain data access for the LOD renderer (heightmaps and imagery per node)."""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from earthling.core.aoi import TilePlan
from earthling.core.geo import tile_bounds_mercator
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


def fill_from_parent(heights: np.ndarray, parent: np.ndarray, qx: int, qy: int) -> np.ndarray:
    """Fill the NaN samples of a node heightmap with the (bilinearly sampled) heights of its
    parent. (qx, qy): the node's quadrant in the parent. Both grids store sample s at index
    s + 1 (one border sample each side), GRID_INTERVALS intervals across the node."""
    n = heights.shape[0]
    half = GRID_INTERVALS / 2.0
    coords = np.arange(n, dtype=np.float64) - 1.0  # own sample index
    px = qx * half + coords / 2.0 + 1.0  # parent array index
    py = qy * half + coords / 2.0 + 1.0
    x0 = np.clip(np.floor(px).astype(int), 0, parent.shape[1] - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, parent.shape[0] - 2)
    fx, fy = px - x0, py - y0
    top = parent[y0][:, x0] * (1 - fx) + parent[y0][:, x0 + 1] * fx
    bottom = parent[y0 + 1][:, x0] * (1 - fx) + parent[y0 + 1][:, x0 + 1] * fx
    sampled = top * (1 - fy)[:, None] + bottom * fy[:, None]
    out = heights.copy()
    gaps = ~np.isfinite(out) & np.isfinite(sampled)
    out[gaps] = sampled[gaps]
    return out.astype(np.float32)


class TerrainData:
    """Heightmaps and textures for LOD nodes.

    ``sources`` maps tile-source names used by texture layers ("imagery", "topo", ...) to
    providers, or to lists of providers in priority order: regional providers are composited
    over the others within their coverage, feathered at its edge, and missing tiles fall back to
    the next provider. ``dem_id`` may likewise be a list of DEM source ids in priority order.
    All texture sources use the imagery zones of the tile plan (capped at each provider's
    maximum zoom); missing tiles can be downloaded on demand from worker threads.
    """

    def __init__(
        self,
        cache: TileCache,
        dem_id: str | list[str],
        sources: dict[str, TileProvider | list[TileProvider]] | TileProvider,
        plan: TilePlan,
        max_heightmaps: int = 512,
        on_demand: bool = False,
        borders=None,
    ) -> None:
        """``borders``: optional BorderData providing the "borders" texture source."""
        if isinstance(sources, TileProvider):
            sources = {"imagery": sources}
        self.cache = cache
        self.dem_ids = [dem_id] if isinstance(dem_id, str) else list(dem_id)
        self.dem_id = self.dem_ids[0]
        self._dem_coverage = {sid: _dem_coverage(sid) for sid in self.dem_ids}
        self.stacks: dict[str, list[TileProvider]] = {
            name: [p] if isinstance(p, TileProvider) else list(p) for name, p in sources.items()
        }
        self.sources = {name: stack[0] for name, stack in self.stacks.items()}
        self.plan = plan
        self._imagery_sets = {
            z: {(int(x), int(y)) for x, y in t} for z, t in plan.levels.get("imagery", {}).items()
        }
        self._heightmaps: OrderedDict[TileKey, np.ndarray | None] = OrderedDict()
        self._max = max_heightmaps
        self._lock = threading.Lock()
        self.downloaders = (
            {p.id: TileDownloader(p, cache) for stack in self.stacks.values() for p in stack}
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
        heights = self._composite_heightmap(key)
        if heights is not None and key[0] > 0 and not np.isfinite(heights).all():
            # gaps no source covers at this zoom (e.g. a national DEM ending at the border):
            # fill them from the parent's heights instead of leaving holes in the terrain
            z, x, y = key
            parent = self._heightmap((z - 1, x >> 1, y >> 1))
            if parent is not None:
                heights = fill_from_parent(heights, parent, x & 1, y & 1)
        with self._lock:
            self._heightmaps[key] = heights
            while len(self._heightmaps) > self._max:
                self._heightmaps.popitem(last=False)
        return heights

    def _composite_heightmap(self, key: TileKey) -> np.ndarray | None:
        """The heightmap of ``key`` from the DEM sources in priority order. All sources are
        baked on the same sample grid, so they blend per sample: feathered at coverage edges,
        lower sources fill where higher ones have no data."""
        z, x, y = key
        bounds = None
        layers: list[tuple[np.ndarray, np.ndarray | None]] = []
        for sid in self.dem_ids:
            coverage = self._dem_coverage[sid]
            kind = "full"
            if coverage is not None:
                bounds = bounds or tile_bounds_mercator(z, x, y)
                kind = coverage.classify(bounds)
                if kind == "none":
                    continue
            if self.cache.is_missing(sid, z, x, y):
                continue
            heights = read_heightmap(self.cache, sid, z, x, y)
            if heights is None:
                continue
            valid = np.isfinite(heights)
            weight = None if kind == "full" else _sample_weights(coverage, z, x, y)
            layers.append((heights, weight))
            if valid.all() and weight is None:
                break
        if not layers:
            return None
        if len(layers) == 1 and layers[0][1] is None:
            return layers[0][0]
        out = np.full(layers[0][0].shape, np.nan, dtype=np.float32)
        for heights, weight in reversed(layers):
            valid = np.isfinite(heights)
            w = valid.astype(np.float32) if weight is None else weight * valid
            base = np.isfinite(out)
            blended = out + (np.where(valid, heights, 0.0) - out) * w
            out = np.where(base, np.where(w > 0, blended, out), np.where(w > 0, heights, np.nan))
        return out.astype(np.float32)

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

    def _ensurer(self, provider: TileProvider):
        downloader = self.downloaders.get(provider.id)
        coverage = provider.coverage_area

        def ensure(z: int, x: int, y: int) -> None:
            if (
                self.on_demand
                and downloader is not None
                and (x, y) in self._imagery_sets.get(z, ())
                and (coverage is None or coverage.classify(tile_bounds_mercator(z, x, y)) != "none")
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
        stack = self.stacks.get(source)
        if not stack:
            return None
        z, x, y = key
        if len(stack) == 1 and stack[0].coverage is None:
            provider = stack[0]
            zoom = self.texture_zoom_for(key, provider)
            return compose_tile_texture(
                self.cache, provider, z, x, y, zoom, self._ensurer(provider)
            )
        return self._composite_texture(key, stack)

    def _composite_texture(self, key: TileKey, stack: list[TileProvider]) -> np.ndarray | None:
        z, x, y = key
        bounds = tile_bounds_mercator(z, x, y)
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        for provider in stack:
            coverage = provider.coverage_area
            kind = "full" if coverage is None else coverage.classify(bounds)
            if kind == "none":
                continue
            zoom = self.texture_zoom_for(key, provider)
            found = compose_tile_texture(
                self.cache, provider, z, x, y, zoom, self._ensurer(provider), with_mask=True
            )
            if found is None:
                continue
            image, mask = found
            if kind == "partial":
                mask = mask * coverage.pixel_weights(bounds, image.shape[1], image.shape[0])
            layers.append((image, mask))
            if mask.min() >= 1.0:
                break  # opaque: providers below are hidden
        if not layers:
            return None
        size = max(image.shape[0] for image, _ in layers)
        out = np.zeros((size, size, 3), dtype=np.float32)
        for image, mask in reversed(layers):
            if image.shape[0] != size:
                image, mask = _resize(image, size), _resize(mask, size)
            out += (image.astype(np.float32) - out) * mask[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

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

    def heights_at(self, lon: np.ndarray, lat: np.ndarray, z: int | None = None) -> np.ndarray:
        """Vectorised :meth:`height_at`; NaN where no DEM data is available."""
        from earthling.core.geo import lonlat_to_tile

        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        if z is None:
            dem_levels = self.plan.levels.get("dem", {})
            z = max(dem_levels) if dem_levels else 12
        tx, ty = lonlat_to_tile(lon, lat, z)
        ix, iy = np.floor(tx).astype(np.int64), np.floor(ty).astype(np.int64)
        out = np.full(lon.shape, np.nan)
        keys = np.stack([ix, iy], axis=-1).reshape(-1, 2)
        flat_tx, flat_ty = tx.ravel(), ty.ravel()
        flat_out = out.ravel()
        for x, y in np.unique(keys, axis=0):
            sel = (keys[:, 0] == x) & (keys[:, 1] == y)
            ref = self.heightmap_for((z, int(x), int(y)))
            if ref is None:
                continue
            h = ref.heights
            sx = ref.offset[0] + (flat_tx[sel] - x) * ref.scale + 1.0
            sy = ref.offset[1] + (flat_ty[sel] - y) * ref.scale + 1.0
            x0 = np.clip(np.floor(sx).astype(np.int64), 0, h.shape[1] - 2)
            y0 = np.clip(np.floor(sy).astype(np.int64), 0, h.shape[0] - 2)
            fx, fy = sx - x0, sy - y0
            top = h[y0, x0] * (1 - fx) + h[y0, x0 + 1] * fx
            bottom = h[y0 + 1, x0] * (1 - fx) + h[y0 + 1, x0 + 1] * fx
            flat_out[sel] = top * (1 - fy) + bottom * fy
        return flat_out.reshape(lon.shape)


def _dem_coverage(source_id: str):
    from earthling.data.dem import DEM_SOURCES

    source = DEM_SOURCES.get(source_id)
    return None if source is None else source.coverage_area


def _sample_weights(coverage, z: int, x: int, y: int) -> np.ndarray:
    from earthling.data.dem import sample_positions_mercator

    xs, ys = sample_positions_mercator(z, x, y)
    return coverage.weights(xs, ys)


def _resize(array: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    if array.dtype == np.uint8:
        return np.asarray(Image.fromarray(array).resize((size, size), Image.Resampling.BILINEAR))
    img = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR))
