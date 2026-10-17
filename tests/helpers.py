"""Shared test helpers: the demo project, a local frame, PNG bytes and terrain fakes."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from earthling.core.geo import LocalFrame
from earthling.data.dem import HEIGHTMAP_SAMPLES
from earthling.data.terrain_data import HeightmapRef

DEMO = Path(__file__).parents[1] / "examples" / "alps_demo"  # two days around Trient
FRAME = LocalFrame(46.0, 7.0, 0.0)


def png(color, size: int = 256) -> bytes:
    """PNG bytes of a plain ``color`` tile, or of an RGB array."""
    if isinstance(color, np.ndarray):
        image = Image.fromarray(color.astype(np.uint8))
    else:
        image = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeTerrainData:
    """Constant plateau with a bump; imagery only at z >= 11."""

    def __init__(self):
        self.heights = np.full((HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 1500.0, dtype=np.float32)
        self.heights[100:150, 100:150] = 2500.0

    def heightmap_for(self, key):
        return HeightmapRef(key, self.heights, 256.0, (0.0, 0.0))

    def texture_for(self, key, source="imagery"):
        return self.imagery_for(key) if source == "imagery" else None

    def imagery_for(self, key):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[..., 1] = 200
        return rgb


class SlopeTerrainData(FakeTerrainData):
    """Height rises towards the east -> the slope faces west."""

    def __init__(self):
        super().__init__()
        cols = np.arange(HEIGHTMAP_SAMPLES, dtype=np.float32)
        self.heights = np.tile(1000.0 + cols * 40.0, (HEIGHTMAP_SAMPLES, 1))

    def imagery_for(self, key):
        return np.full((64, 64, 3), 128, dtype=np.uint8)
