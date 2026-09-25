import io

import numpy as np
from PIL import Image

from earthling.data.cache import TileCache
from earthling.data.imagery import compose_tile_texture, load_tile_rgb
from earthling.data.providers import TileProvider

P = TileProvider(id="img", name="img", kind="imagery", ext="png", max_zoom=14)


def png(color):
    img = Image.new("RGB", (256, 256), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def quadrant_png():
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:128, :128] = (255, 0, 0)  # NW red
    arr[:128, 128:] = (0, 255, 0)  # NE green
    arr[128:, :128] = (0, 0, 255)  # SW blue
    arr[128:, 128:] = (255, 255, 0)  # SE yellow
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_fallback_crops_parent(tmp_path):
    cache = TileCache(tmp_path)
    cache.write("img", 10, 4, 6, "png", quadrant_png())
    img, up = load_tile_rgb(cache, P, 11, 9, 13)  # SE child of (10, 4, 6)
    assert up == 1
    assert tuple(img[128, 128]) == (255, 255, 0)


def test_compose_mosaic_with_missing_parts(tmp_path):
    cache = TileCache(tmp_path)
    cache.write("img", 12, 8, 8, "png", png((10, 20, 30)))
    cache.write("img", 12, 9, 9, "png", png((200, 100, 50)))
    tex = compose_tile_texture(cache, P, 11, 4, 4, 12)
    assert tex.shape == (512, 512, 3)
    assert tuple(tex[10, 10]) == (10, 20, 30)
    assert tuple(tex[300, 300]) == (200, 100, 50)
    assert tuple(tex[10, 300]) == (0, 0, 0)  # missing, no ancestor
    assert compose_tile_texture(cache, P, 11, 100, 100, 12) is None


def test_compose_clamps_to_provider_max_zoom(tmp_path):
    cache = TileCache(tmp_path)
    cache.write("img", 14, 0, 0, "png", png((1, 2, 3)))
    tex = compose_tile_texture(cache, P, 14, 0, 0, 18)
    assert tex.shape == (256, 256, 3)
