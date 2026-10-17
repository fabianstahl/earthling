"""Step 18.6: imagery colours matched to a consistent reference mosaic."""

import numpy as np

from earthling.data.cache import TileCache
from earthling.data.harmonize import ColourReference
from earthling.data.imagery import compose_tile_texture
from earthling.data.providers import TileProvider
from helpers import png

SOURCE = TileProvider(id="src", name="source", kind="imagery", ext="png")
REFERENCE = TileProvider(id="ref", name="reference", kind="imagery", ext="png", max_zoom=4)


def finer(key):
    """The imagery zoom of every node: one level finer than the node."""
    return key[0] + 1


def fill(cache, provider, z, colour_of, span=range(8, 18)):
    """Tiles of zoom z (x, y in ``span``) with a colour from ``colour_of(x, y)`` plus detail."""
    rng = np.random.default_rng(1)
    for x in span:
        for y in span:
            tile = np.ones((256, 256, 3)) * np.asarray(colour_of(x, y), dtype=float)
            tile += rng.normal(0.0, 6.0, tile.shape)  # detail that must survive
            cache.write(provider.id, z, x, y, provider.ext, png(np.clip(tile, 0, 255)))


def test_large_scale_colour_follows_the_reference_detail_stays(tmp_path):
    cache = TileCache(tmp_path)
    fill(cache, SOURCE, 5, lambda x, y: (200, 200, 205))  # a glinting white sea
    fill(cache, REFERENCE, 4, lambda x, y: (35, 70, 95), range(3, 10))  # the reference sea
    harmonise = ColourReference(cache, SOURCE, REFERENCE, max_zoom=4)
    image = compose_tile_texture(cache, SOURCE, 4, 5, 5, 5)
    out = harmonise.apply(image, 4, 5, 5, finer)
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), (35, 70, 95), atol=5.0)
    assert out.astype(float).std() > 0.5  # the source's detail is kept (scaled)


def test_gain_is_continuous_between_neighbouring_nodes(tmp_path):
    cache = TileCache(tmp_path)
    # the source mosaic changes colour abruptly between tile columns, the reference smoothly
    fill(cache, SOURCE, 5, lambda x, y: (120 + 40 * (x % 2), 110, 100))
    fill(cache, REFERENCE, 4, lambda x, y: (100 + 4 * x, 110, 100), range(3, 10))
    harmonise = ColourReference(cache, SOURCE, REFERENCE, max_zoom=4)
    # the gain field (applied to a flat grey) meets without a step at the node border
    grey = np.full((512, 512, 3), 128, dtype=np.uint8)
    left = harmonise.apply(grey, 4, 6, 6, finer).astype(float)
    right = harmonise.apply(grey, 4, 7, 6, finer).astype(float)
    assert np.abs(left[:, -1] - right[:, 0]).max() <= 2.0
    # the abrupt jump of the source between its tile columns is largely evened out
    image = compose_tile_texture(cache, SOURCE, 4, 6, 6, 5)
    out = harmonise.apply(image, 4, 6, 6, finer).astype(float)
    before = abs(image[:, 128:256].mean() - image[:, 256:384].mean())
    after = abs(out[:, 128:256].mean() - out[:, 256:384].mean())
    assert after < 0.6 * before


def test_no_seam_where_the_imagery_zoom_of_neighbours_differs(tmp_path):
    """Two nodes of one level, one textured from finer imagery (the edge of a download zone):
    the same mosaic on both sides must stay seamless after harmonising."""
    cache = TileCache(tmp_path)
    fill(cache, SOURCE, 5, lambda x, y: (120, 110, 100))
    fill(cache, SOURCE, 6, lambda x, y: (120, 110, 100), range(16, 36))
    fill(cache, REFERENCE, 4, lambda x, y: (100 + 6 * x, 110, 100), range(3, 10))
    harmonise = ColourReference(cache, SOURCE, REFERENCE, max_zoom=4)

    def zoom_of(key):
        return 6 if key[1] >= 7 else 5

    left = harmonise.apply(compose_tile_texture(cache, SOURCE, 4, 6, 6, 5), 4, 6, 6, zoom_of)
    right = harmonise.apply(compose_tile_texture(cache, SOURCE, 4, 7, 6, 6), 4, 7, 6, zoom_of)
    edge_left = left[:, -8:].reshape(-1, 3).mean(axis=0)
    edge_right = right[:, :16].reshape(-1, 3).mean(axis=0)
    assert np.allclose(edge_left, edge_right, atol=3.0)


def test_strength_zero_leaves_the_imagery_alone(tmp_path):
    cache = TileCache(tmp_path)
    fill(cache, SOURCE, 5, lambda x, y: (200, 200, 205))
    fill(cache, REFERENCE, 4, lambda x, y: (35, 70, 95), range(3, 10))
    image = compose_tile_texture(cache, SOURCE, 4, 5, 5, 5)
    harmonise = ColourReference(cache, SOURCE, REFERENCE, max_zoom=4, strength=0.0)
    assert np.array_equal(harmonise.apply(image, 4, 5, 5, finer), image)
