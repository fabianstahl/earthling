import moderngl
import numpy as np
import pytest

from earthling.core.geo import lonlat_to_tile, tile_to_lonlat
from earthling.data.dem import HEIGHTMAP_SAMPLES
from earthling.data.terrain_data import HeightmapRef
from earthling.render import lod
from earthling.render.camera import Camera, OrbitController
from earthling.render.renderer import Renderer
from earthling.render.terrain import MESH_GRID, grid_indices, shared_grid_attributes, tile_geometry
from helpers import FRAME, FakeTerrainData, SlopeTerrainData

N = MESH_GRID + 1


def test_grid_and_skirt_layout():
    idx = grid_indices(2, skirt=False)
    assert idx.size == 2 * 2 * 6 and idx.max() == 8
    full = grid_indices()
    attrs = shared_grid_attributes()
    assert full.max() == len(attrs) - 1
    assert attrs[: N * N, 2].max() == 0 and attrs[N * N :, 2].min() == 1
    assert len(attrs) == N * N + 4 * MESH_GRID


def test_adjacent_tiles_share_edge_vertices():
    tx, ty = lonlat_to_tile(7.0, 46.0, 12)
    x, y = int(tx), int(ty)
    a = tile_geometry(FRAME, 12, x, y)
    b = tile_geometry(FRAME, 12, x + 1, y)
    pa = (a.positions[: N * N] + a.origin).reshape(N, N, 3)
    pb = (b.positions[: N * N] + b.origin).reshape(N, N, 3)
    assert np.allclose(pa[:, -1], pb[:, 0], atol=1e-6)


def test_tangent_frame_and_width():
    tx, ty = lonlat_to_tile(7.0, 46.0, 12)
    g = tile_geometry(FRAME, 12, int(tx), int(ty))
    east, north, up = g.tangent
    assert east == pytest.approx([1, 0, 0], abs=0.01)
    assert north == pytest.approx([0, 1, 0], abs=0.01)
    assert up == pytest.approx([0, 0, 1], abs=0.01)
    assert g.width_m == pytest.approx(9783.9 * np.cos(np.radians(46.0)), rel=0.01)


def test_render_lod_terrain(gl_ctx):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    nodes = lod.NodeSet({10: [(int(tx), int(ty))], 11: [(2 * int(tx), 2 * int(ty))]})
    renderer.set_terrain_source(FakeTerrainData(), nodes)
    camera = Camera()
    fbo = gl_ctx.simple_framebuffer((96, 96))
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 96)  # roots
    lo, hi = renderer.scene_bounds()
    assert lo[2] == pytest.approx(1500) and hi[2] == pytest.approx(2500)
    OrbitController(camera).frame_bounds(lo, hi)
    assert renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 96)
    renderer.render(fbo, 96, 96, camera)
    assert renderer.terrain.fully_loaded()
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(96, 96, 3).astype(int)
    green = (img[..., 1] > 150) & (img[..., 0] < 80)
    assert green.mean() > 0.2


def test_pick_depth_hits_terrain(gl_ctx):
    from earthling.app.viewport import unproject_log_depth

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.pitch = -60.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
    fbo = gl_ctx.simple_framebuffer((128, 128))
    renderer.render(fbo, 128, 128, camera)
    depth = renderer.read_depth(64, 64)
    assert depth is not None and depth < 1.0
    hit = unproject_log_depth(camera, 64, 64, 128, 128, depth)
    assert 1400 < hit[2] < 2600  # on the plateau or the bump (z ~ up near the origin)


def test_sun_direction_lights_facing_slopes(gl_ctx):
    from earthling.render.lighting import SunPosition

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(SlopeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.pitch = -89.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
    fbo = gl_ctx.simple_framebuffer((64, 64))

    def brightness(azimuth):
        sun = SunPosition(azimuth, 30.0).direction_enu()
        renderer.terrain.lighting_uniforms = {
            "u_sun_dir": sun, "u_sun_radiance": (1.0, 1.0, 1.0),
            "u_sky_ambient": (0.1, 0.1, 0.1), "u_ground_ambient": (0.1, 0.1, 0.1),
        }  # fmt: skip
        renderer.render(fbo, 64, 64, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(64, 64, 3)
        return img[24:40, 24:40].mean()

    assert brightness(270.0) > brightness(90.0) + 20


def test_dense_fog_hides_terrain_contrast(gl_ctx):
    from earthling.core.scene import Scene

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    renderer.store = Scene().store
    renderer.timezone = "Europe/Paris"
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(SlopeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 64)
    fbo = gl_ctx.simple_framebuffer((64, 64))

    def contrast():
        renderer.render(fbo, 64, 64, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(64, 64, 3)
        return img[20:44, 20:44].astype(float).mean(axis=2).std()  # luminance contrast

    clear = contrast()
    renderer.store.set("fog.enabled", True)
    renderer.store.set("fog.density", 20.0)
    renderer.store.set("fog.base", 6000.0)
    renderer.store.set("fog.falloff", 5000.0)
    renderer.store.set("fog.sun_scatter", 0.0)  # no view-dependent glow
    assert contrast() < 0.3 * clear


def test_skirt_depth_covers_relief_and_exaggeration():
    from types import SimpleNamespace

    from earthling.render.terrain import skirt_depth

    geometry = SimpleNamespace(width_m=6400.0)  # 100 m vertex spacing
    steep = SimpleNamespace(geometry=geometry, min_h=1000.0, max_h=3500.0)
    flat = SimpleNamespace(geometry=geometry, min_h=1000.0, max_h=1001.0)
    assert skirt_depth(steep, 1.0) >= 2500.0
    assert skirt_depth(steep, 2.5) >= 2.5 * 2500.0
    assert skirt_depth(flat, 1.0) >= 400.0  # at least a few vertex spacings


def test_failed_child_keeps_parent_drawn(gl_ctx):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    x, y = int(tx), int(ty)
    children = [(2 * x + dx, 2 * y + dy) for dx in (0, 1) for dy in (0, 1)]
    nodes = lod.NodeSet({10: [(x, y)], 11: children})

    class Flaky(FakeTerrainData):
        def heightmap_for(self, key):
            if key == (11, 2 * x, 2 * y):
                raise RuntimeError("corrupt tile")
            return super().heightmap_for(key)

    renderer.set_terrain_source(Flaky(), nodes)
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 400)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.distance *= 0.3
    orbit.apply()
    assert renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 400, timeout_s=10)
    drawn = renderer.terrain.last_selection.draw
    assert (10, x, y) in drawn  # the parent still covers the failed quadrant
    assert (11, 2 * x, 2 * y) not in drawn


class LeveledTerrainData(FakeTerrainData):
    """A west-east ramp; the z >= 11 heightmaps add a ripple their z 10 parent lacks."""

    def heightmap_for(self, key):
        z, x, _ = key
        cols = np.arange(HEIGHTMAP_SAMPLES, dtype=np.float32)
        n = 1 << (z - 10)
        start = (x % n) / n  # position of this tile within the z 10 parent
        east = start + (cols - 1.0) / 256.0 / n  # 0..1 across the parent
        heights = np.tile(1500.0 + 800.0 * east, (HEIGHTMAP_SAMPLES, 1))
        if z >= 11:
            heights = heights + 150.0 * np.sin(cols / 6.0)[None, :]
        return HeightmapRef(key, heights.astype(np.float32), 256.0, (0.0, 0.0))


def test_geomorphing_starts_from_the_parent_surface(gl_ctx):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    parent = (10, int(tx), int(ty))
    kids = lod.children(parent)
    renderer.set_terrain_source(
        LeveledTerrainData(), lod.NodeSet({10: [parent[1:]], 11: [k[1:] for k in kids]})
    )
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.distance *= 0.35
    orbit.pitch = -40.0
    orbit.apply()
    assert renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
    assert all(k in renderer.terrain._resident for k in (parent, *kids))
    fbo = gl_ctx.simple_framebuffer((128, 128))

    def frame(draw, morph):
        renderer.terrain.update = lambda *a: lod.Selection(draw=list(draw), morph=morph)
        renderer.render(fbo, 128, 128, camera)
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).astype(float)

    coarse = frame([parent], {})
    start = frame(kids, {k: 0.0 for k in kids})
    detail = frame(kids, {k: 1.0 for k in kids})
    assert np.abs(start - coarse).mean() < 1.0  # the split itself is invisible
    assert np.abs(detail - coarse).mean() > 3.0 * max(np.abs(start - coarse).mean(), 0.3)
    halfway = frame(kids, {k: 0.5 for k in kids})
    assert np.abs(halfway - coarse).mean() < np.abs(detail - coarse).mean()
    renderer.terrain.geomorph = False  # switched off: new nodes appear with their own heights
    assert np.abs(frame(kids, {k: 0.0 for k in kids}) - detail).mean() < 1.0


def test_selection_morph_factors():
    from earthling.render.lod import MORPH_END

    assert MORPH_END < 2.0  # fully morphed before the children split themselves


class CheckerTerrainData(FakeTerrainData):
    """Imagery with a coarse checkerboard (luminance edges for the detail normals)."""

    def imagery_for(self, key):
        yy, xx = np.mgrid[0:256, 0:256]
        checker = ((xx // 2 + yy // 2) % 2).astype(np.uint8)
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb[...] = (60 + 140 * checker)[..., None]
        return rgb


def test_detail_normals_add_relief_without_artifacts(gl_ctx):
    from earthling.core.scene import Scene

    def render(data, strength):
        renderer = Renderer(gl_ctx)
        renderer.set_scene(FRAME, [])
        tx, ty = lonlat_to_tile(7.0, 46.0, 11)
        renderer.set_terrain_source(data, lod.NodeSet({11: [(int(tx), int(ty))]}))
        scene = Scene()
        scene.store.set("terrain.detail_normals", strength)
        scene.store.set("terrain.detail_distance", 30.0)
        scene.store.set("shadows.enabled", False)
        renderer.store = scene.store
        camera = Camera()
        renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
        orbit = OrbitController(camera)
        orbit.frame_bounds(*renderer.scene_bounds())
        orbit.distance *= 0.08  # close range: this is where the relief matters
        orbit.pitch = -50.0
        orbit.apply()
        renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
        fbo = gl_ctx.simple_framebuffer((128, 128))
        renderer.render(fbo, 128, 128, camera)
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(128, 128, 3)

    flat, relief = render(CheckerTerrainData(), 0.0), render(CheckerTerrainData(), 3.0)
    assert np.abs(flat.astype(int) - relief.astype(int)).mean() > 0.5
    black = lambda img: int((img.max(axis=2) < 3).sum())  # noqa: E731
    assert black(relief) <= black(flat)  # no NaN normals
    uniform_a, uniform_b = render(FakeTerrainData(), 0.0), render(FakeTerrainData(), 3.0)
    assert np.abs(uniform_a.astype(int) - uniform_b.astype(int)).max() <= 1  # flat imagery


def test_child_without_dem_keeps_the_parent_drawn(gl_ctx):
    """A child node without heights must not leave a hole: the parent stays drawn, and the
    export's wait for a fully loaded view still finishes."""
    tx, ty = (int(v) for v in lonlat_to_tile(7.0, 46.0, 10))
    kids = [(11, 2 * tx + dx, 2 * ty + dy) for dx in (0, 1) for dy in (0, 1)]

    class PartialData(FakeTerrainData):
        def heightmap_for(self, key):
            return None if key == kids[0] else super().heightmap_for(key)

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    nodes = lod.NodeSet({10: [(tx, ty)], 11: [k[1:] for k in kids]})
    renderer.set_terrain_source(PartialData(), nodes)
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 400)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.distance *= 0.3  # close enough to want the children (see the full-data case)
    orbit.apply()
    assert renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 400, timeout_s=10)
    sel = renderer.terrain.last_selection
    assert (10, tx, ty) in sel.draw and not any(k in sel.draw for k in kids)


def test_refined_nodes_fade_in_from_the_parent_imagery(gl_ctx):
    tx, ty = (int(v) for v in lonlat_to_tile(7.0, 46.0, 10))
    kids = [(11, 2 * tx + dx, 2 * ty + dy) for dx in (0, 1) for dy in (0, 1)]

    class TwoColours(FakeTerrainData):
        def imagery_for(self, key):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            rgb[..., 0 if key[0] == 10 else 1] = 220  # parent red, children green
            return rgb

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    renderer.set_terrain_source(
        TwoColours(), lod.NodeSet({10: [(tx, ty)], 11: [k[1:] for k in kids]})
    )
    renderer.store = None
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 400)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.distance *= 0.3
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 400)
    fbo = gl_ctx.simple_framebuffer((64, 64))
    renderer.render(fbo, 64, 64, camera)  # sets up lighting and the scene target

    def colour(morph):
        renderer.target.fbo.use()
        renderer.target.fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        gl_ctx.enable(moderngl.DEPTH_TEST)
        renderer.terrain.draw(camera, camera.view_projection(1.0), kids,
                              morph={k: morph for k in kids})  # fmt: skip
        raw = renderer.target.color.read()
        img = np.frombuffer(raw, dtype=np.float16).reshape(64, 64, 4)[24:40, 24:40, :3]
        return img.astype(np.float32).reshape(-1, 3).mean(axis=0)

    start, done = colour(0.0), colour(1.0)
    assert start[0] > start[1] * 2 and done[1] > done[0] * 2


def test_nodes_without_dem_pass_through_to_their_children(gl_ctx):
    """Coarse roots often have no DEM (the plan starts deeper): draw their children instead."""
    tx, ty = (int(v) for v in lonlat_to_tile(7.0, 46.0, 10))
    kids = [(11, 2 * tx + dx, 2 * ty + dy) for dx in (0, 1) for dy in (0, 1)]

    class NoRoot(FakeTerrainData):
        def heightmap_for(self, key):
            return None if key[0] == 10 else super().heightmap_for(key)

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    renderer.set_terrain_source(NoRoot(), lod.NodeSet({10: [(tx, ty)], 11: [k[1:] for k in kids]}))
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 100)
    OrbitController(camera).frame_bounds(
        np.array([-40e3, -40e3, 1500.0]), np.array([40e3, 40e3, 2500.0])
    )
    assert renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 100, timeout_s=10)
    assert sorted(renderer.terrain.last_selection.draw) == sorted(kids)  # far away, still drawn


def test_eviction_keeps_nodes_of_the_previous_frame(gl_ctx):
    """Over the memory budget, only nodes unused for more than one frame are evicted: the shadow
    casters loaded for an export frame (not in the camera selection) must survive the eviction
    in that frame's own update (step 19.27)."""
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    x, y = int(tx), int(ty)
    children = [(2 * x + i, 2 * y + j) for i in (0, 1) for j in (0, 1)]
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(x, y)], 11: children}))
    terrain = renderer.terrain
    camera = Camera(position=np.array([0.0, -150000.0, 150000.0]), pitch=-45.0)  # parent only
    vp = camera.view_projection(1.0)
    terrain.finish_loading(camera, vp, 256)
    for key in [(11, cx, cy) for cx, cy in children]:  # e.g. shadow casters loaded just now
        terrain._submit(key)
    for future in list(terrain._in_flight.values()):
        future.result(timeout=30)
    terrain._collect(None)
    terrain.memory_budget_mb = 0  # everything is over the budget
    terrain.update(camera, vp, 256)
    assert not any((11, cx, cy) in terrain.last_selection.draw for cx, cy in children)  # far
    assert all((11, cx, cy) in terrain._resident for cx, cy in children)  # loaded last frame
    for _ in range(3):  # unused for a few frames: now they may go
        terrain.update(camera, vp, 256)
    sel = set(terrain.last_selection.draw)
    keys = [(11, cx, cy) for cx, cy in children]
    assert all(k in sel or k not in terrain._resident for k in keys)


def test_eviction_keeps_siblings_needed_for_refinement(gl_ctx):
    """A node is only refined when all four children are resident, also children outside the
    frustum that are never drawn. Evicting such a sibling makes the selection fall back to the
    coarse parent (the dark shadow flash of step 19.27)."""
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    x, y = int(tx), int(ty)
    children = [(2 * x + i, 2 * y + j) for i in (0, 1) for j in (0, 1)]
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(x, y)], 11: children}))
    terrain = renderer.terrain
    # close above one child, looking down at it: the other children are outside the frustum
    lon, lat = tile_to_lonlat(2 * x + 0.5, 2 * y + 0.5, 11)
    camera = Camera(position=np.asarray(FRAME.geodetic_to_enu(lat, lon, 2500.0)), pitch=-89.0,
                    fov_y=20.0)  # fmt: skip
    vp = camera.view_projection(1.0)
    terrain.finish_loading(camera, vp, 512)
    draw = terrain.last_selection.draw
    assert draw and all(k[0] == 11 for k in draw)  # refined
    assert len(draw) < 4  # not all children visible
    terrain.memory_budget_mb = 0  # everything over the budget
    for _ in range(4):  # every frame refined: the hidden siblings were never evicted
        sel = terrain.update(camera, vp, 512)
        assert sel.draw and all(k[0] == 11 for k in sel.draw)


def test_texture_pool_recycles_equal_sizes(gl_ctx):
    from earthling.render.terrain import TexturePool

    pool = TexturePool(gl_ctx, max_free_per_size=1)
    first = pool.acquire(np.full((8, 8, 3), 10, dtype=np.uint8))
    pool.release(first)
    again = pool.acquire(np.full((8, 8, 3), 200, dtype=np.uint8))
    assert again is first  # reused, with the new content
    assert np.frombuffer(again.read(), dtype=np.uint8)[:3].tolist() == [200, 200, 200]
    other = pool.acquire(np.zeros((16, 16, 3), dtype=np.uint8))
    field = pool.acquire(np.zeros((8, 8, 2), dtype=np.float32))
    assert other is not first and field.dtype == "f2"  # (float fields as half floats)
    spare = pool.acquire(np.zeros((8, 8, 3), dtype=np.uint8))
    pool.release(again)
    pool.release(spare)  # one free texture per size is kept, the other one released
    assert sum(len(v) for v in pool._free.values()) == 1
    pool.clear()
    assert pool._free == {}


def terrain_shaders(gl_ctx):
    from earthling.render.layers import generate_glsl
    from earthling.render.shader_library import ShaderLibrary

    library = ShaderLibrary(gl_ctx)
    library.register_virtual("layers_generated.glsl", generate_glsl())  # (as the renderer)
    return library


def single_node_layer(gl_ctx, data):
    from earthling.render.terrain import TerrainLayer

    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    layer = TerrainLayer(gl_ctx, terrain_shaders(gl_ctx), workers=2)
    key = (10, int(tx), int(ty))
    layer.set_source(FRAME, data, lod.NodeSet({10: [key[1:]]}))
    camera = Camera()
    OrbitController(camera).frame_bounds(*layer_bounds(key))
    return layer, key, camera


def layer_bounds(key):
    lon, lat = tile_to_lonlat(key[1] + 0.5, key[2] + 0.5, key[0])
    centre = np.asarray(FRAME.geodetic_to_enu(lat, lon, 2000.0))
    return centre - 20_000.0, centre + 20_000.0


def test_a_layer_switched_on_later_loads_only_its_texture(gl_ctx):
    """Another texture source (e.g. the topo map) refreshes resident nodes; the imagery they
    have is kept, sources still initialising are waited for."""

    class Sources(FakeTerrainData):
        def __init__(self):
            super().__init__()
            self.ready, self.asked = False, []

        def source_ready(self, source):
            return source != "borders" or self.ready

        def texture_for(self, key, source="imagery"):
            self.asked.append(source)
            return np.full((32, 32, 3), 50, np.uint8)

    data = Sources()
    layer, key, camera = single_node_layer(gl_ctx, data)
    vp = camera.view_projection(1.0)
    assert layer.finish_loading(camera, vp, 256) and layer.resident_count == 1
    imagery = layer._resident[key].imagery
    assert data.asked == ["imagery"]
    layer.required_sources = frozenset({"imagery", "topo", "borders"})
    assert not layer.fully_loaded()
    assert layer.finish_loading(camera, vp, 256)
    node = layer._resident[key]
    assert node.imagery is imagery and node.textures["topo"] is not None
    assert "borders" not in node.textures and data.asked == ["imagery", "topo"]
    assert layer.fully_loaded()  # (borders: not ready, nothing to wait for yet)
    data.ready = True
    layer.update(camera, vp, 256)  # the refresh is requested with the next frame
    assert layer.pending_count == 1 and not layer.fully_loaded()
    assert layer.finish_loading(camera, vp, 256) and "borders" in layer._resident[key].textures
    layer.shutdown()


def test_clear_releases_everything_and_closes_the_data(gl_ctx):
    class Closable(FakeTerrainData):
        closed = 0

        def close(self):
            self.closed += 1

    data = Closable()
    layer, key, camera = single_node_layer(gl_ctx, data)
    assert layer.bounds() is None  # nothing loaded yet
    far = layer.node_bounds(key)  # (default heights until the node is there)
    layer.finish_loading(camera, camera.view_projection(1.0), 256)
    lo, hi = layer.bounds()
    assert lo[2] == pytest.approx(1500.0) and hi[2] == pytest.approx(2500.0)
    layer.invalidate_bounds()
    assert layer.node_bounds(key).radius < far.radius  # real heights: a tighter sphere
    layer._submit((11, 2 * key[1], 2 * key[2]))
    layer.clear()
    assert data.closed == 1 and layer.resident_count == 0 and layer.pending_count == 0
    assert layer._heightmaps == {} and layer.bounds() is None
    layer.shutdown()


def test_hot_reloaded_terrain_shader_gets_new_vertex_arrays(gl_ctx):
    layer, key, camera = single_node_layer(gl_ctx, FakeTerrainData())
    vp = camera.view_projection(1.0)
    layer.finish_loading(camera, vp, 64)
    fbo = gl_ctx.simple_framebuffer((64, 64))
    fbo.use()
    layer.draw(camera, vp, [key])
    node = layer._resident[key]
    (old_glo,) = node.vaos
    library = layer.shaders

    class Reloaded:
        program = terrain_shaders(gl_ctx).get("terrain")

        def get(self, name, defines=None):
            return self.program if not defines else library.get(name, defines=defines)

    layer.shaders = Reloaded()
    layer.draw(camera, vp, [key, (10, key[1] + 5, key[2])])  # (an unknown node is skipped)
    assert old_glo not in node.vaos and list(node.vaos) == [Reloaded.program.glo]
    layer.shutdown()
