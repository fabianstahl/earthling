import numpy as np

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.render import lod
from earthling.render.camera import Camera

FRAME = LocalFrame(46.0, 7.0, 0.0)


def node_set_around(lon, lat, zmin, zmax):
    levels = {}
    for z in range(zmin, zmax + 1):
        tx, ty = lonlat_to_tile(lon, lat, z)
        levels[z] = [(int(tx) + dx, int(ty) + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    return lod.NodeSet(levels)


def run_selection(camera, nodes, ready, height=1080):
    vp = camera.view_projection(16 / 9)
    planes = lod.frustum_planes(vp)
    ppr = height / (2 * np.tan(np.radians(camera.fov_y) / 2))
    cache = {}

    def bounds(key):
        if key not in cache:
            cache[key] = lod.node_bounds(FRAME, key, 0, 3000)
        return cache[key]

    return lod.select_nodes(nodes, bounds, ready, camera.position, planes, ppr)


def test_node_set_from_plan_is_parent_closed():
    dem = {10: np.array([[530, 360]]), 11: np.array([[1061, 721]])}
    imagery = {15: np.array([[16980, 11540]])}
    ns = lod.NodeSet.from_plan(dem, imagery)
    assert (13, 16980 >> 2, 11540 >> 2) in ns
    for z in range(11, 14):
        for x, y in ns.levels[z]:
            assert (z - 1, x >> 1, y >> 1) in ns


def test_frustum_planes_cull_behind_camera():
    cam = Camera(position=np.zeros(3), heading=0.0, pitch=0.0)
    planes = lod.frustum_planes(cam.view_projection(1.0))
    assert lod.sphere_visible(planes, np.array([0.0, 1000.0, 0.0]), 1.0)
    assert not lod.sphere_visible(planes, np.array([0.0, -1000.0, 0.0]), 1.0)
    assert not lod.sphere_visible(planes, np.array([5000.0, 1000.0, 0.0]), 1.0)


def test_refines_near_camera_only_when_children_ready():
    nodes = node_set_around(7.0, 46.0, 8, 14)
    cam = Camera(position=np.array([0.0, -3000.0, 2500.0]), heading=0.0, pitch=-20.0)
    # nothing ready: roots are requested, nothing drawn
    sel = run_selection(cam, nodes, lambda k: False)
    assert sel.draw == [] and set(sel.request) == set(nodes.roots())
    # everything ready: deep refinement near the camera
    sel = run_selection(cam, nodes, lambda k: True)
    zooms = [k[0] for k in sel.draw]
    assert max(zooms) >= 13 and sel.request == []
    # only roots ready: roots are drawn, their children requested
    roots = set(nodes.roots())
    sel = run_selection(cam, nodes, lambda k: k in roots)
    assert set(sel.draw) <= roots and sel.request


def test_far_camera_uses_coarse_nodes():
    nodes = node_set_around(7.0, 46.0, 8, 14)
    cam = Camera(position=np.array([0.0, -200_000.0, 150_000.0]))
    cam.look_at([0.0, 0.0, 0.0])
    sel = run_selection(cam, nodes, lambda k: True)
    assert max(k[0] for k in sel.draw) <= 10


def test_selection_has_no_overlaps():
    nodes = node_set_around(7.0, 46.0, 8, 14)
    cam = Camera(position=np.array([0.0, -3000.0, 2500.0]), heading=10.0, pitch=-25.0)
    sel = run_selection(cam, nodes, lambda k: True)
    drawn = set(sel.draw)
    for key in drawn:
        p = key
        while p[0] > 8:
            p = lod.parent(p)
            assert p not in drawn
