"""Cascaded shadow maps for sunlight on terrain.

* Up to 4 cascades, each rendered into one quadrant of a single depth atlas.
* Split distances blend logarithmic and uniform schemes (``lambda`` = 0.75).
* Each cascade covers the bounding sphere of its view-frustum slice. The sphere radius is
  quantised and its centre snapped to the shadow texel grid in a *world-anchored* frame, so the
  shadows do not shimmer when the camera moves (important for video).
* Light-space matrices are camera-relative like all other rendering, which keeps precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import moderngl
import numpy as np
from pyglm import glm

from earthling.render.camera import Camera

CASCADES = 4
SHADOW_UNIT = 3  # texture unit of the shadow atlas
# how far behind a cascade (towards the sun) occluders are included
OCCLUDER_MARGIN_M = 25_000.0


def cascade_splits(
    near: float, far: float, count: int = CASCADES, lam: float = 0.75
) -> list[float]:
    splits = []
    for i in range(1, count + 1):
        f = i / count
        log = near * (far / near) ** f
        uni = near + (far - near) * f
        splits.append(lam * log + (1.0 - lam) * uni)
    return splits


def frustum_slice_sphere(camera: Camera, aspect: float, d0: float, d1: float):
    """Bounding sphere (centre camera-relative, radius) of the view frustum between depths."""
    t = math.tan(math.radians(camera.fov_y) / 2.0)
    f = np.asarray(camera.forward, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    right = np.cross(f, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    corners = []
    for d in (d0, d1):
        for sx in (-1, 1):
            for sy in (-1, 1):
                corners.append(f * d + right * sx * d * t * aspect + up * sy * d * t)
    pts = np.array(corners)
    # centre on the view axis minimising the radius (closed form for symmetric slices)
    h0 = np.linalg.norm(pts[0] - f * d0)
    h1 = np.linalg.norm(pts[4] - f * d1)
    z = (d1 * d1 - d0 * d0 + h1 * h1 - h0 * h0) / (2.0 * (d1 - d0))
    z = min(max(z, d0), d1)
    centre = f * z
    radius = float(np.linalg.norm(pts - centre, axis=1).max())
    return centre, radius


def light_rotation(sun_dir) -> glm.mat4:
    """View rotation looking from the sun towards the scene (camera at the origin)."""
    s = glm.normalize(glm.vec3(*sun_dir))
    up = glm.vec3(0.0, 0.0, 1.0) if abs(s.z) < 0.99 else glm.vec3(0.0, 1.0, 0.0)
    return glm.lookAt(glm.vec3(0.0), -s, up)


@dataclass
class Cascade:
    near: float
    far: float
    centre_rel: np.ndarray  # camera-relative
    radius: float
    view_proj: glm.mat4  # camera-relative world -> light clip space
    texel_world: float  # world size of one shadow texel


@dataclass
class ShadowSetup:
    cascades: list[Cascade] = field(default_factory=list)
    atlas_size: int = 4096

    def splits(self) -> tuple[float, float, float, float]:
        far = [c.far for c in self.cascades] + [1e30] * (CASCADES - len(self.cascades))
        return tuple(far[:CASCADES])  # type: ignore[return-value]


def compute_cascades(
    camera: Camera,
    aspect: float,
    sun_dir,
    distance: float,
    atlas_size: int,
    near: float = 20.0,
) -> ShadowSetup:
    cascade_res = atlas_size // 2
    rot = light_rotation(sun_dir)
    rot_np = np.array(rot.to_list(), dtype=np.float64).T  # row-major 4x4
    r3 = rot_np[:3, :3]
    sun = np.asarray(sun_dir, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    setup = ShadowSetup(atlas_size=atlas_size)
    d0 = near
    for d1 in cascade_splits(near, distance):
        centre_rel, radius = frustum_slice_sphere(camera, aspect, d0, d1)
        # (the radius only depends on fov/aspect/splits, so it is stable while flying)
        texel = 2.0 * radius / cascade_res
        # snap the centre to the texel grid in world-anchored light space
        centre_world = camera.position + centre_rel
        ls = r3 @ centre_world
        ls[0] = math.floor(ls[0] / texel) * texel
        ls[1] = math.floor(ls[1] / texel) * texel
        centre_world = r3.T @ ls
        c = centre_world - camera.position
        back = radius + OCCLUDER_MARGIN_M
        eye = glm.vec3(*(c + sun * back))
        view = glm.lookAt(eye, glm.vec3(*c), glm.vec3(*rot_np[1, :3]))
        proj = glm.ortho(-radius, radius, -radius, radius, 0.0, back + radius)
        setup.cascades.append(Cascade(d0, d1, c, radius, proj * view, texel))
        d0 = d1
    return setup


class ShadowMaps:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.size = 0
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.setup: ShadowSetup | None = None

    def ensure(self, atlas_size: int) -> None:
        if self.size == atlas_size and self.fbo is not None:
            return
        self.release()
        self.depth = self.ctx.depth_texture((atlas_size, atlas_size))
        self.depth.compare_func = "<="  # hardware PCF
        self.depth.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth.repeat_x = self.depth.repeat_y = False
        self.fbo = self.ctx.framebuffer(depth_attachment=self.depth)
        self.size = atlas_size

    def viewport(self, index: int) -> tuple[int, int, int, int]:
        half = self.size // 2
        return ((index % 2) * half, (index // 2) * half, half, half)

    def atlas_matrix(self, index: int, view_proj: glm.mat4) -> glm.mat4:
        """Camera-relative world -> atlas texture coordinates (xy) and depth (z) in [0, 1]."""
        ox, oy = (index % 2) * 0.5, (index // 2) * 0.5
        bias = glm.mat4(
            glm.vec4(0.25, 0.0, 0.0, 0.0),
            glm.vec4(0.0, 0.25, 0.0, 0.0),
            glm.vec4(0.0, 0.0, 0.5, 0.0),
            glm.vec4(ox + 0.25, oy + 0.25, 0.5, 1.0),
        )
        return bias * view_proj

    def uniforms(self) -> dict[str, object]:
        if self.setup is None:
            return {"u_shadows_enabled": False}
        mats = [self.atlas_matrix(i, c.view_proj) for i, c in enumerate(self.setup.cascades)]
        return {
            "u_shadows_enabled": True,
            "u_shadow_map": SHADOW_UNIT,
            "u_cascade_splits": self.setup.splits(),
            "u_shadow_texel": tuple(c.texel_world for c in self.setup.cascades),
            "u_shadow_atlas_texel": 1.0 / self.size,
            "_matrices": mats,
        }

    def release(self) -> None:
        for obj in (self.fbo, self.depth):
            if obj is not None:
                obj.release()
        self.fbo = self.depth = None
        self.size = 0
