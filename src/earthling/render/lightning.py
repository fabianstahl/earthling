"""Thunderstorms: a deterministic lightning schedule, branching bolts and flashes.

Strikes are decided per 50 ms slot of the timeline with a hash of (seed, slot): a slot fires
with probability rate(t) * slot length, so any frame can be rendered on its own and exports are
reproducible. A strike lasts DURATION_S: a main flash and two return strokes. Bolts run from
the cloud base to the terrain, are drawn as glowing, depth-tested screen-space ribbons, and the
flash lights clouds, sky and terrain.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import moderngl
import numpy as np

SLOT_S = 0.05
DURATION_S = 0.7


def _mix(value: int) -> int:
    """splitmix64 finaliser."""
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def hash01(seed: int, slot: int, salt: int = 0) -> float:
    return _mix((seed * 0x100000001B3) ^ (slot * 0x9E3779B1) ^ (salt * 0x85EBCA6B)) / 2.0**64


@dataclass(frozen=True)
class Strike:
    slot: int
    start: float
    seed: int

    def rand(self, salt: int) -> float:
        return hash01(self.seed, self.slot, salt)


def strikes_between(rate_at: Callable[[float], float], seed: int, t0: float, t1: float):
    """Strikes starting in [t0, t1] (rate in strikes per minute, may vary with time)."""
    out = []
    for slot in range(math.floor(t0 / SLOT_S), math.floor(t1 / SLOT_S) + 1):
        ts = slot * SLOT_S
        rate = max(rate_at(ts), 0.0)
        if rate <= 0.0:
            continue
        if hash01(seed, slot, 1) < rate / 60.0 * SLOT_S:
            start = ts + hash01(seed, slot, 2) * SLOT_S
            if t0 <= start <= t1:
                out.append(Strike(slot, start, seed))
    return out


def active_strikes(rate_at, seed: int, t: float) -> list[Strike]:
    return [s for s in strikes_between(rate_at, seed, t - DURATION_S, t) if s.start <= t]


def flash_intensity(strike: Strike, t: float) -> float:
    """Main stroke plus two return strokes, each a fast exponential decay."""
    dt = t - strike.start
    if dt < 0.0 or dt > DURATION_S:
        return 0.0
    strokes = ((0.0, 1.0), (0.07 + 0.06 * strike.rand(3), 0.65), (0.2 + 0.12 * strike.rand(4), 0.4))
    value = 0.0
    for offset, amplitude in strokes:
        d = dt - offset
        if d >= 0.0:
            value += amplitude * math.exp(-d / 0.045)
    return value * (1.0 - dt / DURATION_S) ** 0.5


def bolt_paths(top: np.ndarray, bottom: np.ndarray, strike: Strike, levels: int = 7):
    """Jagged main channel and a few branches: [(points (n, 3), intensity)]."""
    rng = np.random.default_rng(strike.seed * 1_000_003 + strike.slot)

    def jag(a: np.ndarray, b: np.ndarray, roughness: float, depth: int) -> np.ndarray:
        points = np.array([a, b], dtype=np.float64)
        for _ in range(depth):
            new = [points[0]]
            for p, q in zip(points[:-1], points[1:], strict=True):
                seg = q - p
                length = np.linalg.norm(seg)
                offset = rng.normal(size=3)
                offset -= offset.dot(seg) / max(length**2, 1e-9) * seg  # perpendicular
                norm = np.linalg.norm(offset)
                if norm > 0:
                    offset *= length * roughness * rng.uniform(0.3, 1.0) / norm
                new += [(p + q) / 2 + offset, q]
            points = np.array(new)
        return points

    main = jag(top, bottom, 0.22, levels)
    paths = [(main, 1.0)]
    total = np.linalg.norm(bottom - top)
    for _ in range(int(rng.integers(2, 5))):
        i = int(rng.integers(len(main) // 10, max(len(main) * 6 // 10, len(main) // 10 + 1)))
        start = main[i]
        down = (bottom - top) / max(total, 1e-9)
        lateral = rng.normal(size=3)
        lateral[2] = 0.0
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        direction = down * rng.uniform(0.4, 0.8) + lateral * rng.uniform(0.5, 1.0)
        direction /= np.linalg.norm(direction)
        end = start + direction * total * rng.uniform(0.15, 0.4)
        paths.append((jag(start, end, 0.25, levels - 2), 0.35))
    return paths


class LightningLayer:
    def __init__(self, ctx: moderngl.Context, shaders) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.active: list[tuple[Strike, float, np.ndarray, list]] = []  # strike, flash, top, paths
        self._buffer: moderngl.Buffer | None = None
        self._vaos: dict[int, moderngl.VertexArray] = {}
        self._vertices: np.ndarray | None = None

    def update(self, store, animation, time: float, camera, frame, terrain_data) -> None:
        """Active strikes of this moment with their bolt geometry (ENU)."""
        self.active = []
        self._vertices = None
        if store is None or store["lightning.rate"] <= 0.0 and not _animated(animation):
            return
        if not (store["weather.enabled"] or store["rain.enabled"]):
            return

        def rate_at(t: float) -> float:
            curve = animation.curves.get("lightning.rate") if animation is not None else None
            return (
                float(curve.evaluate(t))
                if curve is not None and curve.keys
                else store["lightning.rate"]
            )

        seed = int(store["lightning.seed"])
        radius = store["lightning.radius_km"] * 1000.0
        heading = math.radians(camera.heading)
        cloud_base = store["rain.cloud_base"] if store["rain.enabled"] else 3500.0
        heights = getattr(terrain_data, "heights_at", None) if frame is not None else None
        for strike in active_strikes(rate_at, seed, time):
            x, y, ground = self._place(strike, camera, heading, radius, cloud_base, frame,
                                       heights)  # fmt: skip
            top_h = max(cloud_base, ground + 300.0)
            bottom = np.array([x, y, ground])
            top = np.array([x + (strike.rand(7) - 0.5) * 1500.0,
                            y + (strike.rand(8) - 0.5) * 1500.0, top_h])  # fmt: skip
            paths = bolt_paths(top, bottom, strike) if store["lightning.bolts"] else []
            self.active.append((strike, flash_intensity(strike, time), top, paths))

    @staticmethod
    def _place(strike, camera, heading, radius, cloud_base, frame, heights):
        """A strike position in view: up to 8 deterministic candidates; the first whose
        ground lies below the cloud base and whose lower half is visible from the camera."""
        cam = np.asarray(camera.position, dtype=np.float64)
        fallback = None
        for attempt in range(8):
            salt = 10 + attempt * 3
            angle = heading + (strike.rand(salt) - 0.5) * math.radians(100.0)
            dist = max(1500.0, radius * (0.2 + 0.8 * strike.rand(salt + 1)))
            x = cam[0] + math.sin(angle) * dist
            y = cam[1] + math.cos(angle) * dist
            if heights is None:
                return x, y, 1500.0
            # terrain along the line of sight to the point halfway up the bolt
            t = np.linspace(0.0, 1.0, 24)[1:-1]
            ground_enu = np.array([x, y, 0.0])
            lat, lon, _ = frame.enu_to_geodetic(ground_enu)
            ground = heights(np.array([float(lon)]), np.array([float(lat)]))[0]
            if not np.isfinite(ground):
                continue
            mid = np.array([x, y, ground + 0.5 * max(cloud_base - ground, 300.0)])
            line = cam[None, :] + (mid - cam)[None, :] * t[:, None]
            la, lo, _ = frame.enu_to_geodetic(line)
            terrain = heights(np.asarray(lo), np.asarray(la))
            visible = bool(np.all(~np.isfinite(terrain) | (terrain < line[:, 2] - 20.0)))
            candidate = (float(x), float(y), float(ground))
            if fallback is None:
                fallback = candidate
            if visible and ground < cloud_base - 200.0:
                return candidate
        return fallback or (float(cam[0]), float(cam[1]) + radius, 1500.0)

    def flash(self) -> float:
        return sum(f for _, f, _, _ in self.active)

    def flash_uniforms(self, store, camera) -> dict[str, object]:
        """Point lights in the clouds (camera-relative) for the cloud pass."""
        lights = sorted(self.active, key=lambda a: -a[1])[:2]
        color = np.array(store["lightning.color"]) ** 2.2 * store["lightning.intensity"] * 5.0
        pos = [(*(top - camera.position), 3000.0) for _, _, top, _ in lights]
        cols = [tuple(color * f) for _, f, _, _ in lights]
        while len(pos) < 2:
            pos.append((0.0, 0.0, 0.0, 1.0))
            cols.append((0.0, 0.0, 0.0))
        return {
            "u_flash_count": len(lights),
            "u_flash_pos": [tuple(float(v) for v in p) for p in pos],
            "u_flash_color": [tuple(float(v) for v in c) for c in cols],
        }

    def _geometry(self, camera) -> np.ndarray:
        if self._vertices is not None:
            return self._vertices
        parts = []
        for _, flash, _, paths in self.active:
            for points, strength in paths:
                rel = points - camera.position
                a, b = rel[:-1], rel[1:]
                n = len(a)
                # per segment: two triangles; (end, side) of the six vertices
                quad = np.zeros((n, 6, 9), dtype=np.float32)
                quad[:, :, 0:3] = a[:, None, :]
                quad[:, :, 3:6] = b[:, None, :]
                corners = ((0, -1), (0, 1), (1, 1), (0, -1), (1, 1), (1, -1))
                for j, (end, side) in enumerate(corners):
                    quad[:, j, 6] = end
                    quad[:, j, 7] = side
                quad[:, :, 8] = strength * min(flash, 3.0)
                parts.append(quad.reshape(-1, 9))
        self._vertices = np.concatenate(parts) if parts else np.zeros((0, 9), np.float32)
        return self._vertices

    def render(self, camera, view_proj, width: int, height: int, store, glow: bool) -> None:
        if not self.active:
            return
        vertices = self._geometry(camera)
        if not len(vertices):
            return
        program = self.shaders.get("bolt", defines={"GLOW_PASS": 1} if glow else None)
        data = vertices.tobytes()
        if self._buffer is None or self._buffer.size < len(data):
            if self._buffer is not None:
                self._buffer.release()
                for vao in self._vaos.values():
                    vao.release()
                self._vaos.clear()
            self._buffer = self.ctx.buffer(reserve=max(len(data), 1 << 16) * 2)
        self._buffer.write(data)
        vao = self._vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(
                program,
                [
                    (
                        self._buffer,
                        "3f 3f 1f 1f 1f",
                        "in_a",
                        "in_b",
                        "in_end",
                        "in_side",
                        "in_strength",
                    )
                ],  # fmt: skip
            )
            self._vaos[program.glo] = vao
        color = np.array(store["lightning.color"]) ** 2.2 * store["lightning.intensity"]
        for name, value in (
            ("u_viewport", (float(width), float(height))),
            ("u_log_depth_coef", camera.log_depth_coef),
            ("u_width_px", 2.2 * height / 1080.0),
            ("u_color", tuple(float(c) for c in color)),
        ):
            if name in program:
                program[name] = value
        program["u_view_proj"].write(view_proj)
        self.ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST)
        # depth test only: clouds and rain use the scene depth (the depth mask is a
        # framebuffer setting in moderngl)
        fbo = self.ctx.fbo
        fbo.depth_mask = False
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        vao.render(moderngl.TRIANGLES, vertices=len(vertices))
        fbo.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        if self._buffer is not None:
            self._buffer.release()
        for vao in self._vaos.values():
            vao.release()


def _animated(animation) -> bool:
    return animation is not None and animation.is_animated("lightning.rate")


def lightning_properties() -> list:
    from earthling.render.parameters import boolean, color, flt, integer, section

    return section(
        "Weather: Rain & Thunder",
        flt("lightning.rate", "Lightning", 0.0, 0.0, 120.0, step=1.0, decimals=0,
            unit="/min", tooltip="Strikes per minute (needs rain or weather on)"),
        flt("lightning.intensity", "Lightning brightness", 1.0, 0.0, 5.0),
        flt("lightning.radius_km", "Lightning distance", 12.0, 1.0, 50.0, step=0.5, unit="km",
            tooltip="Strikes happen up to this far from the camera, mostly in view"),
        integer("lightning.seed", "Lightning seed", 1, 0, 9999, animatable=False,
                tooltip="Another seed gives other strike times and bolt shapes"),
        boolean("lightning.bolts", "Show bolts", True),
        color("lightning.color", "Lightning colour", (0.78, 0.83, 1.0)),
    )  # fmt: skip
