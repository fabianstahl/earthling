"""Generate examples/alps_demo/showcase.json: a 30 s scene using every animated feature.

* 0-14 s  camera flight through 5 keys over the Chamonix valley, day turning to dusk
* 8-12 s  satellite -> slope map crossfade and back (layer B switches while hidden)
* 14-20 s follow-the-hiker camera, borders overlay fades in, track glows
* 20-30 s look-at shot on Mont Blanc while night falls, exposure opens up, valley fog rises

Open it with File > Open Scene... in the alps_demo project.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from earthling.core.animation import Interp
from earthling.core.config import Project
from earthling.core.scene import Scene
from earthling.core.session import Session

DEMO = Path(__file__).resolve().parents[1] / "examples" / "alps_demo"

# (time, lat, lon, height above sea level, heading, pitch)
CAMERA_KEYS = [
    (0.0, 45.880, 6.840, 3300, 40, -18),
    (3.5, 45.935, 6.865, 3100, 50, -22),
    (7.0, 45.975, 6.905, 3400, 60, -20),
    (10.5, 46.010, 6.950, 3600, 90, -24),
    (14.0, 46.030, 7.000, 3900, 140, -28),
]


def build(frame) -> Scene:
    scene = Scene()
    anim, store = scene.animation, scene.store
    anim.duration, anim.fps = 30.0, 60.0
    store.set("view.show_outlines", False)
    store.set("camera.transition", 2.0)
    for t, lat, lon, h, heading, pitch in CAMERA_KEYS:
        x, y, z = frame.geodetic_to_enu(lat, lon, h)
        anim.set_key("camera.pose", t, (x, y, z, heading, pitch, 0.0), Interp.LINEAR)
    anim.curves["camera.pose"].keys[0].interp = Interp.EASE_IN
    anim.curves["camera.pose"].keys[-2].interp = Interp.EASE_OUT
    for t, mode in ((0.0, "keys"), (14.0, "follow"), (20.0, "look_at")):
        anim.set_key("camera.mode", t, mode)
    mont_blanc = frame.geodetic_to_enu(45.8326, 6.8652, 4808.0)
    store.set("camera.target", tuple(float(v) for v in mont_blanc))
    for t, when, interp in ((0.0, datetime(2026, 7, 1, 17, 30), Interp.LINEAR),
                            (20.0, datetime(2026, 7, 1, 21, 20), Interp.EASE_OUT),
                            (30.0, datetime(2026, 7, 1, 22, 30), Interp.LINEAR)):  # fmt: skip
        anim.set_key("sun.datetime", t, when, interp)
    anim.set_key("progress.head", 0.0, 0.0, Interp.EASE_IN_OUT)
    anim.set_key("progress.head", 26.0, 1.0)
    # texture layers: B (hidden) switches to slope before the crossfade, borders later
    for t, value in ((0.0, "slope"), (15.0, "borders")):
        anim.set_key("layers.b", t, value)
    for t, value in ((0.0, 0.0), (8.0, 0.0), (9.5, 1.0), (11.0, 1.0), (12.5, 0.0)):
        anim.set_key("layers.mix", t, value, Interp.EASE_IN_OUT)
    for t, value in ((0.0, False), (15.0, True)):
        anim.set_key("borders.overlay", t, value)
    for t, value in ((20.0, 0.0), (26.0, 1.5)):
        anim.set_key("post.exposure", t, value, Interp.EASE_IN_OUT)
    store.set("fog.enabled", True)
    store.set("fog.base", 1100.0)
    for t, value in ((0.0, 0.0), (20.0, 0.0), (28.0, 1.5)):
        anim.set_key("fog.density", t, value, Interp.EASE_IN_OUT)
    for t, value in ((0.0, 1.0), (22.0, 1.0), (28.0, 4.0)):
        anim.set_key("tracks.glow", t, value)
    # overlays: labels (on by default), stats fading in once the hike starts, credits
    store.set("stats.visible", True)
    store.set("stats.show_time", True)
    store.set("stats.attribution", True)
    for t, value in ((0.0, 0.0), (1.0, 0.0), (2.5, 1.0)):
        anim.set_key("stats.opacity", t, value, Interp.EASE_IN_OUT)
    return scene


def main() -> None:
    session = Session(Project.load(DEMO))
    scene = build(session.frame)
    path = scene.save(DEMO / "showcase.json")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
