"""Render the README screenshots from the demo project into docs/images/.

    uv run earthling fetch examples/alps_demo      # once: imagery and elevation data
    uv run python tools/make_readme_images.py [name ...]

Every image is a small scene built in code (camera, time of day, layers, weather, POIs) and
rendered offscreen at full quality, like a video frame.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from earthling.core.animation import Interp
from earthling.core.config import Project
from earthling.core.pois import Poi
from earthling.core.scene import Scene
from earthling.core.session import Session
from earthling.export.frames import FrameRenderer
from earthling.render.camera_rig import look_orientation
from earthling.render.renderer import Renderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
SIZE = (1600, 900)

session = Session(Project.load(ROOT / "examples" / "alps_demo"))
frame = session.frame


def enu(lat: float, lon: float, h: float) -> np.ndarray:
    return np.asarray(frame.geodetic_to_enu(lat, lon, h), dtype=np.float64)


def pose(eye, target) -> tuple:
    h, p, r = look_orientation(np.asarray(eye), np.asarray(target))
    return (float(eye[0]), float(eye[1]), float(eye[2]), h, p, r)


def new_scene(eye, target, **props) -> Scene:
    scene = Scene()
    scene.set_dynamic("track_groups", session.track_group_properties())
    s = scene.store
    s.set("camera.mode", "keys")
    s.set("export.samples", "8")
    s.set("tracks.casing", 2.5)
    s.set("tracks.color_mode", "single")
    s.set("tracks.color", (1.0, 0.42, 0.12))
    scene.animation.duration = 10.0
    scene.animation.set_key("camera.pose", 0.0, pose(eye, target), Interp.LINEAR)
    for pid, value in props.items():
        s.set(pid.replace("__", "."), value)
    return scene


def render(scene: Scene, time: float = 0.0, size=SIZE) -> Image.Image:
    ctx = moderngl.create_standalone_context(require=430)
    renderer = Renderer(ctx)
    renderer.set_scene(frame, session.all_tracks)
    renderer.set_terrain_source(session.terrain_data(), session.terrain_nodes())
    renderer.store = scene.store
    renderer.timezone = session.config.project.timezone
    renderer.set_labels(session.labels, session)
    renderer.pois.set_pois(scene.pois, None)
    renderer.hud.attribution = session.attribution()
    frames = FrameRenderer(renderer, scene.animation)
    image = frames.render(time, *size)
    renderer.terrain.shutdown()
    ctx.release()
    return Image.fromarray(image)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jpg"
    image.convert("RGB").save(path, quality=88, optimize=True, progressive=True)
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")


def caption(image: Image.Image, text: str) -> Image.Image:
    """A small label in the corner (for the layer grid)."""
    from PIL import ImageDraw, ImageFont

    img = image.copy()
    d = ImageDraw.Draw(img)
    font_path = ROOT / "src" / "earthling" / "render" / "fonts"
    fonts = sorted(font_path.glob("*.ttf"))
    font = ImageFont.truetype(str(fonts[0]), 34) if fonts else ImageFont.load_default()
    x, y = 24, img.height - 64
    w = d.textlength(text, font=font)
    d.rounded_rectangle((x - 12, y - 8, x + w + 12, y + 44), radius=10, fill=(0, 0, 0, 150))
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


# --- the images --------------------------------------------------------------------------------
def hero() -> None:
    """Golden hour above the Col de Balme, looking down the Chamonix valley to Mont Blanc."""
    scene = new_scene(enu(46.075, 6.99, 3200.0), enu(45.975, 6.925, 1250.0),
                      sun__datetime=dt.datetime(2026, 7, 1, 20, 50), progress__head=0.5,
                      stats__visible=True, stats__show_time=True, labels__max_count=14,
                      labels__min_prominence=300.0, view__fov=55.0)  # fmt: skip
    save(render(scene), "hero")


def layers() -> None:
    """The same view with four texture layers."""
    eye, target = enu(46.005, 6.90, 3900.0), enu(46.04, 6.985, 1900.0)
    base = {"sun__datetime": dt.datetime(2026, 7, 1, 11, 0), "labels__visible": False,
            "haze__aerial": 0.6, "view__fov": 50.0}  # fmt: skip
    variants = [
        ("Satellite", {}),
        ("Border map", {"layers__a": "borders", "borders__overlay": True,
                        "layer_borders__land": (0.42, 0.47, 0.58), "layer_borders__gain": 1.0,
                        "borders__color": (0.35, 0.85, 1.0), "borders__glow_strength": 1.2}),
        ("Avalanche slope classes", {"layers__a": "slope"}),
        ("Contour map", {"layers__a": "contours", "contours__overlay": False}),
    ]  # fmt: skip
    tiles = []
    for label, props in variants:
        scene = new_scene(eye, target, **base, **props)
        tiles.append(caption(render(scene, size=(800, 450)), label))
    grid = Image.new("RGB", (1600, 900))
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % 2) * 800, (i // 2) * 450))
    save(grid, "layers")


def weather() -> None:
    """A sea of fog in the valleys and drifting clouds below the ridges, seen from above."""
    scene = new_scene(enu(46.0, 6.925, 3400.0), enu(46.055, 7.02, 1500.0),
                      sun__datetime=dt.datetime(2026, 7, 1, 8, 30), weather__enabled=True,
                      weather__layer1__base=1000.0, weather__layer1__thickness=700.0,
                      weather__layer1__coverage=0.75, weather__layer1__density=0.5,
                      weather__layer2__enabled=True, weather__layer2__base=2300.0,
                      weather__layer2__thickness=500.0, weather__layer2__coverage=0.3,
                      weather__layer2__scale=2500.0, labels__visible=False,
                      view__fov=55.0)  # fmt: skip
    save(render(scene, time=3.0), "weather")


def storm() -> None:
    """A thunderstorm at night with rain and lightning over the Fenêtre d'Arpette."""
    from earthling.render.lightning import strikes_between

    scene = new_scene(enu(46.07, 7.02, 2200.0), enu(46.035, 7.06, 2300.0),
                      sun__datetime=dt.datetime(2026, 7, 1, 1, 30), rain__enabled=True,
                      rain__cloud_base=3000.0, rain__intensity=0.9, lightning__rate=30.0,
                      lightning__seed=3, light__night_ambient=3.0, post__exposure=1.0,
                      labels__visible=False, progress__head=0.8, view__fov=55.0)  # fmt: skip
    strikes = strikes_between(lambda _t: 30.0, 3, 0.0, 10.0)
    save(render(scene, time=strikes[0].start + 0.02), "storm")


def pois() -> None:
    """Animated POI icons along day 2 at midday."""
    scene = new_scene(enu(46.085, 7.095, 3900.0), enu(46.042, 7.055, 1700.0),
                      sun__datetime=dt.datetime(2026, 7, 2, 13, 0), progress__head=1.0,
                      labels__visible=False, view__fov=55.0)  # fmt: skip
    scene.set_pois([
        Poi("cafe", "Trient", 7.003, 46.057, "builtin:coffee", caption="Kaffeepause"),
        Poi("pass", "Fenêtre d'Arpette", 7.043, 46.038, "builtin:flag",
            caption="Fenêtre d'Arpette\n2665 m"),
        Poi("photo", "View", 7.068, 46.033, "builtin:camera"),
        Poi("camp", "Champex", 7.116, 46.03, "builtin:tent", caption="Champex"),
    ])  # fmt: skip
    for poi in scene.pois:
        scene.store.set(f"poi.{poi.id}.caption_size", 26.0)
    save(render(scene, time=1.5), "pois")


def night() -> None:
    """Stars over the Col de Balme, the track glowing in the dark."""
    scene = new_scene(enu(46.00, 6.93, 2400.0), enu(46.035, 6.97, 2300.0),
                      sun__datetime=dt.datetime(2026, 7, 1, 23, 50), progress__head=1.0,
                      light__night_ambient=5.0, post__exposure=0.8, labels__visible=False,
                      view__fov=60.0,
                      tracks__glow=3.0)  # fmt: skip
    save(render(scene), "night")


def gui() -> None:
    """The application window with the demo project and its showcase scene."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from earthling.app.main_window import MainWindow
    from earthling.app.viewport import configure_default_surface_format

    configure_default_surface_format()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    window.resize(1600, 1000)
    window.show()
    project = ROOT / "examples" / "alps_demo"
    window.open_project(project)
    window.load_scene(project / "showcase.json")
    window.timeline.set_time(4.0)

    def grab() -> None:
        window.scene.mark_dirty(False)
        window.undo_stack.setClean()
        image = window.grab().toImage()
        path = OUT / "gui.png"
        OUT.mkdir(parents=True, exist_ok=True)
        image.save(str(path))
        save(Image.open(path), "gui")
        path.unlink()
        app.quit()

    QTimer.singleShot(12000, grab)  # let the terrain load
    app.exec()


IMAGES = {"hero": hero, "layers": layers, "weather": weather, "storm": storm, "pois": pois,
          "night": night, "gui": gui}  # fmt: skip

if __name__ == "__main__":
    for name in sys.argv[1:] or IMAGES:
        IMAGES[name]()
