import numpy as np
from PIL import Image

from earthling.core.photos import Photo
from earthling.core.scene import Scene
from earthling.render.photos import PhotoLayer, load_photo
from earthling.render.shader_library import ShaderLibrary


def _picture(path, size=(400, 300)):
    """Left half red, right half blue."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:, : size[0] // 2] = (220, 30, 30)
    img[:, size[0] // 2 :] = (30, 30, 220)
    Image.fromarray(img).save(path)


def test_scene_round_trip(tmp_path):
    scene = Scene()
    scene.set_photos([Photo("km10", "10 km", "Images/10.jpg")])
    scene.animation.set_key("photo.km10.opacity", 1.0, 0.0)
    scene.animation.set_key("photo.km10.opacity", 1.5, 1.0)
    scene.store.set("photo.km10.x", 0.2)
    scene.save(tmp_path / "s.json")
    other = Scene()
    assert other.load(tmp_path / "s.json") == []
    assert other.photos == [Photo("km10", "10 km", "Images/10.jpg")]
    assert other.store["photo.km10.x"] == 0.2
    assert len(other.animation.curves["photo.km10.opacity"].keys) == 2
    assert "photos" not in Scene().to_json()  # (no photos: older readers see nothing new)


def test_exif_orientation_and_size(tmp_path):
    _picture(tmp_path / "a.png", (400, 300))
    img = Image.open(tmp_path / "a.png")
    exif = img.getexif()
    exif[0x0112] = 6  # stored sideways: rotate 90 degrees clockwise for display
    img.save(tmp_path / "rot.jpg", exif=exif)
    rgb = load_photo(tmp_path / "rot.jpg")
    assert rgb.shape == (400, 300, 3)  # upright: portrait
    assert rgb[5, 150, 0] > 150 and rgb[-5, 150, 2] > 150  # red on top, blue at the bottom
    assert load_photo(tmp_path / "a.png", max_px=100).shape == (75, 100, 3)


def test_layer_draws_frame_and_fades(gl_ctx, tmp_path):
    _picture(tmp_path / "p.png")
    scene = Scene()
    scene.set_photos([Photo("p", "p", "p.png")])
    store = scene.store
    layer = PhotoLayer(gl_ctx, ShaderLibrary(gl_ctx))
    layer.set_photos(scene.photos, tmp_path)
    fbo = gl_ctx.simple_framebuffer((960, 540))

    def draw(**props):
        for pid, value in props.items():
            store.set("photo.p." + pid, value)
        fbo.use()
        fbo.clear(0.0, 0.5, 0.0, 1.0)
        layer.render(fbo, 960, 540, store)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(540, 960, 3)[::-1]
        return img.astype(int)

    img = draw()
    assert layer.last_drawn == [] and (img[..., 1] == 127).all()  # opacity 0 by default
    img = draw(opacity=1.0, x=0.5, y=0.5, height=0.4, border=10.0, shadow=0.0)
    assert layer.last_drawn == ["p"]
    # 216 px high, 288 px wide around (480, 270); the border (10 px at 1080p = 5 px here)
    assert img[270, 400, 0] > 180 and img[270, 400, 2] < 80  # left: red
    assert img[270, 560, 2] > 180 and img[270, 560, 0] < 80  # right: blue
    assert img[270, 480 - 144 - 3].min() > 220  # the white frame
    assert img[270, 480 - 144 - 12, 1] == 127 and img[100, 480, 1] == 127  # untouched outside
    img = draw(opacity=0.5)
    assert 90 < img[270, 400, 0] < 150  # half faded over the green
    img = draw(opacity=1.0, zoom=2.0)
    assert img[270, 400, 0] > 180 and img[270, 470, 0] > 180  # zoomed: red reaches the centre
    img = draw(zoom=1.0, shadow=1.0)
    assert img[270 + 108 + 5 + 6, 480, 1] < 110  # the shadow darkens below the frame
