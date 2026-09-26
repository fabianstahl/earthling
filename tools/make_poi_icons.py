"""Draw the built-in placeholder POI icons into src/earthling/assets/icons/.

    python tools/make_poi_icons.py

All icons are drawn at 4x and downsampled (anti-aliasing). Replace them with your own artwork
by pointing a POI's icon at any PNG / GIF / APNG file.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "src" / "earthling" / "assets" / "icons"
SIZE = 256
SS = 4  # supersampling
S = SIZE * SS


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def finish(img: Image.Image) -> Image.Image:
    return img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def shadow(img: Image.Image, radius: int = 10) -> Image.Image:
    """Soft drop shadow under the shape."""
    alpha = img.getchannel("A")
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shade.putalpha(alpha.point(lambda a: int(a * 0.45)))
    shade = shade.filter(ImageFilter.GaussianBlur(radius * SS))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(shade, (0, 4 * SS))
    out.alpha_composite(img)
    return out


def badge(ring: tuple[int, int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img, d = canvas()
    m = 14 * SS
    d.ellipse((m, m, S - m, S - m), fill=(*ring, 255))
    m2 = m + 14 * SS
    d.ellipse((m2, m2, S - m2, S - m2), fill=(255, 255, 255, 255))
    return img, d


def p(x: float, y: float) -> tuple[float, float]:
    """Icon coordinates 0..1 -> canvas pixels."""
    return x * S, y * S


def draw_pin() -> Image.Image:
    img, d = canvas()
    cx, top, r = 0.5, 0.1, 0.3
    d.ellipse((*p(cx - r, top), *p(cx + r, top + 2 * r)), fill=(226, 60, 50, 255))
    d.polygon([p(cx - r * 0.86, top + r * 1.5), p(cx + r * 0.86, top + r * 1.5), p(cx, 0.95)],
              fill=(226, 60, 50, 255))  # fmt: skip
    d.ellipse((*p(cx - 0.11, top + r - 0.11), *p(cx + 0.11, top + r + 0.11)),
              fill=(255, 255, 255, 255))  # fmt: skip
    return finish(shadow(img))


def draw_coffee_frame(phase: float) -> Image.Image:
    """phase 0..1: the cup lifts and tilts for a sip, steam rises all the time."""
    img, d = badge((122, 78, 48))
    sip = max(0.0, math.sin(phase * 2.0 * math.pi)) ** 2  # lift during the first half
    lift = 0.07 * sip
    tilt = -18.0 * sip
    cup = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    c = ImageDraw.Draw(cup)
    body = [p(0.33, 0.47), p(0.63, 0.47), p(0.6, 0.72), p(0.36, 0.72)]
    c.polygon(body, fill=(245, 245, 240, 255), outline=(90, 60, 40, 255), width=6 * SS)
    c.rectangle((*p(0.335, 0.475), *p(0.625, 0.52)), fill=(111, 70, 40, 255))  # coffee
    c.arc((*p(0.56, 0.52), *p(0.72, 0.66)), start=-90, end=90, fill=(90, 60, 40, 255),
          width=7 * SS)  # fmt: skip
    c.ellipse((*p(0.28, 0.72), *p(0.7, 0.78)), fill=(210, 205, 195, 255))  # saucer
    cup = cup.rotate(tilt, center=p(0.48, 0.6), resample=Image.Resampling.BICUBIC)
    img.alpha_composite(cup, (0, int(-lift * S)))
    # three steam wisps rising and fading
    for k in range(3):
        t = (phase + k / 3.0) % 1.0
        x0 = 0.4 + 0.08 * k
        points = []
        for j in range(12):
            u = j / 11.0
            y = 0.44 - lift - 0.16 * u - 0.06 * t
            x = x0 + 0.025 * math.sin((u * 2.0 + t * 2.0) * math.pi)
            points.append(p(x, y))
        alpha = int(200 * math.sin(t * math.pi))
        d.line(points, fill=(150, 150, 150, alpha), width=5 * SS, joint="curve")
    return finish(shadow(img))


def draw_turnaround() -> Image.Image:
    img, d = canvas()
    m = 12 * SS
    d.ellipse((m, m, S - m, S - m), fill=(210, 30, 30, 255))
    m2 = m + 22 * SS
    d.ellipse((m2, m2, S - m2, S - m2), fill=(255, 255, 255, 255))
    # U-turn arrow: down the right, around, up the left with an arrow head
    w = 16 * SS
    d.line([p(0.62, 0.72), p(0.62, 0.45)], fill=(20, 20, 20, 255), width=w)
    d.arc((*p(0.38, 0.33), *p(0.62, 0.57)), start=180, end=360, fill=(20, 20, 20, 255), width=w)
    d.line([p(0.38, 0.45), p(0.38, 0.62)], fill=(20, 20, 20, 255), width=w)
    d.polygon([p(0.29, 0.6), p(0.47, 0.6), p(0.38, 0.74)], fill=(20, 20, 20, 255))
    return finish(shadow(img))


def draw_tent() -> Image.Image:
    img, d = badge((46, 130, 70))
    d.polygon([p(0.5, 0.3), p(0.76, 0.7), p(0.24, 0.7)], fill=(46, 130, 70, 255))
    d.polygon([p(0.5, 0.42), p(0.58, 0.7), p(0.42, 0.7)], fill=(255, 255, 255, 255))
    d.line([p(0.2, 0.71), p(0.8, 0.71)], fill=(60, 60, 60, 255), width=5 * SS)
    return finish(shadow(img))


def draw_camera() -> Image.Image:
    img, d = badge((40, 110, 200))
    d.rounded_rectangle((*p(0.28, 0.38), *p(0.72, 0.66)), radius=18 * SS, fill=(40, 40, 50, 255))
    d.rectangle((*p(0.42, 0.32), *p(0.58, 0.39)), fill=(40, 40, 50, 255))
    d.ellipse((*p(0.41, 0.42), *p(0.59, 0.6)), fill=(230, 230, 235, 255))
    d.ellipse((*p(0.45, 0.46), *p(0.55, 0.56)), fill=(40, 110, 200, 255))
    return finish(shadow(img))


def draw_flag() -> Image.Image:
    img, d = badge((240, 140, 20))
    d.line([p(0.38, 0.28), p(0.38, 0.74)], fill=(60, 60, 60, 255), width=7 * SS)
    d.polygon([p(0.39, 0.29), p(0.7, 0.37), p(0.39, 0.47)], fill=(240, 140, 20, 255))
    return finish(shadow(img))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_pin().save(OUT / "pin.png")
    draw_turnaround().save(OUT / "turnaround.png")
    draw_tent().save(OUT / "tent.png")
    draw_camera().save(OUT / "camera.png")
    draw_flag().save(OUT / "flag.png")
    frames = [draw_coffee_frame(i / 24) for i in range(24)]
    # animated PNG (APNG); disposal "none" + blend "source" stores complete frames
    frames[0].save(OUT / "coffee.png", save_all=True, append_images=frames[1:], duration=70,
                   loop=0, disposal=0, blend=0, format="PNG")  # fmt: skip
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
