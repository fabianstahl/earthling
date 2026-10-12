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


def draw_power() -> Image.Image:
    """Electricity / charging: a lightning bolt."""
    img, d = badge((245, 185, 20))
    d.polygon([p(0.56, 0.24), p(0.32, 0.54), p(0.47, 0.54), p(0.42, 0.78), p(0.68, 0.44),
               p(0.53, 0.44), p(0.6, 0.24)], fill=(30, 30, 35, 255))  # fmt: skip
    return finish(shadow(img))


def _cloud(d: ImageDraw.ImageDraw, fill) -> None:
    for x0, y0, x1, y1 in ((0.26, 0.36, 0.5, 0.56), (0.38, 0.26, 0.66, 0.54),
                           (0.54, 0.34, 0.76, 0.56)):  # fmt: skip
        d.ellipse((*p(x0, y0), *p(x1, y1)), fill=fill)
    d.rounded_rectangle((*p(0.3, 0.44), *p(0.72, 0.56)), radius=10 * SS, fill=fill)


def draw_storm_frame(phase: float) -> Image.Image:
    """Bad weather: a dark cloud with falling rain and a flashing bolt (animated)."""
    img, d = badge((70, 90, 125))
    flash = phase < 0.08 or 0.14 < phase < 0.2  # double flash, then a pause
    _cloud(d, (95, 105, 125, 255) if not flash else (150, 160, 180, 255))
    for i, x in enumerate((0.34, 0.46, 0.58, 0.68)):
        y = 0.6 + ((phase * 2.0 + i * 0.27) % 1.0) * 0.16
        d.line([p(x, y), p(x - 0.025, y + 0.06)], fill=(60, 120, 220, 255), width=5 * SS)
    if flash:
        d.polygon([p(0.52, 0.52), p(0.44, 0.66), p(0.5, 0.66), p(0.46, 0.78), p(0.58, 0.62),
                   p(0.52, 0.62), p(0.56, 0.52)], fill=(255, 215, 40, 255))  # fmt: skip
    return finish(shadow(img))


def draw_supermarket() -> Image.Image:
    """Supermarket: a shopping cart."""
    img, d = badge((30, 120, 200))
    ink, w = (30, 30, 35, 255), 7 * SS
    d.line([p(0.24, 0.33), p(0.32, 0.33), p(0.4, 0.6), p(0.7, 0.6)], fill=ink, width=w,
           joint="curve")  # fmt: skip
    d.polygon([p(0.34, 0.38), p(0.76, 0.38), p(0.7, 0.54), p(0.39, 0.54)], fill=ink)
    for x in (0.44, 0.65):
        d.ellipse((*p(x - 0.045, 0.64), *p(x + 0.045, 0.73)), fill=ink)
    return finish(shadow(img))


def draw_couscous() -> Image.Image:
    """A meal: a bowl of couscous with vegetables and a spoon."""
    img, d = badge((200, 120, 40))
    d.pieslice((*p(0.24, 0.3), *p(0.76, 0.78)), 0, 180, fill=(170, 90, 40, 255))  # bowl
    d.ellipse((*p(0.26, 0.47), *p(0.74, 0.61)), fill=(236, 196, 110, 255))  # couscous
    d.ellipse((*p(0.34, 0.4), *p(0.66, 0.56)), fill=(236, 196, 110, 255))
    for x, y, c in ((0.42, 0.46, (220, 60, 40)), (0.55, 0.44, (60, 150, 60)),
                    (0.5, 0.5, (240, 140, 30)), (0.6, 0.5, (220, 60, 40))):  # fmt: skip
        d.ellipse((*p(x - 0.03, y - 0.03), *p(x + 0.03, y + 0.03)), fill=(*c, 255))
    d.line([p(0.62, 0.42), p(0.72, 0.24)], fill=(120, 120, 130, 255), width=6 * SS)  # spoon
    d.ellipse((*p(0.69, 0.2), *p(0.76, 0.27)), fill=(120, 120, 130, 255))
    return finish(shadow(img))


def draw_summit_decision() -> Image.Image:
    """A decision: a signpost pointing up to a summit."""
    img, d = badge((110, 70, 160))
    d.polygon([p(0.22, 0.72), p(0.46, 0.34), p(0.56, 0.48), p(0.63, 0.4), p(0.8, 0.72)],
              fill=(120, 130, 145, 255))  # fmt: skip
    d.polygon([p(0.46, 0.34), p(0.52, 0.44), p(0.41, 0.42)], fill=(255, 255, 255, 255))  # snow
    d.line([p(0.34, 0.8), p(0.34, 0.32)], fill=(110, 70, 40, 255), width=6 * SS)  # post
    d.polygon(
        [p(0.34, 0.33), p(0.58, 0.33), p(0.64, 0.39), p(0.58, 0.45), p(0.34, 0.45)],
        fill=(240, 190, 40, 255),
    )  # sign pointing on
    d.polygon([p(0.46, 0.22), p(0.53, 0.3), p(0.39, 0.3)], fill=(110, 70, 160, 255))  # up
    return finish(shadow(img))


def draw_cold() -> Image.Image:
    """Freezing: a snowflake and a thermometer deep in the blue."""
    img, d = badge((60, 150, 230))
    ink, w = (60, 150, 230, 255), 5 * SS
    cx, cy, r = 0.43, 0.5, 0.2
    for k in range(3):  # six-armed snowflake
        a = math.pi / 3 * k
        dx, dy = math.cos(a) * r, math.sin(a) * r
        d.line([p(cx - dx, cy - dy), p(cx + dx, cy + dy)], fill=ink, width=w)
        for sgn in (-1, 1):
            ex, ey = cx + sgn * dx * 0.65, cy + sgn * dy * 0.65
            for b in (a + 0.6, a - 0.6):
                bx, by = math.cos(b) * 0.07 * sgn, math.sin(b) * 0.07 * sgn
                d.line([p(ex, ey), p(ex + bx, ey + by)], fill=ink, width=w - SS)
    d.rounded_rectangle((*p(0.66, 0.26), *p(0.74, 0.66)), radius=8 * SS, fill=(90, 90, 100, 255))
    d.rounded_rectangle((*p(0.68, 0.28), *p(0.72, 0.64)), radius=5 * SS, fill=(235, 240, 245, 255))
    d.ellipse((*p(0.63, 0.62), *p(0.77, 0.76)), fill=(90, 90, 100, 255))
    d.ellipse((*p(0.65, 0.64), *p(0.75, 0.74)), fill=(60, 150, 230, 255))
    d.rectangle((*p(0.685, 0.56), *p(0.715, 0.66)), fill=(60, 150, 230, 255))  # low reading
    return finish(shadow(img))


def _rotated(points, angle_deg: float, cx: float = 0.5, cy: float = 0.5):
    """Rotate icon coordinates (0..1) around (cx, cy), clockwise on screen."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [p(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca)
            for x, y in points]  # fmt: skip


def draw_airport() -> Image.Image:
    """Airport: an airliner seen from above, heading up and to the right."""
    img, d = badge((45, 95, 175))
    ink = (30, 45, 80, 255)
    plane = [(0.5, 0.2), (0.535, 0.26), (0.535, 0.43), (0.78, 0.56), (0.78, 0.61), (0.535, 0.54),
             (0.535, 0.7), (0.6, 0.76), (0.6, 0.8), (0.5, 0.77), (0.4, 0.8), (0.4, 0.76),
             (0.465, 0.7), (0.465, 0.54), (0.22, 0.61), (0.22, 0.56), (0.465, 0.43),
             (0.465, 0.26)]  # fmt: skip
    d.polygon(_rotated(plane, 45.0), fill=ink)
    return finish(shadow(img))


def draw_ferry() -> Image.Image:
    """Ferry / pier: a ship on the waves."""
    img, d = badge((20, 140, 175))
    ink = (25, 50, 75, 255)
    d.polygon([p(0.22, 0.55), p(0.78, 0.55), p(0.7, 0.67), p(0.3, 0.67)], fill=ink)  # hull
    d.rectangle((*p(0.34, 0.44), *p(0.66, 0.55)), fill=ink)  # deck house
    d.rectangle((*p(0.42, 0.36), *p(0.58, 0.44)), fill=ink)  # bridge
    d.rectangle((*p(0.6, 0.3), *p(0.65, 0.44)), fill=(220, 60, 50, 255))  # funnel
    for x in (0.38, 0.47, 0.56):  # windows
        d.rectangle((*p(x, 0.47), *p(x + 0.05, 0.51)), fill=(255, 255, 255, 255))
    wave = [p(0.24 + 0.52 * u, 0.73 + 0.022 * math.sin(u * 6.0 * math.pi)) for u in
            (i / 40.0 for i in range(41))]  # fmt: skip
    d.line(wave, fill=(20, 140, 175, 255), width=6 * SS, joint="curve")
    return finish(shadow(img))


def draw_bus() -> Image.Image:
    """Bus station: a coach from the front."""
    img, d = badge((205, 55, 60))
    ink = (40, 40, 48, 255)
    d.rounded_rectangle((*p(0.31, 0.25), *p(0.69, 0.7)), radius=16 * SS, fill=ink)
    d.rounded_rectangle((*p(0.35, 0.31), *p(0.65, 0.5)), radius=6 * SS,
                        fill=(225, 235, 245, 255))  # windscreen  # fmt: skip
    for x in (0.39, 0.61):  # headlights
        d.ellipse((*p(x - 0.035, 0.57), *p(x + 0.035, 0.64)), fill=(250, 220, 90, 255))
    for x in (0.37, 0.63):  # wheels
        d.rounded_rectangle((*p(x - 0.045, 0.68), *p(x + 0.045, 0.77)), radius=4 * SS, fill=ink)
    d.rectangle((*p(0.27, 0.33), *p(0.31, 0.42)), fill=ink)  # mirrors
    d.rectangle((*p(0.69, 0.33), *p(0.73, 0.42)), fill=ink)
    return finish(shadow(img))


def draw_hotel() -> Image.Image:
    """Hotel / hostel: a bed with a pillow."""
    img, d = badge((125, 70, 175))
    ink = (50, 35, 80, 255)
    d.rectangle((*p(0.24, 0.32), *p(0.29, 0.72)), fill=ink)  # headboard
    d.rectangle((*p(0.24, 0.55), *p(0.77, 0.63)), fill=ink)  # frame
    d.rectangle((*p(0.72, 0.55), *p(0.77, 0.72)), fill=ink)  # foot
    d.rounded_rectangle((*p(0.31, 0.43), *p(0.43, 0.54)), radius=6 * SS, fill=ink)  # pillow
    d.rounded_rectangle((*p(0.45, 0.42), *p(0.77, 0.54)), radius=8 * SS,
                        fill=(125, 70, 175, 255))  # blanket  # fmt: skip
    return finish(shadow(img))


def draw_taxi() -> Image.Image:
    """Taxi: a car from the side with a roof sign."""
    img, d = badge((245, 190, 25))
    ink = (35, 35, 40, 255)
    d.polygon([p(0.36, 0.44), p(0.62, 0.44), p(0.7, 0.53), p(0.78, 0.55), p(0.78, 0.65),
               p(0.22, 0.65), p(0.22, 0.55), p(0.29, 0.53)], fill=ink)  # fmt: skip
    d.polygon([p(0.38, 0.47), p(0.48, 0.47), p(0.48, 0.53), p(0.33, 0.53)],
              fill=(225, 235, 245, 255))  # fmt: skip
    d.polygon([p(0.51, 0.47), p(0.6, 0.47), p(0.66, 0.53), p(0.51, 0.53)],
              fill=(225, 235, 245, 255))  # fmt: skip
    d.rounded_rectangle((*p(0.43, 0.36), *p(0.56, 0.43)), radius=3 * SS, fill=ink)  # sign
    d.rectangle((*p(0.455, 0.38), *p(0.535, 0.41)), fill=(245, 190, 25, 255))
    for x in (0.34, 0.66):
        d.ellipse((*p(x - 0.065, 0.6), *p(x + 0.065, 0.73)), fill=ink)
        d.ellipse((*p(x - 0.03, 0.635), *p(x + 0.03, 0.695)), fill=(190, 190, 195, 255))
    return finish(shadow(img))


def _star(cx: float, cy: float, r_out: float, r_in: float):
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        r = r_out if k % 2 == 0 else r_in
        pts.append(p(cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def draw_attraction() -> Image.Image:
    """Sight / attraction: a star."""
    img, d = badge((225, 60, 120))
    d.polygon(_star(0.5, 0.52, 0.27, 0.115), fill=(225, 60, 120, 255))
    return finish(shadow(img))


def draw_finish() -> Image.Image:
    """Finish line: a chequered flag."""
    img, d = badge((40, 40, 46))
    d.line([p(0.34, 0.26), p(0.34, 0.76)], fill=(60, 60, 60, 255), width=7 * SS)
    x0, y0, cw, ch = 0.355, 0.28, 0.07, 0.065
    for row in range(4):
        for col in range(5):
            dark = (row + col) % 2 == 0
            wave = 0.012 * math.sin((col / 4.0) * math.pi)
            box = (*p(x0 + col * cw, y0 + row * ch - wave), *p(x0 + (col + 1) * cw,
                   y0 + (row + 1) * ch - wave))  # fmt: skip
            d.rectangle(box, fill=(30, 30, 35, 255) if dark else (255, 255, 255, 255))
    d.rectangle((*p(x0, y0 - 0.012), *p(x0 + 5 * cw, y0 + 4 * ch)), outline=(30, 30, 35, 255),
                width=2 * SS)  # fmt: skip
    return finish(shadow(img))


def draw_peak() -> Image.Image:
    """Summit: a mountain with a flag on top."""
    img, d = badge((70, 125, 75))
    d.polygon([p(0.2, 0.72), p(0.48, 0.32), p(0.58, 0.45), p(0.64, 0.4), p(0.8, 0.72)],
              fill=(80, 105, 85, 255))  # fmt: skip
    d.polygon([p(0.48, 0.32), p(0.55, 0.42), p(0.5, 0.4), p(0.45, 0.44), p(0.41, 0.42)],
              fill=(170, 200, 165, 255))  # fmt: skip
    d.line([p(0.48, 0.33), p(0.48, 0.18)], fill=(60, 60, 60, 255), width=5 * SS)
    d.polygon([p(0.485, 0.18), p(0.6, 0.215), p(0.485, 0.25)], fill=(225, 50, 45, 255))
    return finish(shadow(img))


# --- vehicle markers (the head of a track; see render/vehicles.py) ---------------------------
# Top views point up (north = the direction of travel; they are rotated with the track), side
# views face right (mirrored when the track runs to the left). Light shapes with a dark rim
# and a soft shadow stay readable on any imagery.
VEHICLE_FILL = (250, 250, 248, 255)
VEHICLE_RIM = (25, 30, 40, 255)


def vehicle(shapes, rim: float = 7.0) -> Image.Image:
    """``shapes``: (kind, points or box, fill) with kind "polygon", "ellipse" or "line"."""
    img, d = canvas()
    for kind, geom, fill in shapes:  # the rim: every shape drawn larger in the rim colour
        if kind == "polygon":
            pts = [p(x, y) for x, y in geom]
            d.polygon(pts, fill=VEHICLE_RIM)
            d.line([*pts, pts[0], pts[1]], fill=VEHICLE_RIM, width=int(2 * rim * SS),
                   joint="curve")  # fmt: skip
        elif kind == "ellipse":
            x0, y0, x1, y1 = geom
            r = rim * SS / 2 / S
            d.ellipse((*p(x0 - r, y0 - r), *p(x1 + r, y1 + r)), fill=VEHICLE_RIM)
        else:
            d.line([p(x, y) for x, y in geom], fill=VEHICLE_RIM, width=int((fill + rim) * SS),
                   joint="curve")  # fmt: skip
    for kind, geom, fill in shapes:
        if kind == "polygon":
            d.polygon([p(x, y) for x, y in geom], fill=fill)
        elif kind == "ellipse":
            d.ellipse((*p(geom[0], geom[1]), *p(geom[2], geom[3])), fill=fill)
        else:
            d.line([p(x, y) for x, y in geom], fill=VEHICLE_FILL, width=int(fill * SS),
                   joint="curve")  # fmt: skip
    return finish(shadow(img, 6))


def draw_vehicle_ship() -> Image.Image:
    """A ship from above, bow up."""
    hull = [(0.5, 0.08), (0.6, 0.22), (0.63, 0.4), (0.63, 0.86), (0.6, 0.92), (0.4, 0.92),
            (0.37, 0.86), (0.37, 0.4), (0.4, 0.22)]  # fmt: skip
    house = [(0.43, 0.52), (0.57, 0.52), (0.57, 0.78), (0.43, 0.78)]
    return vehicle([("polygon", hull, VEHICLE_FILL), ("polygon", house, (210, 70, 55, 255))])


def draw_vehicle_plane() -> Image.Image:
    """An airliner from above, nose up."""
    plane = [(0.5, 0.06), (0.54, 0.14), (0.54, 0.38), (0.9, 0.56), (0.9, 0.62), (0.54, 0.53),
             (0.54, 0.78), (0.64, 0.86), (0.64, 0.9), (0.5, 0.87), (0.36, 0.9), (0.36, 0.86),
             (0.46, 0.78), (0.46, 0.53), (0.1, 0.62), (0.1, 0.56), (0.46, 0.38),
             (0.46, 0.14)]  # fmt: skip
    return vehicle([("polygon", plane, VEHICLE_FILL)])


def draw_vehicle_train() -> Image.Image:
    """A locomotive and a coach from above, front up."""
    loco = [(0.4, 0.12), (0.6, 0.12), (0.62, 0.18), (0.62, 0.5), (0.38, 0.5), (0.38, 0.18)]
    coach = [(0.38, 0.54), (0.62, 0.54), (0.62, 0.92), (0.38, 0.92)]
    return vehicle([("polygon", loco, (220, 60, 50, 255)), ("polygon", coach, VEHICLE_FILL)])


def draw_vehicle_car() -> Image.Image:
    """A car from above, front up."""
    body = [(0.4, 0.16), (0.6, 0.16), (0.64, 0.24), (0.64, 0.82), (0.6, 0.88), (0.4, 0.88),
            (0.36, 0.82), (0.36, 0.24)]  # fmt: skip
    roof = [(0.41, 0.4), (0.59, 0.4), (0.59, 0.7), (0.41, 0.7)]
    screen = [(0.41, 0.33), (0.59, 0.33), (0.58, 0.4), (0.42, 0.4)]
    return vehicle([("polygon", body, (60, 130, 220, 255)), ("polygon", roof, VEHICLE_FILL),
                    ("polygon", screen, (40, 50, 70, 255))])  # fmt: skip


def draw_vehicle_bus() -> Image.Image:
    """A coach from above, front up."""
    body = [(0.39, 0.08), (0.61, 0.08), (0.63, 0.12), (0.63, 0.92), (0.37, 0.92),
            (0.37, 0.12)]  # fmt: skip
    screen = [(0.4, 0.12), (0.6, 0.12), (0.6, 0.19), (0.4, 0.19)]
    roof = [(0.42, 0.25), (0.58, 0.25), (0.58, 0.86), (0.42, 0.86)]
    return vehicle([("polygon", body, (205, 55, 60, 255)), ("polygon", screen, (40, 50, 70, 255)),
                    ("polygon", roof, VEHICLE_FILL)])  # fmt: skip


def draw_vehicle_canoe() -> Image.Image:
    """A canoe with two paddlers from above, bow up."""
    hull = [(0.5, 0.05), (0.56, 0.2), (0.58, 0.5), (0.56, 0.8), (0.5, 0.95), (0.44, 0.8),
            (0.42, 0.5), (0.44, 0.2)]  # fmt: skip
    return vehicle([("polygon", hull, (225, 120, 40, 255)),
                    ("ellipse", (0.455, 0.3, 0.545, 0.39), VEHICLE_FILL),
                    ("ellipse", (0.455, 0.62, 0.545, 0.71), VEHICLE_FILL),
                    ("line", [(0.3, 0.27), (0.7, 0.42)], 5.0),
                    ("line", [(0.7, 0.6), (0.3, 0.75)], 5.0)])  # fmt: skip


def draw_vehicle_bike() -> Image.Image:
    """A cyclist from the side, riding to the right."""
    img, d = canvas()
    w = 9 * SS
    for cx in (0.27, 0.73):  # wheels
        d.ellipse((*p(cx - 0.17, 0.53), *p(cx + 0.17, 0.87)), outline=VEHICLE_RIM, width=w + 8 * SS)
        d.ellipse((*p(cx - 0.17, 0.53), *p(cx + 0.17, 0.87)), outline=VEHICLE_FILL, width=w)
    frame = [p(0.27, 0.7), p(0.45, 0.7), p(0.62, 0.48), p(0.73, 0.7)]
    rider = [p(0.45, 0.7), p(0.42, 0.4), p(0.62, 0.3), p(0.66, 0.46)]
    for line, width in ((frame, w), (rider, 14 * SS)):
        d.line(line, fill=VEHICLE_RIM, width=width + 8 * SS, joint="curve")
        d.line(line, fill=VEHICLE_FILL, width=width, joint="curve")
    d.ellipse((*p(0.6, 0.12), *p(0.74, 0.26)), fill=VEHICLE_RIM)
    d.ellipse((*p(0.615, 0.135), *p(0.725, 0.245)), fill=(225, 60, 50, 255))  # helmet
    return finish(shadow(img, 6))


def _figure(stride: float) -> Image.Image:
    """A person from the side, moving to the right (``stride``: how far the legs swing)."""
    img, d = canvas()
    w = 15 * SS
    body = [p(0.5, 0.3), p(0.47, 0.56)]
    legs = [[p(0.47, 0.56), p(0.47 + stride, 0.73), p(0.5 + stride, 0.92)],
            [p(0.47, 0.56), p(0.44 - stride * 0.6, 0.74), p(0.36 - stride, 0.88)]]  # fmt: skip
    arms = [[p(0.5, 0.34), p(0.6, 0.45), p(0.68, 0.4)], [p(0.5, 0.34), p(0.4, 0.46)]]
    for line in (body, *legs, *arms):
        d.line(line, fill=VEHICLE_RIM, width=w + 8 * SS, joint="curve")
    for line in (body, *legs, *arms):
        d.line(line, fill=VEHICLE_FILL, width=w, joint="curve")
    d.ellipse((*p(0.43, 0.08), *p(0.59, 0.24)), fill=VEHICLE_RIM)
    d.ellipse((*p(0.445, 0.095), *p(0.575, 0.225)), fill=(225, 60, 50, 255))  # head
    return finish(shadow(img, 6))


def draw_vehicle_runner() -> Image.Image:
    return _figure(0.16)


def draw_vehicle_hiker() -> Image.Image:
    img = _figure(0.08)
    return img


def save_vehicles() -> None:
    folder = OUT / "vehicles"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("ship", "plane", "train", "car", "bus", "canoe", "bike", "runner", "hiker"):
        globals()[f"draw_vehicle_{name}"]().save(folder / f"{name}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_pin().save(OUT / "pin.png")
    draw_turnaround().save(OUT / "turnaround.png")
    draw_tent().save(OUT / "tent.png")
    draw_camera().save(OUT / "camera.png")
    draw_flag().save(OUT / "flag.png")
    draw_power().save(OUT / "power.png")
    draw_supermarket().save(OUT / "supermarket.png")
    draw_couscous().save(OUT / "couscous.png")
    draw_summit_decision().save(OUT / "summit.png")
    draw_cold().save(OUT / "cold.png")
    draw_airport().save(OUT / "airport.png")
    draw_ferry().save(OUT / "ferry.png")
    draw_bus().save(OUT / "bus.png")
    draw_hotel().save(OUT / "hotel.png")
    draw_taxi().save(OUT / "taxi.png")
    draw_attraction().save(OUT / "attraction.png")
    draw_finish().save(OUT / "finish.png")
    draw_peak().save(OUT / "peak.png")
    storm = [draw_storm_frame(i / 30) for i in range(30)]
    storm[0].save(OUT / "storm.png", save_all=True, append_images=storm[1:], duration=60,
                  loop=0, disposal=0, blend=0, format="PNG")  # fmt: skip
    frames = [draw_coffee_frame(i / 24) for i in range(24)]
    # animated PNG (APNG); disposal "none" + blend "source" stores complete frames
    frames[0].save(OUT / "coffee.png", save_all=True, append_images=frames[1:], duration=70,
                   loop=0, disposal=0, blend=0, format="PNG")  # fmt: skip
    save_vehicles()
    for path in sorted(OUT.rglob("*.png")):
        print(f"{path.relative_to(OUT)}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
