from PyQt6.QtGui import QIcon

from earthling.ui import theme


def test_every_ui_icon_renders_in_its_colour(qapp):
    names = [p.stem for p in theme.ICON_DIR.glob("*.svg")]
    assert "play" in names and len(names) > 50
    for name in names:
        image = theme.render_icon(name, 24, theme.ACCENT)
        alpha = [image.pixelColor(x, y).alpha() for x in range(24) for y in range(24)]
        assert max(alpha) > 0, name


def test_tinted_icon_follows_the_state(qapp):
    icon = theme.icon("star", theme.TEXT_3, theme.TEXT, on=theme.KEY)

    def colour(mode, state):
        image = icon.pixmap(32, 32, mode, state).toImage()
        pixels = [image.pixelColor(x, y) for x in range(32) for y in range(32)]
        return max(pixels, key=lambda c: c.alpha()).name().upper()

    assert colour(QIcon.Mode.Normal, QIcon.State.Off) == theme.TEXT_3
    assert colour(QIcon.Mode.Active, QIcon.State.Off) == theme.TEXT
    assert colour(QIcon.Mode.Normal, QIcon.State.On) == theme.KEY


def test_theme_applies_inter_and_the_dark_palette(qapp):
    theme.apply(qapp)
    assert qapp.font().family() == "Inter"
    assert qapp.palette().window().color().name().upper() == theme.PANEL
    assert "QToolTip" in qapp.styleSheet()
