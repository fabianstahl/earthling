"""The application's look: one dark theme from a few colour tokens, the Inter font and the
Phosphor icons (regular weight), tinted per state.

Colours with a meaning: ACCENT = selected / changed, KEY = animated, RECORD = auto-key on.
"""

from __future__ import annotations

import tempfile
from contextlib import suppress
from functools import cache
from pathlib import Path

from PyQt6.QtCore import QByteArray, QPointF, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QIconEngine,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
)
from PyQt6.QtSvg import QSvgRenderer

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ICON_DIR = ASSETS / "ui_icons"
FONT_DIR = ASSETS / "fonts"
FONT_FAMILY = "Inter"

GROUND = "#0C0E12"  # behind everything, letterbox around the export frame
PANEL = "#14171C"
FIELD = "#1B1F26"
HOVER = "#232932"
LINE = "#242A33"
LINE_STRONG = "#323946"
TEXT = "#E8EBF0"
TEXT_2 = "#B9C1CC"
TEXT_3 = "#8D97A6"
ACCENT = "#4D9FFF"
ACCENT_BG = "#1C3352"
ACCENT_TEXT = "#8CC2FF"
KEY = "#F0B43C"
KEY_FIELD = "#3B3119"  # fill of an animated value
KEY_FIELD_ON = "#4F3E16"  # ... with a key at the playhead
VALUE_FILL = "#253A57"  # fill of a number field
RECORD = "#FF5B52"
OK = "#4CC38A"


def qcolor(token: str, alpha: int = 255) -> QColor:
    c = QColor(token)
    c.setAlpha(alpha)
    return c


# --- fonts ----------------------------------------------------------------------------------
@cache
def load_fonts() -> str:
    """Register the bundled Inter files once; returns the family name."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(path))
    return FONT_FAMILY


def ui_font(px: float = 13.0, weight: QFont.Weight = QFont.Weight.Normal,
            tabular: bool = False) -> QFont:  # fmt: skip
    """Inter at ``px`` logical pixels; ``tabular``: figures of equal width (values, times)."""
    font = QFont(load_fonts())
    font.setPixelSize(round(px))
    font.setWeight(weight)
    if tabular:
        with suppress(AttributeError, TypeError):  # (Qt before 6.7)
            font.setFeature(QFont.Tag("tnum"), 1)
    return font


# --- icons ----------------------------------------------------------------------------------
@cache
def _svg(name: str) -> bytes:
    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        raise KeyError(f"no UI icon '{name}'")
    return path.read_bytes()


def render_icon(name: str, size: int, color: str | QColor) -> QImage:
    """The icon ``name`` at ``size`` device pixels in one colour."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    QSvgRenderer(QByteArray(_svg(name))).render(p, QRectF(0, 0, size, size))
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(image.rect(), QColor(color))
    p.end()
    return image


class _TintedIcon(QIconEngine):
    """Draws a Phosphor icon sharply at any size, coloured by mode and state."""

    def __init__(self, name: str, colors: dict[str, str]) -> None:
        super().__init__()
        self.name = name
        self.colors = colors

    def _color(self, mode: QIcon.Mode, state: QIcon.State) -> str:
        if mode == QIcon.Mode.Disabled:
            return self.colors["disabled"]
        if state == QIcon.State.On and "on" in self.colors:
            return self.colors["on"]
        if mode in (QIcon.Mode.Active, QIcon.Mode.Selected):
            return self.colors["active"]
        return self.colors["normal"]

    def paint(self, painter: QPainter, rect: QRect, mode, state) -> None:
        ratio = painter.device().devicePixelRatioF() if painter.device() else 1.0
        side = min(rect.width(), rect.height())
        image = render_icon(self.name, max(1, round(side * ratio)), self._color(mode, state))
        image.setDevicePixelRatio(ratio)
        x = rect.x() + (rect.width() - side) / 2
        y = rect.y() + (rect.height() - side) / 2
        painter.drawImage(QPointF(x, y), image)

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        side = min(size.width(), size.height())
        return QPixmap.fromImage(render_icon(self.name, max(1, side), self._color(mode, state)))

    def scaledPixmap(self, size: QSize, mode, state, scale: float) -> QPixmap:  # noqa: N802
        side = max(1, round(min(size.width(), size.height()) * scale))
        pixmap = QPixmap.fromImage(render_icon(self.name, side, self._color(mode, state)))
        pixmap.setDevicePixelRatio(scale)
        return pixmap

    def clone(self) -> QIconEngine:
        return _TintedIcon(self.name, dict(self.colors))


def icon(name: str, color: str = TEXT_2, active: str = TEXT, on: str | None = None) -> QIcon:
    """A Phosphor icon (file name without .svg). ``on``: colour of checked buttons."""
    colors = {"normal": color, "active": active, "disabled": qcolor(TEXT_3, 110).name(
        QColor.NameFormat.HexArgb)}  # fmt: skip
    if on is not None:
        colors["on"] = on
    return QIcon(_TintedIcon(name, colors))


def tool_icon(name: str) -> QIcon:
    """Icon of a checkable tool button: accent colour while checked."""
    return icon(name, on=ACCENT_TEXT)


@cache
def tinted_svg_file(name: str, color: str) -> str:
    """A copy of an icon in one colour, for style sheets (``image: url(...)``)."""
    folder = Path(tempfile.gettempdir()) / "earthling-ui-icons"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}-{color.lstrip('#')}.svg"
    path.write_bytes(_svg(name).replace(b"currentColor", color.encode("ascii")))
    return path.as_posix()


# --- application ----------------------------------------------------------------------------
def palette() -> QPalette:
    pal = QPalette()
    role, group = QPalette.ColorRole, QPalette.ColorGroup
    for r, token in (
        (role.Window, PANEL), (role.WindowText, TEXT), (role.Base, FIELD),
        (role.AlternateBase, "#181C22"), (role.Text, TEXT), (role.Button, FIELD),
        (role.ButtonText, TEXT), (role.BrightText, "#FFFFFF"), (role.Highlight, ACCENT_BG),
        (role.HighlightedText, "#FFFFFF"), (role.ToolTipBase, "#232932"),
        (role.ToolTipText, TEXT), (role.PlaceholderText, TEXT_3), (role.Link, ACCENT_TEXT),
        (role.Mid, LINE_STRONG), (role.Dark, GROUND), (role.Midlight, HOVER),
        (role.Light, "#3A4250"), (role.Shadow, "#000000"),
    ):  # fmt: skip
        pal.setColor(r, QColor(token))
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(group.Disabled, r, QColor(TEXT_3))
    return pal


def stylesheet() -> str:
    down = tinted_svg_file("caret-down", TEXT_3)
    up = tinted_svg_file("caret-up", TEXT_3)
    check = tinted_svg_file("check", "#06101F")
    return f"""
QWidget {{ font-family: "{FONT_FAMILY}"; font-size: 13px; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {PANEL}; }}
QMainWindow::separator {{ background: {LINE}; width: 1px; height: 1px; }}
QMainWindow::separator:hover {{ background: {ACCENT}; }}
QToolTip {{ background: {HOVER}; color: {TEXT}; border: 1px solid {LINE_STRONG};
    padding: 5px 8px; border-radius: 6px; }}
QMenuBar {{ background: {PANEL}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 6px 10px; border-radius: 6px; color: {TEXT_2}; }}
QMenuBar::item:selected {{ background: {HOVER}; color: {TEXT}; }}
QMenu {{ background: #181C22; border: 1px solid {LINE_STRONG}; padding: 6px;
    border-radius: 8px; }}
QMenu::item {{ padding: 6px 28px 6px 10px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_BG}; }}
QMenu::item:disabled {{ color: {TEXT_3}; }}
QMenu::icon {{ padding-left: 8px; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 5px 6px; }}
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {LINE};
    padding: 4px 8px; spacing: 2px; }}
QToolBar::separator {{ background: {LINE}; width: 1px; margin: 6px 6px; }}
QToolButton {{ background: transparent; border: none; border-radius: 7px; padding: 5px;
    color: {TEXT_2}; }}
QToolButton:hover {{ background: {HOVER}; color: {TEXT}; }}
QToolButton:checked {{ background: {ACCENT_BG}; color: {ACCENT_TEXT}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QPushButton {{ background: {FIELD}; border: 1px solid {LINE_STRONG}; border-radius: 7px;
    padding: 5px 12px; color: {TEXT_2}; }}
QPushButton:hover {{ background: {HOVER}; color: {TEXT}; }}
QPushButton:pressed {{ background: {LINE_STRONG}; }}
QPushButton:disabled {{ color: {TEXT_3}; border-color: {LINE}; }}
QPushButton[accent="true"] {{ background: {ACCENT}; color: #06101F; border: none;
    font-weight: 600; }}
QPushButton[accent="true"]:hover {{ background: #6EB1FF; }}
QPushButton[ghost="true"] {{ background: transparent; }}
QLineEdit, QAbstractSpinBox, QDateTimeEdit, QPlainTextEdit, QTextEdit {{
    background: {FIELD}; border: 1px solid {LINE_STRONG}; border-radius: 6px;
    padding: 3px 6px; selection-background-color: {ACCENT_BG}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 14px; border: none;
    background: transparent; }}
QAbstractSpinBox::up-arrow {{ image: url("{up}"); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}
QComboBox {{ background: {FIELD}; border: 1px solid {LINE_STRONG}; border-radius: 6px;
    padding: 3px 8px; }}
QComboBox:hover {{ border-color: #4A5463; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: url("{down}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: #181C22; border: 1px solid {LINE_STRONG};
    selection-background-color: {ACCENT_BG}; outline: none; padding: 4px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator, QTreeView::indicator, QListView::indicator {{ width: 14px; height: 14px;
    border-radius: 4px; border: 1px solid #4A5463; background: {FIELD}; }}
QCheckBox::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT}; image: url("{check}"); }}
QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {TEXT_3}; padding: 7px 14px;
    border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QDockWidget {{ titlebar-close-icon: none; color: {TEXT_2}; }}
QDockWidget::title {{ background: {PANEL}; padding: 7px 10px; border-bottom: 1px solid {LINE};
    text-align: left; }}
QTreeView, QListView, QTableView {{ background: {PANEL}; alternate-background-color: #171A20;
    border: none; outline: none; }}
QTreeView::item, QListView::item {{ padding: 3px 2px; border-radius: 5px; }}
QTreeView::item:hover, QListView::item:hover {{ background: {HOVER}; }}
QTreeView::item:selected, QListView::item:selected {{ background: {ACCENT_BG}; color: #FFFFFF; }}
QHeaderView::section {{ background: {PANEL}; color: {TEXT_3}; border: none;
    border-bottom: 1px solid {LINE}; padding: 5px 8px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: #353C48; border-radius: 3px; min-height: 24px;
    min-width: 24px; }}
QScrollBar::handle:hover {{ background: #4A5463; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE_STRONG}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; height: 12px; margin: -4px 0;
    border-radius: 6px; }}
QProgressBar {{ background: {FIELD}; border: 1px solid {LINE_STRONG}; border-radius: 6px;
    text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QStatusBar {{ background: #0F1216; color: {TEXT_3}; border-top: 1px solid {LINE}; }}
QStatusBar QLabel {{ color: {TEXT_3}; font-size: 12px; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 8px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {TEXT_3}; }}
"""


def apply(app) -> None:
    """Fusion with the dark palette, Inter and the style sheet (idempotent)."""
    load_fonts()
    app.setStyle("Fusion")
    app.setPalette(palette())
    app.setFont(ui_font())
    app.setStyleSheet(stylesheet())
