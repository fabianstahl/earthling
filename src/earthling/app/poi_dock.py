"""Points of interest dock: add POIs of a kind from the library (at the hiker, by clicking the
terrain or at a place), edit and remove them (see app/item_dock.py)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, QStandardPaths
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from earthling.app.item_dock import SceneItemDock
from earthling.core.poi_library import KINDS, PoiKind, add_user_image, load_user_images
from earthling.core.pois import Poi, slug
from earthling.render.pois import BADGE, badge_image, builtin_icons, resolve_icon
from earthling.ui.widgets import double_spin, row

CUSTOM = "__custom__"
ADD_IMAGE = "__add_image__"
IMAGE_FILTER = "Images (*.png *.gif *.apng *.webp)"


def user_images_file() -> Path:
    """The user's own POI images (shared by all projects)."""
    folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(folder) / "poi_images.json"


def icon_for(ref: str, caption: str = "", base_dir: Path | None = None) -> QIcon:
    """A small picture of an icon reference (a builtin, a file, the text badge, none)."""
    if ref == BADGE:
        image = badge_image(caption or "Text")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
        return QIcon(QPixmap.fromImage(qimage.copy()))
    path = resolve_icon(ref, base_dir)
    return QIcon(QPixmap(str(path))) if path is not None else QIcon()


class PoiDock(SceneItemDock):
    ATTR, PREFIX, NOUN, ITEM = "pois", "poi.", "POI", Poi
    NOTE = "Animate visibility, effects … in the POI's section of the parameter panel."

    def __init__(self, scene, undo_stack, hiker_position, request_pick, base_dir, parent=None,
                 choose_place=None) -> None:  # fmt: skip
        super().__init__("Points of Interest", scene, undo_stack, hiker_position, request_pick,
                         base_dir, parent, choose_place)  # fmt: skip
        self.setObjectName("poi_dock")

    # --- layout ---------------------------------------------------------------------------
    def _top(self, layout: QVBoxLayout) -> None:
        self.kind = QComboBox()
        self.kind.setIconSize(QSize(22, 22))
        self.kind.setToolTip("What the new POI is: its icon, name and caption")
        self._fill_kinds()
        layout.addWidget(row(QLabel("New"), (self.kind, 1)))

    def _add_buttons(self) -> list[QPushButton]:
        self.add_hiker = QPushButton("+ At hiker")
        self.add_hiker.setToolTip("New POI at the head of the drawn track")
        self.add_click = QPushButton("+ On map…")
        self.add_click.setToolTip("New POI where you click on the terrain")
        self.add_place = QPushButton("+ At…")
        self.add_place.setToolTip("New POI at coordinates or at a point of a track (start, end, "
                                  "highest or lowest point, a distance)")  # fmt: skip
        return [self.add_hiker, self.add_click, self.add_place]

    def _form_rows(self, form: QFormLayout) -> list[QWidget]:
        self.name = QLineEdit()
        self.caption = QLineEdit()
        self.icon = QComboBox()
        self.icon.addItem("No icon (caption only)", "none")
        self.icon.addItem(icon_for(BADGE), "Text badge (caption inside)", BADGE)
        for name in builtin_icons():
            self.icon.addItem(icon_for(f"builtin:{name}"), name, f"builtin:{name}")
        self.icon.addItem("Custom file…", CUSTOM)
        self.size_px = double_spin(8.0, 512.0, suffix=" px")
        self.lift = double_spin(0.0, 400.0, suffix=" px")
        self.cols = QSpinBox()
        self.rows = QSpinBox()
        for spin in (self.cols, self.rows):
            spin.setRange(1, 64)
        self.fps = double_spin(0.5, 60.0)
        sprite = row(QLabel("cols"), self.cols, QLabel("rows"), self.rows, QLabel("fps"), self.fps)
        form.addRow("Name", self.name)
        form.addRow("Caption", self.caption)
        form.addRow("Icon", self.icon)
        form.addRow("Size (1080p)", self.size_px)
        form.addRow("Pin height", self.lift)
        form.addRow("Sprite sheet", sprite)
        return [self.name, self.caption, self.icon, self.size_px, self.lift, self.cols,
                self.rows, self.fps]  # fmt: skip

    def _fill_kinds(self, select: str | None = None) -> None:
        """The library grouped by category, the user's images and "Add image…"."""
        self.kind.clear()
        category = None
        for kind in [*KINDS, *load_user_images(user_images_file())]:
            if kind.category != category:
                if category is not None:
                    self.kind.insertSeparator(self.kind.count())
                category = kind.category
            label = kind.name if kind.icon != "none" else f"{kind.name} (no icon)"
            self.kind.addItem(icon_for(kind.icon, kind.caption), label, kind)
            self.kind.setItemData(self.kind.count() - 1, category, 3)  # (tooltip: the category)
        self.kind.insertSeparator(self.kind.count())
        self.kind.addItem("Add image…", ADD_IMAGE)
        index = next((i for i in range(self.kind.count())
                      if isinstance(self.kind.itemData(i), PoiKind)
                      and self.kind.itemData(i).icon == select), 0)  # fmt: skip
        self.kind.setCurrentIndex(index)

    def _connect_fields(self) -> None:
        self.kind.activated.connect(self._kind_chosen)
        self.add_hiker.clicked.connect(self._add_at_hiker)
        self.add_click.clicked.connect(lambda: self.request_pick(self.add_poi))
        self.add_place.clicked.connect(self._add_at_place)
        self.icon.activated.connect(self._icon_chosen)
        for field, widget, value in (
            ("name", self.name, self.name.text),
            ("caption", self.caption, self.caption.text),
            ("size_px", self.size_px, self.size_px.value),
            ("lift_px", self.lift, self.lift.value),
            ("sprite_cols", self.cols, self.cols.value),
            ("sprite_rows", self.rows, self.rows.value),
            ("fps", self.fps, self.fps.value),
        ):
            widget.editingFinished.connect(lambda f=field, v=value: self._edit_field(f, v()))

    # --- items ----------------------------------------------------------------------------
    def item_icon(self, poi: Poi) -> QIcon:
        return icon_for(poi.icon, poi.caption, self.base_dir())

    def item_tooltip(self, poi: Poi) -> str:
        return f"{poi.caption}\n{poi.lat:.5f}, {poi.lon:.5f}".strip()

    def _show_item(self, poi: Poi) -> None:
        self.name.setText(poi.name)
        self.caption.setText(poi.caption)
        index = self.icon.findData(poi.icon)
        if index < 0:
            self.icon.insertItem(self.icon.count() - 1, Path(poi.icon).name, poi.icon)
            index = self.icon.findData(poi.icon)
        self.icon.setCurrentIndex(index)
        self.size_px.setValue(poi.size_px)
        self.lift.setValue(poi.lift_px)
        self.cols.setValue(poi.sprite_cols)
        self.rows.setValue(poi.sprite_rows)
        self.fps.setValue(poi.fps)

    def current_kind(self) -> PoiKind:
        kind = self.kind.currentData()
        return kind if isinstance(kind, PoiKind) else KINDS[0]

    def add_poi(self, lon: float, lat: float, name: str | None = None,
                kind: PoiKind | None = None) -> Poi:  # fmt: skip
        """A new POI of ``kind`` (default: the one chosen in the dock) at lon/lat."""
        kind = kind or self.current_kind()
        name = name or kind.name
        poi = Poi(slug(name, {p.id for p in self.scene.pois}), name, float(lon), float(lat),
                  icon=self._icon_ref(kind.icon), caption=kind.caption,
                  size_px=kind.size_px)  # fmt: skip
        self.add_item(poi)
        return poi

    def _add_at_hiker(self) -> None:
        position = self.hiker_position()
        if position is not None:
            self.add_poi(*position)

    def _add_at_place(self) -> None:
        if self.choose_place is not None:
            place = self.choose_place("New POI at…", self.hiker_position())
            if place is not None:
                self.add_poi(*place)

    # --- icons ----------------------------------------------------------------------------
    def _kind_chosen(self, index: int) -> None:
        if self.kind.itemData(index) != ADD_IMAGE:
            return
        start = str(self.base_dir() or Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "POI image", start, IMAGE_FILTER)
        if path:
            self._fill_kinds(select=add_user_image(user_images_file(), Path(path)).icon)
        else:
            self.kind.setCurrentIndex(0)

    def _icon_ref(self, ref: str) -> str:
        """Image files relative to the scene folder where possible."""
        if ref in ("none", BADGE) or ref.startswith("builtin:"):
            return ref
        return self.file_ref(ref)

    def _icon_chosen(self, index: int) -> None:
        data = self.icon.itemData(index)
        if data == CUSTOM:
            start = str(self.base_dir() or Path.home())
            path, _ = QFileDialog.getOpenFileName(self, "POI icon", start, IMAGE_FILTER)
            if not path:
                self._show_current()
                return
            data = self._icon_ref(path)
        self._edit_field("icon", data)
