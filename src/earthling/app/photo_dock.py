"""Photos dock: pictures shown on the screen (with a frame, a shadow and a slow zoom). A photo
can know where it was taken (EXIF GPS, or its EXIF time on the tracks) to appear when the drawn
line reaches that place; placement, size and the fades are properties (see app/item_dock.py)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QLabel, QLineEdit, QPushButton, QWidget

from earthling.app.item_dock import SceneItemDock
from earthling.core.photos import Photo, read_exif
from earthling.core.pois import slug
from earthling.render.photos import resolve_photo
from earthling.ui.widgets import row

PHOTO_FILTER = "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff)"


class PhotoDock(SceneItemDock):
    ATTR, PREFIX, NOUN, ITEM = "photos", "photo.", "Photo", Photo
    NOTE = ("Placement, size, the fade (opacity keys or 'Appear when the line reaches…') are in "
            "the photo's section of the parameter panel.")  # fmt: skip

    def __init__(self, scene, undo_stack, hiker_position, request_pick, base_dir, parent=None,
                 choose_place=None, place_at_time: Callable | None = None) -> None:  # fmt: skip
        """``place_at_time(datetime) -> (lon, lat) | None``: where the tracks were at that time
        (photos without GPS)."""
        self.place_at_time = place_at_time
        super().__init__("Photos", scene, undo_stack, hiker_position, request_pick, base_dir,
                         parent, choose_place)  # fmt: skip
        self.setObjectName("photo_dock")

    def _add_buttons(self) -> list[QPushButton]:
        self.add_files = QPushButton("+ Photos…")
        self.add_files.setToolTip(
            "Add pictures; they are placed where they were taken "
            "(EXIF GPS, or their time on the tracks), else at the hiker"
        )
        return [self.add_files]

    def _form_rows(self, form: QFormLayout) -> list[QWidget]:
        self.name = QLineEdit()
        self.file = QLabel()
        self.file.setWordWrap(True)
        self.change_file = QPushButton("Change…")
        form.addRow("Name", self.name)
        form.addRow("File", row((self.file, 1), self.change_file))
        return [self.name, self.change_file]

    def _connect_fields(self) -> None:
        self.add_files.clicked.connect(self._add_dialog)
        self.change_file.clicked.connect(self._change_file)
        self.name.editingFinished.connect(lambda: self._edit_field("name", self.name.text()))

    def item_icon(self, photo: Photo) -> QIcon:
        path = resolve_photo(photo.path, self.base_dir())
        return QIcon(QPixmap(str(path))) if path is not None else QIcon()

    def _show_item(self, photo: Photo) -> None:
        self.name.setText(photo.name)
        self.file.setText(photo.path)

    # --- adding ---------------------------------------------------------------------------
    def _add_dialog(self) -> None:
        start = str(self.base_dir() or Path.home())
        paths, _ = QFileDialog.getOpenFileNames(self, "Photos", start, PHOTO_FILTER)
        for path in paths:
            self.add_photo(Path(path))

    def where_taken(self, path: Path) -> tuple[float, float] | None:
        """(lon, lat) of a photo: EXIF GPS, else its EXIF time on the tracks, else the hiker."""
        lat, lon, taken = read_exif(path)
        if lat is not None:
            return lon, lat
        if taken is not None and self.place_at_time is not None:
            place = self.place_at_time(taken)
            if place is not None:
                return place
        return self.hiker_position()

    def add_photo(self, path: Path) -> Photo:
        lon, lat = self.where_taken(path) or (None, None)
        photo = Photo(slug(path.stem, {p.id for p in self.scene.photos}), path.stem,
                      self.file_ref(str(path)), lat, lon)  # fmt: skip
        self.add_item(photo, "Add photo")
        return photo

    def _change_file(self) -> None:
        photo = self.current()
        if photo is None:
            return
        start = str(self.base_dir() or Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Photo", start, PHOTO_FILTER)
        if path:
            self._edit_field("path", self.file_ref(path))
