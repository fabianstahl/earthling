"""The scene list: everything in the scene in one tree (camera, track groups and their tracks,
POIs, photos, map layers). Selecting an entry shows its settings in the inspector; the
checkboxes show and hide; "+" adds; GPX files, photos and project folders can be dropped."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from earthling.ui import theme

KEY = Qt.ItemDataRole.UserRole
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def colour_dot(rgb) -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor.fromRgbF(*rgb))
    p.drawEllipse(QPointF(16, 16), 9, 9)
    p.end()
    return QIcon(pixmap)


def fmt_km(metres: float) -> str:
    return f"{metres / 1000:.1f} km"


class SceneList(QWidget):
    """``selected(key)``: ("camera",), ("group", name), ("track", index), ("poi", i),
    ("photo", i), ("layers",) or ("header", name)."""

    selected = pyqtSignal(tuple)

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent)
        self.window = window
        self.setObjectName("sceneList")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QWidget()
        header.setObjectName("sceneHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 8, 8, 8)
        title = QLabel("Scene")
        title.setObjectName("panelTitle")
        hl.addWidget(title)
        hl.addStretch()
        self.add_button = self._plus("Add tracks, a point of interest, a photo or a line",
                                     self._add_menu)  # fmt: skip
        hl.addWidget(self.add_button)
        layout.addWidget(header)
        self.tree = QTreeWidget()
        self.tree.setObjectName("sceneTree")
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.setIndentation(14)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setUniformRowHeights(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree, 1)
        self.drop_hint = QLabel("Drop GPX files, photos or a project folder here")
        self.drop_hint.setObjectName("dropHint")
        self.drop_hint.setWordWrap(True)
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.drop_hint)
        self._expanded: dict[str, bool] = {}
        self._updating = False
        self._in_item_change = False  # (a checkbox is being handled: see refresh)
        self._refresh_pending = False
        self.refresh()

    @staticmethod
    def _plus(tip: str, menu_factory) -> QToolButton:
        button = QToolButton()
        button.setIcon(theme.icon("plus"))
        button.setIconSize(QSize(16, 16))
        button.setToolTip(tip)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        menu.aboutToShow.connect(lambda: menu_factory(menu))
        button.setMenu(menu)
        return button

    # --- content ----------------------------------------------------------------------
    def current_key(self) -> tuple | None:
        items = self.tree.selectedItems()
        return items[0].data(0, KEY) if items else None

    def refresh(self) -> None:
        """Rebuild the tree (keeping the selection and what is expanded)."""
        if self._in_item_change:
            # A checkbox's handlers ask for this; clearing the tree now would delete the item
            # while QTreeWidgetItem::setData still uses it (a crash): rebuild afterwards.
            if not self._refresh_pending:
                self._refresh_pending = True
                QTimer.singleShot(0, self._deferred_refresh)
            return
        selected = self.current_key()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            self._remember_expanded(item)
        self._updating = True
        self.tree.blockSignals(True)
        self.tree.clear()
        w = self.window
        store = w.scene.store
        keys = len(w.scene.animation.curves.get("camera.pose").keys) \
            if w.scene.animation.is_animated("camera.pose") else 0  # fmt: skip
        self._leaf(None, "Camera", ("camera",), theme.icon("video-camera"),
                   f"{keys} keys" if keys else "free")  # fmt: skip
        session = w.session
        if session is not None:
            tracks = self._header("Tracks", "path", f"{len(session.all_tracks)}",
                                  self._tracks_menu)  # fmt: skip
            for g in session.groups:
                pid = f"trackgroup.{g.name}.visible"
                total = sum(t.stats.distance_m for t in g.tracks)
                group = self._leaf(tracks, g.label, ("group", g.name), theme.icon("path"),
                                   fmt_km(total))  # fmt: skip
                if pid in store.registry:
                    self._check(group, bool(store[pid]))
                for i, t in enumerate(g.tracks):
                    colour = store.get(f"trackgroup.{g.name}.track{i}.color", (1.0, 0.4, 0.2))
                    index = session.all_tracks.index(t)
                    track = self._leaf(group, t.name, ("track", index), colour_dot(colour),
                                       fmt_km(t.stats.distance_m))  # fmt: skip
                    self._check(track, t.visible)
                group.setExpanded(self._expanded.get(f"group:{g.name}", False))
        pois = self._header("Points of interest", "map-pin", f"{len(w.scene.pois)}",
                            self._poi_menu)  # fmt: skip
        for i, poi in enumerate(w.scene.pois):
            self._leaf(pois, poi.name, ("poi", i), w.poi_editor.item_icon(poi), poi.caption,
                       w.poi_editor.item_tooltip(poi))  # fmt: skip
        photos = self._header("Photos", "image", f"{len(w.scene.photos)}", self._photo_menu)
        for i, photo in enumerate(w.scene.photos):
            self._leaf(photos, photo.name, ("photo", i), w.photo_editor.item_icon(photo), "",
                       w.photo_editor.item_tooltip(photo))  # fmt: skip
        layers = self._header("Map layers", "stack", "", None)
        labels = dict(store.registry["layers.a"].options)
        mix = store["layers.mix"]
        self._leaf(layers, labels.get(store["layers.a"], "?"), ("layers",), theme.icon("image"),
                   "A")  # fmt: skip
        if mix > 0:
            self._leaf(layers, labels.get(store["layers.b"], "?"), ("layers",),
                       theme.icon("image"), f"B · {mix:.0%}")  # fmt: skip
        for pid, label, icon in (("borders.overlay", "Borders", "path"),
                                 ("contours.overlay", "Contour lines", "mountains")):  # fmt: skip
            item = self._leaf(layers, label, ("layers",), theme.icon(icon), "overlay")
            item.setData(1, KEY, pid)
            self._check(item, bool(store[pid]))
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)
        header.setSectionResizeMode(1, header.ResizeMode.Fixed)
        metrics = self.tree.fontMetrics()
        widths = [metrics.horizontalAdvance(item.text(1)) for item in self._all_items()]
        header.resizeSection(1, max([28, *widths]) + 18)
        self.tree.blockSignals(False)
        self._updating = False
        if selected is not None:
            self.select(selected, notify=False)

    def _deferred_refresh(self) -> None:
        self._refresh_pending = False
        self.refresh()

    def _all_items(self) -> list[QTreeWidgetItem]:
        out, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            out.append(item)
            stack.extend(item.child(i) for i in range(item.childCount()))
        return out

    def _remember_expanded(self, item: QTreeWidgetItem) -> None:
        key = item.data(0, KEY)
        if key and key[0] in ("header", "group"):
            self._expanded[f"{key[0]}:{key[1]}"] = item.isExpanded()
        for i in range(item.childCount()):
            self._remember_expanded(item.child(i))

    def _header(self, title: str, icon: str, meta: str, menu_factory) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title, meta])
        item.setIcon(0, theme.icon(icon))
        item.setData(0, KEY, ("header", title))
        font = item.font(0)
        font.setWeight(font.Weight.DemiBold)
        item.setFont(0, font)
        item.setForeground(1, QColor(theme.TEXT_3))
        self.tree.addTopLevelItem(item)
        item.setExpanded(self._expanded.get(f"header:{title}", True))
        if menu_factory is not None:
            item.setText(1, "")
            button = self._plus(f"Add to {title.lower()}", menu_factory)
            button.setFixedSize(22, 22)
            self.tree.setItemWidget(item, 1, button)
        return item

    def _leaf(self, parent, text: str, key: tuple, icon: QIcon, meta: str = "",
              tip: str = "") -> QTreeWidgetItem:  # fmt: skip
        item = QTreeWidgetItem([text, meta])
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        item.setIcon(0, icon)
        item.setData(0, KEY, key)
        item.setForeground(1, QColor(theme.TEXT_3))
        item.setToolTip(0, tip or text)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    @staticmethod
    def _check(item: QTreeWidgetItem, on: bool) -> None:
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def _find(self, key: tuple, item: QTreeWidgetItem | None = None) -> QTreeWidgetItem | None:
        items = (
            [item.child(i) for i in range(item.childCount())]
            if item is not None
            else [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        )
        for child in items:
            if child.data(0, KEY) == key and child.data(1, KEY) is None:
                return child
            found = self._find(key, child)
            if found is not None:
                return found
        return None

    def select(self, key: tuple, notify: bool = True) -> None:
        item = self._find(tuple(key))
        if item is None:
            return
        self.tree.blockSignals(not notify)
        self.tree.setCurrentItem(item)
        item.setSelected(True)
        self.tree.blockSignals(False)
        if notify:
            self._on_selection()

    # --- events -----------------------------------------------------------------------
    def _on_selection(self) -> None:
        key = self.current_key()
        if key is not None:
            self.selected.emit(tuple(key))

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        on = item.checkState(0) == Qt.CheckState.Checked
        key, pid = item.data(0, KEY), item.data(1, KEY)
        w = self.window
        self._in_item_change = True
        try:
            if key[0] == "track":
                w.tracks[key[1]].visible = on
                w.tracks_changed.emit()
            elif key[0] == "group":
                w.set_property(f"trackgroup.{key[1]}.visible", on)
            elif pid:
                w.set_property(pid, on)
        finally:
            self._in_item_change = False

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        key = item.data(0, KEY)
        if key and key[0] == "track":
            self.window._frame_track(key[1])

    def _context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        key = item.data(0, KEY) if item is not None else None
        if not key:
            return
        menu = QMenu(self)
        w = self.window
        if key[0] in ("poi", "photo"):
            editor = w.poi_editor if key[0] == "poi" else w.photo_editor
            menu.addAction(theme.icon("trash"), "Remove",
                           lambda: editor.remove_index(key[1]))  # fmt: skip
        elif key[0] == "track":
            menu.addAction(theme.icon("corners-out"), "Frame in the view",
                           lambda: w._frame_track(key[1]))  # fmt: skip
        elif key == ("header", "Points of interest"):
            self._poi_menu(menu)
        elif key == ("header", "Photos"):
            self._photo_menu(menu)
        elif key == ("header", "Tracks"):
            self._tracks_menu(menu)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def keyPressEvent(self, event) -> None:
        key = self.current_key()
        if event.key() == Qt.Key.Key_Delete and key and key[0] in ("poi", "photo"):
            editor = self.window.poi_editor if key[0] == "poi" else self.window.photo_editor
            editor.remove_index(key[1])
        else:
            super().keyPressEvent(event)

    # --- add menus --------------------------------------------------------------------
    def _poi_menu(self, menu: QMenu) -> None:
        menu.clear()
        editor = self.window.poi_editor
        kind = editor.current_kind().name
        menu.addAction(theme.icon("bicycle"), f"{kind} at the marker", editor.add_at_hiker)
        menu.addAction(theme.icon("crosshair"), f"{kind} where I click…", editor.add_by_click)
        menu.addAction(theme.icon("magnifying-glass"), f"{kind} at a place…",
                       editor.add_at_place)  # fmt: skip

    def _photo_menu(self, menu: QMenu) -> None:
        menu.clear()
        menu.addAction(theme.icon("image"), "Add photos…", self.window.photo_editor.add_dialog)

    def _tracks_menu(self, menu: QMenu) -> None:
        menu.clear()
        menu.addAction(theme.icon("file-plus"), "Add GPX files…", self.window.add_gpx_dialog)
        menu.addAction(theme.icon("path"), "New line (flight, ferry, road)…",
                       self.window.new_line)  # fmt: skip

    def _add_menu(self, menu: QMenu) -> None:
        menu.clear()
        self._tracks_menu(menu.addMenu(theme.icon("path"), "Tracks"))
        self._poi_menu(menu.addMenu(theme.icon("map-pin"), "Point of interest"))
        self._photo_menu(menu.addMenu(theme.icon("image"), "Photos"))

    # --- drag and drop ----------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        self.window.drop_paths(paths)
        event.acceptProposedAction()
