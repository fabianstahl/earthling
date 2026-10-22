"""The inspector: a rail of categories, a search field, filters and the settings of one
category (or of the item selected in the scene list) at a time.

Sections are assigned to categories by their title (``CATEGORIES``); a section no category
claims lands in "More". Search and the filters (changed, animated, pinned) list matching
settings of the whole scene, grouped by section, with a link to each section.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.core.properties import PropertyDef, PropertyStore
from earthling.ui import theme
from earthling.ui.property_panel import (
    EditCallback,
    KeyState,
    PropertyEditor,
    PropertyRow,
    SectionView,
    editor_for,
    has_editor,
)


@dataclass(frozen=True)
class Category:
    id: str
    title: str
    short: str  # under the rail icon
    icon: str
    subtitle: str
    sections: tuple[str, ...] = ()  # section titles, or prefixes ending in ": "
    looks: tuple = field(default=(), compare=False)  # filled by the application


CATEGORIES = [
    Category("camera", "Camera", "Camera", "video-camera", "How the camera moves",
             ("Camera", "View")),
    Category("terrain", "Terrain", "Terrain", "mountains", "Shape and detail of the ground",
             ("Terrain",)),
    Category("light", "Sun & Light", "Light", "sun-horizon",
             "Time of day, sunlight, shadows, exposure",
             ("Sun & Time", "Lighting", "Shadows", "Exposure", "Glow", "Night Lights")),
    Category("weather", "Atmosphere & Weather", "Weather", "cloud-fog",
             "Sky, haze, fog, clouds, rain and snow",
             ("Sky & Atmosphere", "Height Fog", "Weather: ")),
    Category("water", "Water", "Water", "waves", "Sea and lakes from OpenStreetMap", ("Water",)),
    Category("tracks", "Tracks", "Tracks", "path",
             "All tracks · one group's style: select it in the scene list",
             ("Tracks", "Tracks: ")),
    Category("progress", "Progress & Marker", "Progress", "person-simple-hike",
             "How much of the journey is drawn", ("Hike Progress", "Marker")),
    Category("labels", "Labels", "Labels", "tag", "Peaks, passes and places",
             ("Labels", "Which Labels")),
    Category("overlays", "Overlays", "Overlays", "squares-four",
             "Stats, elevation profile and mini-map",
             ("Stats Overlay", "Stats Values", "Elevation Profile", "Mini-map")),
    Category("layers", "Map Layers", "Layers", "stack", "Imagery and map styles on the terrain",
             ("Texture Layers", "Borders", "Contour Lines", "Layer: ")),
    Category("export", "Render & Export", "Export", "film-strip", "Output size and quality",
             ("Export",)),
]  # fmt: skip
OTHER = Category("other", "More", "More", "dots-three", "Settings of no other category")
# items selected in the scene list have their own sections, found by search only
ITEM_PREFIXES = ("POI: ", "Photo: ")

# German words -> English parts of labels (search finds "Nebel", "Wolken", ...)
SYNONYMS = {
    "nebel": "fog", "dunst": "haze", "wolke": "cloud", "wolken": "cloud", "regen": "rain",
    "schnee": "snow", "gewitter": "lightning", "blitz": "lightning", "sonne": "sun",
    "licht": "light", "schatten": "shadow", "himmel": "sky", "sterne": "star",
    "wasser": "water", "see": "water", "meer": "water", "wellen": "wave", "kamera": "camera",
    "höhe": "height", "breite": "width", "farbe": "colour", "helligkeit": "brightness",
    "deckkraft": "opacity", "größe": "size", "groesse": "size", "spur": "track",
    "linie": "line", "beschriftung": "label", "gipfel": "peak", "hütte": "hut",
    "karte": "map", "minikarte": "mini-map", "profil": "profile", "geschwindigkeit": "speed",
    "puls": "heart", "herzfrequenz": "heart", "datum": "date", "zeit": "time",
    "uhrzeit": "time", "sprache": "language", "belichtung": "exposure", "gelände": "terrain",
    "überhöhung": "exaggeration", "grenzen": "border", "höhenlinien": "contour",
    "satellit": "satellite", "auflösung": "resolution", "unschärfe": "blur",
    "kantenglättung": "anti-aliasing", "leuchten": "glow", "markierung": "marker",
    "fortschritt": "progress", "fahrzeug": "vehicle", "nacht": "night", "stadt": "cit",
    "städte": "cit", "länder": "countr", "regenbogen": "rainbow", "sicht": "visibility",
    "nass": "wet", "wind": "wind", "böen": "gust", "strecke": "distance", "aufstieg": "ascent",
    "steigung": "gradient", "umriss": "outline", "schrift": "text", "position": "position",
}  # fmt: skip


def category_of(title: str) -> Category | None:
    """The category of a section title (None: an item's section)."""
    if title.startswith(ITEM_PREFIXES):
        return None
    for category in CATEGORIES:
        for name in category.sections:
            if title == name or (name.endswith(": ") and title.startswith(name)):
                return category
    return OTHER


def section_display(title: str) -> str:
    """The title inside its category: "Weather: Rain" -> "Rain", "Layer: Slope" -> ...."""
    if title.startswith("Weather: "):
        return title.removeprefix("Weather: ")
    if title.startswith("Layer: "):
        return f"{title.removeprefix('Layer: ')} layer"
    if title.startswith("Tracks: "):
        return f"Group: {title.removeprefix('Tracks: ')}"
    return title


def item_section_display(title: str) -> str:
    """The title of an item's own section: "POI: Camp" -> "Settings", the track colours."""
    if title.startswith(ITEM_PREFIXES):
        return "Settings"
    if title.startswith("Tracks: "):
        return "Track colours" if title.endswith(" – colours") else "Line & marker"
    return title


def normalise(text: str) -> str:
    return text.lower().replace("color", "colour").replace("grey", "gray")


def search_terms(query: str) -> list[list[str]]:
    """Each word of the query with its English synonyms (all words must match)."""
    out = []
    for word in re.split(r"\s+", normalise(query).strip()):
        if word:
            out.append([word] + ([SYNONYMS[word]] if word in SYNONYMS else []))
    return out


def matches(d: PropertyDef, terms: list[list[str]]) -> bool:
    haystack = normalise(f"{d.label} {d.section} {d.tooltip} {d.keywords} {d.id}")
    return all(any(t in haystack for t in alternatives) for alternatives in terms)


def highlight(label: str, terms: list[list[str]]) -> str:
    """The label as rich text with the first matching word marked."""
    low = normalise(label)
    for alternatives in terms:
        for t in alternatives:
            i = low.find(t)
            if i >= 0:
                return (html.escape(label[:i]) + f'<span style="color:#FFFFFF;background:'
                        f'#2A4A75">{html.escape(label[i:i + len(t)])}</span>'
                        + html.escape(label[i + len(t):]))  # fmt: skip
    return html.escape(label)


class Rail(QWidget):
    """The category buttons (icon over a short label)."""

    chosen = pyqtSignal(str)

    def __init__(self, categories: list[Category], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("rail")
        self.setFixedWidth(68)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QToolButton] = {}
        for category in categories:
            b = QToolButton()
            b.setObjectName("railButton")
            b.setIcon(theme.icon(category.icon, theme.TEXT_3, theme.TEXT, on=theme.ACCENT_TEXT))
            b.setIconSize(QSize(22, 22))
            b.setText(category.short)
            b.setToolTip(f"{category.title}: {category.subtitle}")
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setFixedSize(56, 50)
            b.clicked.connect(lambda _=False, c=category.id: self.chosen.emit(c))
            self.group.addButton(b)
            self.buttons[category.id] = b
            layout.addWidget(b)
        layout.addStretch()
        self.extra = QVBoxLayout()
        layout.addLayout(self.extra)

    def set_current(self, category_id: str | None) -> None:
        self.group.setExclusive(False)
        for cid, b in self.buttons.items():
            b.setChecked(cid == category_id)
        self.group.setExclusive(True)

    def add_button(self, icon: str, text: str, tip: str, slot: Callable) -> QToolButton:
        b = QToolButton()
        b.setObjectName("railButton")
        b.setIcon(theme.icon(icon, theme.TEXT_3, theme.TEXT))
        b.setIconSize(QSize(22, 22))
        b.setText(text)
        b.setToolTip(tip)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        b.setFixedSize(56, 50)
        b.clicked.connect(slot)
        self.extra.addWidget(b)
        return b


class Inspector(QWidget):
    """Category rail + header, search, filter chips and the sections of the current view.

    ``editors``, ``rows`` and ``sections`` hold the widgets currently shown."""

    FILTERS = (("all", "All"), ("changed", "Changed"), ("animated", "Animated"),
               ("pinned", "Pinned"))  # fmt: skip

    def __init__(self, store: PropertyStore, on_edit: EditCallback,
                 row_extra: Callable[[PropertyDef], QWidget | None] | None = None,
                 key_state: KeyState | None = None,
                 is_animated: Callable[[str], bool] | None = None,
                 batch: Callable[[str, Callable], None] | None = None,
                 settings: QSettings | None = None, parent=None) -> None:  # fmt: skip
        super().__init__(parent)
        self.store = store
        self.on_edit = on_edit
        self.row_extra = row_extra
        self.key_state = key_state
        self.is_animated = is_animated or (lambda pid: False)
        self.batch = batch
        self.settings = settings or QSettings()
        self.pins: set[str] = set(self._load_pins())
        self.category = "camera"
        self.filter = "all"
        self.item: dict | None = None  # {"title", "subtitle", "icon", "sections", "top"}
        self.editors: dict[str, PropertyEditor] = {}
        self.rows: dict[str, PropertyRow] = {}
        self.sections: dict[str, SectionView] = {}
        self._section_state: dict[str, tuple[bool, bool, str | None]] = {}
        self._build()
        self._unsubscribe = store.subscribe(self._on_store_changed)
        self._counts_timer = QTimer(self, singleShot=True, interval=0)
        self._counts_timer.timeout.connect(self._update_counts)
        self._search_timer = QTimer(self, singleShot=True, interval=180)
        self._search_timer.timeout.connect(self.rebuild)
        self.rebuild()

    # --- layout -----------------------------------------------------------------------
    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.rail = Rail(CATEGORIES)
        self.rail.chosen.connect(self.show_category)
        layout.addWidget(self.rail)
        line = QFrame()
        line.setObjectName("vline")
        line.setFixedWidth(1)
        layout.addWidget(line)
        content = QWidget()
        content.setObjectName("inspector")
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._header())
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        column.addWidget(self.scroll, 1)
        layout.addWidget(content, 1)

    def _header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("inspectorHeader")
        box = QVBoxLayout(header)
        box.setContentsMargins(12, 12, 12, 10)
        box.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(10)
        self.tile = QLabel()
        self.tile.setObjectName("tile")
        self.tile.setFixedSize(34, 34)
        self.tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.title = QLabel()
        self.title.setObjectName("inspectorTitle")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("dim")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        top.addWidget(self.tile)
        top.addLayout(titles, 1)
        box.addLayout(top)
        self.search = QLineEdit()
        self.search.setObjectName("search")
        self.search.setClearButtonEnabled(True)
        self.search.addAction(theme.icon("magnifying-glass", theme.TEXT_3),
                              QLineEdit.ActionPosition.LeadingPosition)  # fmt: skip
        self.search.textChanged.connect(lambda _: self._search_timer.start())
        box.addWidget(self.search)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.chip_group = QButtonGroup(self)
        self.chips: dict[str, QToolButton] = {}
        for fid, label in self.FILTERS:
            chip = QToolButton()
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setProperty("label", label)
            chip.clicked.connect(lambda _=False, f=fid: self.set_filter(f))
            self.chip_group.addButton(chip)
            self.chips[fid] = chip
            chips.addWidget(chip)
        chips.addStretch()
        box.addLayout(chips)
        return header

    # --- views ------------------------------------------------------------------------
    def show_category(self, category_id: str) -> None:
        self.category = category_id
        self.item = None
        self.filter = "all"
        if self.search.text():
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
        self.rebuild()

    def show_item(self, title: str, subtitle: str, icon: str, sections: list[str],
                  top: QWidget | None = None) -> None:  # fmt: skip
        """The settings of one scene item (a track group, a POI, a photo)."""
        self.item = {"title": title, "subtitle": subtitle, "icon": icon,
                     "sections": list(sections), "top": top}  # fmt: skip
        self.filter = "all"
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.rebuild()

    def set_filter(self, fid: str) -> None:
        self.filter = "all" if fid == self.filter and fid != "all" else fid
        self.rebuild()

    def focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def jump_to(self, title: str) -> None:
        """Show the section ``title`` in its category (or its item)."""
        category = category_of(title)
        self._section_state[title] = (True, True, None)
        if category is None:
            self.search.setText(title.split(": ", 1)[-1])
            return
        self.show_category(category.id)
        section = self.sections.get(title)
        if section is not None:
            QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(section, 0, 0))

    def all_sections(self) -> list[tuple[str, list[PropertyDef]]]:
        return [(t, [d for d in defs if has_editor(d)])
                for t, defs in self.store.registry.sections()
                if any(has_editor(d) for d in defs)]  # fmt: skip

    def categories(self) -> list[Category]:
        """The rail's categories plus "More" when some section has none."""
        extra = any(category_of(t) is OTHER for t, _ in self.all_sections())
        return CATEGORIES + ([OTHER] if extra else [])

    def rebuild(self) -> None:
        """Rebuild the shown sections (category, item, or search / filter results)."""
        self._remember_sections()
        self.editors.clear()
        self.rows.clear()
        self.sections.clear()
        container = QWidget()
        container.setObjectName("panelBody")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        query = self.search.text().strip()
        if query or self.filter != "all":
            self._build_results(layout, query)
        elif self.item is not None:
            self._build_item(layout)
        else:
            self._build_category(layout)
        layout.addStretch()
        old = self.scroll.takeWidget()
        self.scroll.setWidget(container)
        if old is not None:
            old.deleteLater()
        self.refresh_key_states()
        self._update_counts()

    def _set_header(self, icon: str, title: str, subtitle: str) -> None:
        self.tile.setPixmap(theme.icon(icon, theme.ACCENT_TEXT).pixmap(QSize(20, 20)))
        self.title.setText(title)
        self.subtitle.setText(subtitle)

    def _build_category(self, layout: QVBoxLayout) -> None:
        category = next((c for c in self.categories() if c.id == self.category), CATEGORIES[0])
        self.rail.set_current(category.id)
        self._set_header(category.icon, category.title, category.subtitle)
        if category.looks:
            layout.addWidget(self._looks(category))
        for title, defs in self.all_sections():
            if category_of(title) is category:
                self._add_section(layout, title, defs, section_display(title))

    def _build_item(self, layout: QVBoxLayout) -> None:
        item = self.item
        self.rail.set_current(None)
        self._set_header(item["icon"], item["title"], item["subtitle"])
        if item["top"] is not None:
            layout.addWidget(item["top"])
        by_title = dict(self.all_sections())
        for title in item["sections"]:
            if title in by_title:
                self._add_section(layout, title, by_title[title], item_section_display(title))

    def _build_results(self, layout: QVBoxLayout, query: str) -> None:
        self.rail.set_current(None)
        terms = search_terms(query)
        tests = {
            "all": lambda d: True,
            "changed": lambda d: self.store[d.id] != d.coerce(d.default),
            "animated": lambda d: self.is_animated(d.id),
            "pinned": lambda d: d.id in self.pins,
        }
        test = tests[self.filter]
        total = groups = 0
        for title, defs in self.all_sections():
            hits = [d for d in defs if test(d) and (not terms or matches(d, terms))]
            if not hits:
                continue
            category = category_of(title)
            crumb = f"{category.title} › {section_display(title)}" if category else title
            section = self._add_section(layout, title, hits, crumb, flat=True, terms=terms)
            section.jump_requested.connect(self.jump_to)
            total += len(hits)
            groups += 1
        label = dict(self.FILTERS)[self.filter]
        title = f"Results for “{query}”" if query else f"{label} settings"
        if query and self.filter != "all":
            title += f" · {label.lower()}"
        where = f"{total} settings in {groups} sections" if total else "Nothing found"
        hint = " · German words work too" if query else " · whole scene"
        self._set_header("magnifying-glass" if query else "funnel", title, where + hint)
        if not total:
            empty = QLabel("No setting matches. Actions such as “Export video” are in the "
                           "command palette (Ctrl K).")  # fmt: skip
            empty.setObjectName("dim")
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setContentsMargins(24, 40, 24, 40)
            layout.addWidget(empty)

    def _looks(self, category: Category) -> QWidget:
        box = QWidget()
        box.setObjectName("looks")
        column = QVBoxLayout(box)
        column.setContentsMargins(12, 10, 12, 12)
        column.setSpacing(8)
        caption = QLabel("Looks · one click sets the whole category (undoable)")
        caption.setObjectName("dim")
        column.addWidget(caption)
        row = QHBoxLayout()
        row.setSpacing(6)
        for look in category.looks:
            b = QToolButton()
            b.setObjectName("look")
            b.setText(look.name)
            b.setIcon(theme.icon(look.icon))
            b.setIconSize(QSize(14, 14))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setToolTip(look.tooltip)
            b.clicked.connect(lambda _=False, lk=look: lk.apply())
            row.addWidget(b)
        row.addStretch()
        column.addLayout(row)
        return box

    def _add_section(self, layout, title: str, defs: list[PropertyDef], display: str,
                     flat: bool = False, terms=None) -> SectionView:  # fmt: skip
        def make_row(d: PropertyDef, label: str) -> PropertyRow:
            return self.make_row(d, label, terms)

        section = SectionView(title, defs, self.store, make_row, self.on_edit, self.batch,
                              display=display, flat=flat)  # fmt: skip
        if not flat:
            state = self._section_state.get(title)
            if state is None:  # map styles not on a slot start collapsed
                state = (self._relevant(title, defs), False, None)
            open_, more, tab = state
            section.set_open(open_)
            section.more_open = more
            if tab in section.tabs:
                section.tab = tab
                section.tab_bar.set_index(section.tabs.index(tab))
            section.refresh()
        layout.addWidget(section)
        self.sections[title] = section
        return section

    def _relevant(self, title: str, defs: list[PropertyDef]) -> bool:
        if title.startswith("Layer: ") and defs:
            layer = defs[0].id.split(".", 1)[0].removeprefix("layer_")
            return layer in (self.store.get("layers.a"), self.store.get("layers.b"))
        return not title.startswith("Tracks: ")  # (the groups open from the scene list)

    def _remember_sections(self) -> None:
        for title, section in self.sections.items():
            if not section.flat:
                self._section_state[title] = (section.is_open, section.more_open, section.tab)

    # --- rows -------------------------------------------------------------------------
    def make_row(self, d: PropertyDef, label: str, terms=None) -> PropertyRow:
        editor = editor_for(d)
        editor.set_value(self.store[d.id])
        editor.edited.connect(
            lambda value, interactive, pid=d.id: self.on_edit(pid, value, interactive)
        )
        extras = [self.row_extra(d) if self.row_extra is not None else None, self._pin(d)]
        row = PropertyRow(d, editor, label, extras)
        if terms:
            row.label.setTextFormat(Qt.TextFormat.RichText)
            row.label.setText(highlight(label, terms))
        row.set_changed(self.store[d.id] != d.coerce(d.default))
        self.editors[d.id] = editor
        self.rows[d.id] = row
        return row

    def _pin(self, d: PropertyDef) -> QToolButton:
        pin = QToolButton()
        pin.setObjectName("pin")
        pin.setCheckable(True)
        pin.setChecked(d.id in self.pins)
        pin.setIcon(theme.icon("star", theme.TEXT_3, theme.TEXT, on=theme.KEY))
        pin.setIconSize(QSize(14, 14))
        pin.setFixedSize(20, 20)
        pin.setToolTip("Pin: list it under the Pinned filter")
        pin.toggled.connect(lambda on, pid=d.id: self.set_pinned(pid, on))
        return pin

    def set_pinned(self, pid: str, on: bool) -> None:
        if on:
            self.pins.add(pid)
        else:
            self.pins.discard(pid)
        self.settings.setValue("inspector/pins", sorted(self.pins))
        self._counts_timer.start()

    def _load_pins(self) -> list[str]:
        value = self.settings.value("inspector/pins", [])
        if isinstance(value, str):
            value = [value]
        return list(value or [])

    # --- updates ----------------------------------------------------------------------
    def refresh_key_states(self) -> None:
        if self.key_state is None:
            return
        for pid, editor in self.editors.items():
            editor.set_key_state(self.key_state(pid))
        self._counts_timer.start()

    def _update_counts(self) -> None:
        defs = [d for _, defs in self.all_sections() for d in defs]
        counts = {
            "all": len(defs),
            "changed": sum(self.store[d.id] != d.coerce(d.default) for d in defs),
            "animated": sum(self.is_animated(d.id) for d in defs),
            "pinned": sum(d.id in self.pins for d in defs),
        }
        for fid, chip in self.chips.items():
            chip.setText(f"{chip.property('label')}  {counts[fid]}")
            chip.setChecked(fid == self.filter)
        self.search.setPlaceholderText(f"Search all {counts['all']} settings")

    def _on_store_changed(self, pid: str, value: Any) -> None:
        editor = self.editors.get(pid)
        if editor is not None:
            editor.set_value(value)
            d = self.store.registry[pid]
            self.rows[pid].set_changed(value != d.coerce(d.default))
        for section in self.sections.values():
            if section.depends_on(pid):
                section.refresh()
        self._counts_timer.start()

    def shutdown(self) -> None:
        self._unsubscribe()


def search_action(inspector: Inspector, parent) -> QAction:
    """Ctrl+F: focus the inspector's search."""
    action = QAction("Search Settings", parent)
    action.setShortcut("Ctrl+F")
    action.triggered.connect(inspector.focus_search)
    return action
