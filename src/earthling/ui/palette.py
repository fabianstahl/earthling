"""The command palette (Ctrl K): one search over settings, actions, looks and help.

Entries come from providers (callables returning ``Entry`` lists), so the palette knows
nothing about the application: the main window passes its menu actions, the settings and the
looks. Enter runs the entry; Ctrl+Enter keys a setting at the playhead; ">" searches actions
only.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.ui import theme
from earthling.ui.inspector import normalise, search_terms

GROUPS = ("Settings", "Actions", "Looks", "Help")


@dataclass
class Entry:
    group: str  # one of GROUPS
    title: str
    crumb: str  # where it lives
    icon: str | QIcon  # a theme icon name or an icon
    run: Callable[[], None]
    right: str = ""  # current value or shortcut
    words: str = ""  # more text to search (tooltip, keywords)
    key: Callable[[], None] | None = None  # Ctrl+Enter (settings: key at the playhead)
    recent: bool = False  # listed while nothing is typed


def menu_text(text: str) -> str:
    """A menu entry's text as shown: without the mnemonic marks ("&&" is a literal "&")."""
    return "&".join(part.replace("&", "") for part in text.split("&&")).split("\t")[0]


def menu_entries(menu_bar) -> list[Entry]:
    """Every enabled action of the menus (with its menu path and shortcut)."""
    out: list[Entry] = []

    def walk(menu, path: list[str]) -> None:
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = menu_text(action.text())
            if action.menu() is not None:
                walk(action.menu(), [*path, text])
            elif action.isEnabled() and text:
                icon = action.icon() if not action.icon().isNull() else "command"
                keys = ", ".join(k.toString() for k in action.shortcuts())
                out.append(Entry("Actions", text, " › ".join(path), icon, action.trigger,
                                 keys, action.toolTip(),
                                 recent=action.property("palette_recent") is True))  # fmt: skip

    for action in menu_bar.actions():
        if action.menu() is not None:
            walk(action.menu(), [menu_text(action.text())])
    return out


def mark_recent(*actions: QAction) -> None:
    """Actions listed when the palette opens."""
    for action in actions:
        action.setProperty("palette_recent", True)


def score(entry: Entry, terms: list[list[str]]) -> int:
    """0: no match; higher is better (title start, a word start, the title, other text)."""
    if not terms:
        return 1
    title = normalise(entry.title)
    rest = normalise(f"{entry.crumb} {entry.words}")
    total = 0
    for alternatives in terms:
        best = 0
        for t in alternatives:
            if title.startswith(t):
                best = max(best, 40)
            elif f" {t}" in f" {title}":
                best = max(best, 30)
            elif t in title:
                best = max(best, 20)
            elif t in rest:
                best = max(best, 8)
        if not best:
            return 0
        total += best
    return total


def marked(text: str, terms: list[list[str]]) -> str:
    low = normalise(text)
    for alternatives in terms:
        for t in alternatives:
            i = low.find(t)
            if i >= 0:
                return (html.escape(text[:i]) + '<span style="background:#2A4A75;color:#FFF">'
                        + html.escape(text[i:i + len(t)]) + "</span>"
                        + html.escape(text[i + len(t):]))  # fmt: skip
    return html.escape(text)


class EntryWidget(QWidget):
    def __init__(self, entry: Entry, terms, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 12, 5)
        layout.setSpacing(12)
        tile = QLabel()
        tile.setObjectName("paletteTile")
        tile.setFixedSize(28, 28)
        tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = entry.icon if isinstance(entry.icon, QIcon) else theme.icon(entry.icon)
        tile.setPixmap(icon.pixmap(QSize(16, 16)))
        texts = QVBoxLayout()
        texts.setSpacing(1)
        title = QLabel(marked(entry.title, terms))
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setObjectName("paletteTitle")
        crumb = QLabel(entry.crumb)
        crumb.setObjectName("dim")
        texts.addWidget(title)
        texts.addWidget(crumb)
        right = QLabel(entry.right)
        right.setObjectName("paletteRight")
        layout.addWidget(tile)
        layout.addLayout(texts, 1)
        layout.addWidget(right)


class CommandPalette(QDialog):
    def __init__(self, providers: list[Callable[[], list[Entry]]], parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)
        self.setObjectName("palette")
        self.providers = providers
        self.entries: list[Entry] = []
        self.shown: list[Entry] = []
        self.scope = "All"
        self.setMinimumWidth(680)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("paletteFrame")
        outer.addWidget(frame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.input = QLineEdit()
        self.input.setObjectName("paletteInput")
        self.input.setPlaceholderText("Search settings, actions, looks and help…")
        self.input.addAction(theme.icon("magnifying-glass", theme.TEXT_3),
                             QLineEdit.ActionPosition.LeadingPosition)  # fmt: skip
        self.input.textChanged.connect(lambda _: self.update_list())
        self.input.installEventFilter(self)
        layout.addWidget(self.input)
        scopes = QHBoxLayout()
        scopes.setContentsMargins(16, 8, 16, 8)
        scopes.setSpacing(6)
        self.scope_group = QButtonGroup(self)
        self.scope_buttons: dict[str, QToolButton] = {}
        for name in ("All", *GROUPS):
            b = QToolButton()
            b.setObjectName("chip")
            b.setText(name)
            b.setCheckable(True)
            b.setChecked(name == "All")
            b.clicked.connect(lambda _=False, n=name: self.set_scope(n))
            self.scope_group.addButton(b)
            self.scope_buttons[name] = b
            scopes.addWidget(b)
        scopes.addStretch()
        layout.addLayout(scopes)
        self.list = QListWidget()
        self.list.setObjectName("paletteList")
        self.list.itemActivated.connect(lambda _: self.run_current())
        self.list.itemClicked.connect(lambda _: self.run_current())
        layout.addWidget(self.list, 1)
        foot = QLabel("↑↓ move   ·   Enter open or run   ·   Ctrl+Enter key at the playhead   ·"
                      "   > actions only   ·   Esc close")  # fmt: skip
        foot.setObjectName("paletteFoot")
        layout.addWidget(foot)

    # --- content ----------------------------------------------------------------------
    def open(self, text: str = "") -> None:  # noqa: A003 - Qt name
        self.entries = [e for provider in self.providers for e in provider()]
        self.input.setText(text)
        self.update_list()
        parent = self.parentWidget()
        if parent is not None:
            width = min(720, parent.width() - 80)
            self.resize(width, min(600, parent.height() - 140))
            top_left = parent.mapToGlobal(parent.rect().topLeft())
            self.move(top_left.x() + (parent.width() - width) // 2, top_left.y() + 90)
        self.show()
        self.input.setFocus()

    def set_scope(self, scope: str) -> None:
        self.scope = scope
        self.scope_buttons[scope].setChecked(True)
        self.update_list()

    def results(self, text: str) -> list[tuple[str, list[Entry]]]:
        """(group title, entries) to show for ``text``."""
        scope = self.scope
        if text.startswith(">"):
            scope, text = "Actions", text[1:]
        terms = search_terms(text)
        pool = [e for e in self.entries if scope == "All" or e.group == scope]
        if not terms:
            recent = [e for e in pool if e.recent] if scope == "All" else pool[:40]
            return [("Recent actions" if scope == "All" else scope, recent)]
        scored = [(score(e, terms), i, e) for i, e in enumerate(pool)]
        hits = [e for s, _, e in sorted((x for x in scored if x[0]), key=lambda x: (-x[0], x[1]))]
        return [(g, [e for e in hits if e.group == g][:8]) for g in GROUPS
                if any(e.group == g for e in hits)]  # fmt: skip

    def update_list(self) -> None:
        text = self.input.text()
        terms = search_terms(text.lstrip(">"))
        self.list.clear()
        self.shown = []
        for title, entries in self.results(text):
            heading = QListWidgetItem(title.upper())
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            heading.setData(Qt.ItemDataRole.UserRole, None)
            self.list.addItem(heading)
            for entry in entries:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, len(self.shown))
                widget = EntryWidget(entry, terms)
                item.setSizeHint(QSize(0, 46))
                self.list.addItem(item)
                self.list.setItemWidget(item, widget)
                self.shown.append(entry)
        if not self.shown:
            empty = QListWidgetItem("Nothing found. Try another word; German works too.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)
        self._select_step(0)

    def current(self) -> Entry | None:
        item = self.list.currentItem()
        index = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return self.shown[index] if index is not None else None

    def _select_step(self, step: int) -> None:
        rows = [i for i in range(self.list.count())
                if self.list.item(i).data(Qt.ItemDataRole.UserRole) is not None]  # fmt: skip
        if not rows:
            return
        current = self.list.currentRow()
        if step == 0 or current not in rows:
            target = rows[0]
        else:
            position = rows.index(current) + step
            target = rows[max(0, min(len(rows) - 1, position))]
        self.list.setCurrentRow(target)

    def run_current(self, key: bool = False) -> None:
        entry = self.current()
        if entry is None:
            return
        self.hide()
        if key and entry.key is not None:
            entry.key()
        else:
            entry.run()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        pressed = isinstance(event, QKeyEvent) and event.type() == event.Type.KeyPress
        if obj is self.input and pressed:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._select_step(1 if key == Qt.Key.Key_Down else -1)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.run_current(key=ctrl)
                return True
            if key == Qt.Key.Key_Escape:
                self.hide()
                return True
        return super().eventFilter(obj, event)
