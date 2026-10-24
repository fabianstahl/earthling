"""The command palette of the main window and its entries."""

from __future__ import annotations

from pathlib import Path


class PaletteMixin:
    """Part of :class:`earthling.app.main_window.MainWindow` (its ``self``)."""

    # --- command palette ------------------------------------------------------------------
    def show_palette(self, text: str = "") -> None:
        from earthling.ui.palette import CommandPalette, mark_recent, menu_entries

        if getattr(self, "command_palette", None) is None:
            mark_recent(self.export_action, self.still_action, self.camera_key_action,
                        self.open_action, self.save_action)  # fmt: skip
            self.command_palette = CommandPalette(
                [lambda: menu_entries(self.menuBar()), self._setting_entries,
                 self._look_entries, self._help_entries], self)  # fmt: skip
        self.command_palette.open(text)

    def _setting_entries(self) -> list:
        from earthling.ui.inspector import category_of, section_display
        from earthling.ui.palette import Entry
        from earthling.ui.property_panel import format_value, has_editor

        out = []
        store = self.scene.store
        for d in store.registry:
            if not has_editor(d):
                continue
            category = category_of(d.section)
            crumb = (f"{category.title} › {section_display(d.section)}" if category
                     else d.section)  # fmt: skip
            key = (lambda pid=d.id: self.keys.set_key(pid, store[pid])) if d.animatable else None
            out.append(Entry("Settings", d.label, crumb, category.icon if category else "map-pin",
                             lambda pid=d.id: self.inspector.reveal(pid),
                             format_value(d, store[d.id]), f"{d.tooltip} {d.keywords} {d.id}",
                             key))  # fmt: skip
        return out

    def _look_entries(self) -> list:
        from earthling.ui.inspector import CATEGORIES
        from earthling.ui.palette import Entry

        titles = {c.id: c.title for c in CATEGORIES}
        return [Entry("Looks", look.name, f"Look · {titles[cid]} (undoable)", look.icon,
                      look.apply, "Look", look.tooltip)
                for cid, looks in self.looks.items() for look in looks]  # fmt: skip

    def _help_entries(self) -> list:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        from earthling.ui.palette import Entry

        docs = Path(__file__).resolve().parents[3] / "docs"
        out = [Entry("Help", "Keyboard shortcuts", "Help", "keyboard", self.show_shortcuts, "F1")]
        for name, title in (("configuration.md", "earthling.toml: every key explained"),
                            ("adding-a-region.md", "Adding a region (new data sources)"),
                            ("architecture.md", "How Earthling is built")):  # fmt: skip
            path = docs / name
            if path.exists():
                out.append(Entry("Help", title, f"docs/{name}", "book-open",
                                 lambda p=path: QDesktopServices.openUrl(QUrl.fromLocalFile(
                                     str(p)))))  # fmt: skip
        return out
