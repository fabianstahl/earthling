"""What the scene list selects and adds in the main window: track groups, POIs, photos,
GPX files; places on the map and the camera path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from earthling.app.scene_list import PHOTO_SUFFIXES
from earthling.core.camera_path import camera_path_lines
from earthling.core.poi_library import KINDS


class ItemsMixin:
    """Part of :class:`earthling.app.main_window.MainWindow` (its ``self``)."""

    # --- points of interest ---------------------------------------------------------------
    def _scene_dir(self) -> Path | None:
        if self.scene.path is not None:
            return self.scene.path.parent
        return self.project.folder if self.project else None

    def _pois_changed(self) -> None:
        self.viewport.set_pois(self.scene.pois, self._scene_dir(), self.scene.animation)
        self.viewport.set_photos(self.scene.photos, self._scene_dir())

    def _reveal_item(self, title: str) -> bool:
        """Select the POI or photo that owns the section ``title``."""
        for kind, items, prefix in (("poi", self.scene.pois, "POI: "),
                                    ("photo", self.scene.photos, "Photo: ")):  # fmt: skip
            for i, item in enumerate(items):
                if title == f"{prefix}{item.name}":
                    self.scene_list.select((kind, i))
                    return True
        return False

    def _select_item(self, kind: str, index: int) -> None:
        """An item was added or removed: select it (or its neighbour) in the scene list."""
        self.scene_list.refresh()
        if index >= 0:
            self.scene_list.select((kind, index))
        elif self.inspector.item is not None:
            self.inspector.show_category(self.inspector.category)

    def _scene_selected(self, key: tuple) -> None:
        """The inspector shows what was selected in the scene list."""
        kind = key[0]
        if kind == "camera":
            self.inspector.show_category("camera")
        elif kind == "layers":
            self.inspector.show_category("layers")
        elif kind == "header":
            category = {"Tracks": "tracks", "Map layers": "layers"}.get(key[1])
            if category:
                self.inspector.show_category(category)
        elif kind in ("group", "track"):
            self._show_group(key)
        elif kind in ("poi", "photo"):
            self.show_scene_item(kind, key[1])

    def _show_group(self, key: tuple) -> None:
        session = self.session
        if session is None:
            return
        if key[0] == "track":
            track = self.tracks[key[1]]
            group = next(g for g in session.groups if track in g.tracks)
            s = track.stats
            title = track.name
            subtitle = (f"Track of {group.label} · {s.distance_m / 1000:.1f} km · "
                        f"↑ {s.ascent_m:.0f} m")  # fmt: skip
        else:
            group = session.group(key[1])
            total = sum(t.stats.distance_m for t in group.tracks)
            title = group.label
            subtitle = f"Track group · {len(group.tracks)} tracks · {total / 1000:.1f} km"
        sections = [f"Tracks: {group.label}", f"Tracks: {group.label} – colours"]
        self.inspector.show_item(title, subtitle, "path", sections)

    def show_scene_item(self, kind: str, index: int) -> None:
        """The inspector shows the form and the settings of a POI or a photo."""
        editor = self.poi_editor if kind == "poi" else self.photo_editor
        items = editor.items()
        if not 0 <= index < len(items):
            return
        item = items[index]
        editor.select(index)
        if kind == "poi":
            what = next((k.name for k in KINDS if k.icon == item.icon), "custom icon")
            self.inspector.show_item(item.name, f"Point of interest · {what}", "map-pin",
                                     [f"POI: {item.name}"], top=editor)  # fmt: skip
        else:
            self.inspector.show_item(item.name, f"Photo · {item.path}", "image",
                                     [f"Photo: {item.name}"], top=editor)  # fmt: skip

    def add_gpx_dialog(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "Add GPX files", "Open or create a project first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add GPX files", str(Path.home()),
                                                "GPX files (*.gpx)")  # fmt: skip
        self.add_gpx_files([Path(p) for p in paths])

    def add_gpx_files(self, paths: list[Path]) -> None:
        """Copy GPX files into the project's GPX folder and load them."""
        import shutil

        if self.project is None or not paths:
            return
        folder = self.project.gpx_dir
        if not folder.is_dir():  # (a single file or a pattern: the files go next to it)
            folder = folder.parent
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            target = folder / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
        self._reload_project()

    def drop_paths(self, paths: list[Path]) -> None:
        """Dropped files: a project folder opens, GPX files join the project, pictures
        become photos."""
        for folder in [p for p in paths if p.is_dir()]:
            if (folder / "earthling.toml").exists():
                self.open_project(folder)
                return
        gpx = [p for p in paths if p.suffix.lower() == ".gpx"]
        if gpx:
            self.add_gpx_files(gpx)
        for picture in [p for p in paths if p.suffix.lower() in PHOTO_SUFFIXES]:
            self.photo_editor.add_photo(picture)

    def hiker_lonlat(self) -> tuple[float, float] | None:
        """(lon, lat) of the head of the drawn track."""
        renderer = self.viewport.renderer
        if renderer is None or renderer.tracks.path is None or self.session is None:
            return None
        s = self.scene.store
        path = renderer.tracks.path
        enu = path.position_at(path.distance_for(s["progress.head"], s["progress.mode"]))
        if enu is None:
            return None
        lat, lon, _ = self.session.frame.enu_to_geodetic(enu)
        return float(lon), float(lat)

    def lonlat_at_time(self, when) -> tuple[float, float] | None:
        """(lon, lat) of the walked tracks at a time (naive: the project's local time)."""
        from zoneinfo import ZoneInfo

        renderer = self.viewport.renderer
        if renderer is None or renderer.tracks.path is None or self.session is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=ZoneInfo(self.session.config.project.timezone))
        path = renderer.tracks.path
        t = path.time[np.isfinite(path.time)]
        epoch = when.timestamp()
        if not len(t) or not t[0] - 3600.0 <= epoch <= t[-1] + 3600.0:
            return None  # (not during the tracks)
        enu = path.position_at(path.distance_for_time(epoch))
        lat, lon, _ = self.session.frame.enu_to_geodetic(enu)
        return float(lon), float(lat)

    def built_tracks(self) -> list:
        """The tracks as drawn (BuiltTrack: ENU points, distances, times), [] before a project."""
        renderer = self.viewport.renderer
        return list(renderer.tracks._gpu) if renderer is not None else []

    def choose_place(self, title: str, start: tuple[float, float] | None):
        """(lon, lat) from the place dialog (coordinates or a track point), None if cancelled."""
        from earthling.app.place_dialog import PlaceDialog

        if self.session is None:
            return None
        dialog = PlaceDialog(title, self.built_tracks(), self.session.frame, start, self)
        return dialog.lonlat() if dialog.exec() else None

    def _pick_lonlat(self, callback) -> None:
        self.statusBar().showMessage("Click on the terrain to place the POI", 5000)

        def picked(point) -> None:
            if self.session is not None:
                lat, lon, _ = self.session.frame.enu_to_geodetic(point)
                callback(float(lon), float(lat))

        self.viewport.request_pick(picked)

    def pick_look_at_target(self) -> None:
        self.statusBar().showMessage("Click on the terrain to set the look-at target", 5000)

        def picked(point) -> None:
            self._on_panel_edit("camera.target", tuple(float(v) for v in point), False)
            self.statusBar().showMessage("Look-at target set", 3000)

        self.viewport.request_pick(picked)

    def _animated_pose(self):
        renderer = self.viewport.renderer
        self.rig.path = renderer.tracks.path if renderer is not None else None
        return self.viewport.clamp_pose(self.rig.pose(self.timeline.time))

    def _camera_keys_changed(self) -> None:
        curve = self.scene.animation.curves.get("camera.pose")
        self.viewport.set_camera_path(camera_path_lines(curve))

    def _frame_track(self, index: int) -> None:
        if self.session is None or not (0 <= index < len(self.tracks)):
            return
        track = self.tracks[index]
        lat, lon, ele = track.all_points()
        enu = self.session.frame.geodetic_to_enu(lat, lon, np.nan_to_num(ele))
        self.viewport.frame_box(enu.min(axis=0), enu.max(axis=0))
