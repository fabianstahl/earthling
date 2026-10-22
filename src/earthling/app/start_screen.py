"""The start screen: a new project from GPX files (drop them or choose them), open a project,
the demo, recent projects with a picture of their last view, and three steps to a video."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PyQt6.QtCore import QSize, QStandardPaths, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from earthling.ui import theme
from earthling.ui.property_panel import SwitchButton

DEMO = Path(__file__).resolve().parents[3] / "examples" / "alps_demo"
STEPS = [
    ("path", "Add your tracks", "GPX files become tracks in the scene list; photos find their "
     "place by their time."),
    ("video-camera", "Set up the camera", "Generate a flyover, or press K at good views to "
     "key them."),
    ("film-strip", "Export", "Pick a look, check the export frame and export up to 4K."),
]  # fmt: skip


def thumbnail_path(folder: str | Path) -> Path:
    """Where the picture of a project's last view is kept (not in the project)."""
    base = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
    digest = hashlib.sha1(str(Path(folder).resolve()).encode("utf-8")).hexdigest()[:16]
    return base / "thumbnails" / f"{digest}.jpg"


class StartScreen(QWidget):
    open_requested = pyqtSignal(str)  # a project folder
    new_requested = pyqtSignal(list)  # GPX files (Path) for a new project
    choose_new = pyqtSignal()  # pick GPX files for a new project
    browse_requested = pyqtSignal()  # pick a project folder
    continue_requested = pyqtSignal()  # the empty application
    shortcuts_requested = pyqtSignal()

    def __init__(self, recent: list[str], show_at_start: bool, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("startScreen")
        self.setAcceptDrops(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._left(show_at_start))
        self.right: QWidget | None = None
        self.set_recent(recent)

    # --- left column ------------------------------------------------------------------
    def _left(self, show_at_start: bool) -> QWidget:
        left = QWidget()
        left.setObjectName("startLeft")
        left.setFixedWidth(500)
        column = QVBoxLayout(left)
        column.setContentsMargins(48, 56, 48, 36)
        column.setSpacing(26)
        brand = QHBoxLayout()
        logo = QLabel()
        logo.setObjectName("startLogo")
        logo.setFixedSize(46, 46)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(theme.icon("globe-hemisphere-west", theme.ACCENT_TEXT).pixmap(26, 26))
        names = QVBoxLayout()
        names.setSpacing(0)
        name = QLabel("Earthling")
        name.setObjectName("startName")
        tag = QLabel("Cinematic 3D flyovers of your GPS tracks")
        tag.setObjectName("dim")
        names.addWidget(name)
        names.addWidget(tag)
        brand.addWidget(logo)
        brand.addSpacing(10)
        brand.addLayout(names, 1)
        column.addLayout(brand)
        column.addWidget(self._drop_zone())
        for icon, title, text, signal in (
            ("folder-open", "Open a project folder", "A folder with an earthling.toml",
             self.browse_requested),
            ("mountains", "Try the Alps demo", "Two days on the Tour du Mont Blanc", None),
        ):  # fmt: skip
            if signal is None and not DEMO.exists():
                continue
            button = self._option(icon, title, text)
            if signal is None:
                button.clicked.connect(lambda: self.open_requested.emit(str(DEMO)))
            else:
                button.clicked.connect(signal.emit)
            column.addWidget(button)
        column.addStretch()
        foot = QHBoxLayout()
        foot.setSpacing(14)
        for icon, text, slot in (("keyboard", "Shortcuts", self.shortcuts_requested.emit),
                                 ("arrow-right", "Continue without a project",
                                  self.continue_requested.emit)):  # fmt: skip
            link = QToolButton()
            link.setObjectName("startLink")
            link.setIcon(theme.icon(icon))
            link.setText(text)
            link.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            link.clicked.connect(slot)
            foot.addWidget(link)
        foot.addStretch()
        column.addLayout(foot)
        startup = QHBoxLayout()
        label = QLabel("Show at startup")
        label.setObjectName("dim")
        self.show_switch = SwitchButton()
        self.show_switch.setChecked(show_at_start)
        startup.addStretch()
        startup.addWidget(label)
        startup.addWidget(self.show_switch)
        column.addLayout(startup)
        return left

    def _drop_zone(self) -> QWidget:
        zone = self.drop_zone = QFrame()
        zone.setObjectName("dropZone")
        box = QVBoxLayout(zone)
        box.setContentsMargins(24, 30, 24, 30)
        box.setSpacing(12)
        box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel()
        icon.setObjectName("startLogo")
        icon.setFixedSize(56, 56)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setPixmap(theme.icon("file-plus", theme.ACCENT_TEXT).pixmap(28, 28))
        title = QLabel("New project from GPX files")
        title.setObjectName("startTitle")
        text = QLabel("Drop .gpx files here or choose them. Earthling plans the area, downloads "
                      "terrain and imagery and frames the camera.")  # fmt: skip
        text.setObjectName("dim")
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        choose = QPushButton(theme.icon("upload-simple", "#06101F", "#06101F"),
                             "Choose GPX files…")  # fmt: skip
        choose.setProperty("accent", True)
        choose.setIconSize(QSize(16, 16))
        choose.clicked.connect(self.choose_new.emit)
        for widget in (icon, title, text):
            box.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        box.addWidget(choose, 0, Qt.AlignmentFlag.AlignHCenter)
        return zone

    def _option(self, icon: str, title: str, text: str) -> QPushButton:
        button = QPushButton()
        button.setObjectName("startOption")
        row = QHBoxLayout(button)
        row.setContentsMargins(12, 10, 12, 10)
        tile = QLabel()
        tile.setObjectName("paletteTile")
        tile.setFixedSize(32, 32)
        tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tile.setPixmap(theme.icon(icon).pixmap(18, 18))
        texts = QVBoxLayout()
        texts.setSpacing(0)
        bold = QLabel(title)
        bold.setObjectName("panelTitle")
        small = QLabel(text)
        small.setObjectName("dim")
        texts.addWidget(bold)
        texts.addWidget(small)
        arrow = QLabel()
        arrow.setPixmap(theme.icon("arrow-right", theme.TEXT_3).pixmap(16, 16))
        row.addWidget(tile)
        row.addSpacing(6)
        row.addLayout(texts, 1)
        row.addWidget(arrow)
        button.setMinimumHeight(58)
        return button

    # --- right column -----------------------------------------------------------------
    def set_recent(self, recent: list[str]) -> None:
        """Build the right column again (recent projects and the steps)."""
        right = QWidget()
        right.setObjectName("startRight")
        layout = QVBoxLayout(right)
        layout.setContentsMargins(56, 56, 56, 40)
        layout.setSpacing(18)
        if self.right is not None:
            self.layout().replaceWidget(self.right, right)
            self.right.deleteLater()
        else:
            self.layout().addWidget(right, 1)
        self.right = right
        projects = [p for p in recent if Path(p).exists()][:4]
        heading = QLabel("Recent projects" if projects else "No recent projects yet")
        heading.setObjectName("startHeading")
        layout.addWidget(heading)
        grid = QGridLayout()
        grid.setSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for i, folder in enumerate(projects):
            grid.addWidget(self._card(folder), i // 2, i % 2)
        layout.addLayout(grid)
        layout.addSpacing(8)
        steps_heading = QLabel("Three steps to a video")
        steps_heading.setObjectName("startHeading")
        layout.addWidget(steps_heading)
        steps = QHBoxLayout()
        steps.setSpacing(12)
        for n, (icon, title, text) in enumerate(STEPS, start=1):
            steps.addWidget(self._step(n, icon, title, text))
        layout.addLayout(steps)
        layout.addStretch()

    def _card(self, folder: str) -> QPushButton:
        card = QPushButton()
        card.setObjectName("projectCard")
        card.setToolTip(folder)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box = QVBoxLayout(card)
        box.setContentsMargins(0, 0, 0, 12)
        box.setSpacing(4)
        picture = QLabel()
        picture.setObjectName("cardPicture")
        picture.setFixedHeight(150)
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        path = thumbnail_path(folder)
        pixmap = QPixmap(str(path)) if path.exists() else QPixmap()
        if not pixmap.isNull():
            fill = Qt.AspectRatioMode.KeepAspectRatioByExpanding
            smooth = Qt.TransformationMode.SmoothTransformation
            picture.setPixmap(pixmap.scaled(420, 150, fill, smooth))
        else:
            picture.setPixmap(theme.icon("mountains", theme.TEXT_3).pixmap(36, 36))
        title = QLabel(Path(folder).name)
        title.setObjectName("panelTitle")
        title.setContentsMargins(14, 6, 14, 0)
        where = QLabel(str(Path(folder).parent))
        where.setObjectName("dim")
        where.setContentsMargins(14, 0, 14, 0)
        box.addWidget(picture)
        box.addWidget(title)
        box.addWidget(where)
        card.setMinimumHeight(212)
        card.clicked.connect(lambda: self.open_requested.emit(folder))
        return card

    @staticmethod
    def _step(n: int, icon: str, title: str, text: str) -> QWidget:
        step = QFrame()
        step.setObjectName("startStep")
        row = QHBoxLayout(step)
        row.setContentsMargins(14, 14, 14, 14)
        row.setSpacing(12)
        number = QLabel(str(n))
        number.setObjectName("stepNumber")
        number.setFixedSize(28, 28)
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        texts = QVBoxLayout()
        texts.setSpacing(4)
        head = QLabel(title)
        head.setObjectName("panelTitle")
        body = QLabel(text)
        body.setObjectName("dim")
        body.setWordWrap(True)
        texts.addWidget(head)
        texts.addWidget(body)
        row.addWidget(number, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(texts, 1)
        return step

    # --- drops ------------------------------------------------------------------------
    def set_drop_active(self, on: bool) -> None:
        """The drop zone lights up while files are dragged over the screen."""
        self.drop_zone.setProperty("active", on)
        self.drop_zone.style().unpolish(self.drop_zone)
        self.drop_zone.style().polish(self.drop_zone)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.set_drop_active(True)

    def dragLeaveEvent(self, event) -> None:
        self.set_drop_active(False)

    def dropEvent(self, event) -> None:
        self.set_drop_active(False)
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        folders = [p for p in paths if p.is_dir() and (p / "earthling.toml").exists()]
        if folders:
            self.open_requested.emit(str(folders[0]))
        elif gpx := [p for p in paths if p.suffix.lower() == ".gpx"]:
            self.new_requested.emit(gpx)
        event.acceptProposedAction()
