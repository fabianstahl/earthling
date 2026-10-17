"""Helpers for repeated Qt widget boilerplate (rows, spin boxes, tool buttons)."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QToolButton, QWidget


def tight_hbox(parent: QWidget) -> QHBoxLayout:
    """A QHBoxLayout without margins on ``parent``."""
    layout = QHBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    return layout


def row(*items: QWidget | tuple[QWidget, int]) -> QWidget:
    """A widget showing ``items`` side by side; ``(widget, stretch)`` for stretching ones."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    for item in items:
        w, stretch = item if isinstance(item, tuple) else (item, 0)
        layout.addWidget(w, stretch)
    return widget


def double_spin(
    lo: float,
    hi: float,
    *,
    decimals: int | None = None,
    suffix: str = "",
    value: float | None = None,
    tracking: bool = True,
) -> QDoubleSpinBox:
    """QDoubleSpinBox set up in the order range, decimals, suffix, value, keyboard tracking."""
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    if decimals is not None:
        spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    if value is not None:
        spin.setValue(value)
    if not tracking:
        spin.setKeyboardTracking(False)
    return spin


def tool_button(text: str, tip: str, slot: Callable, checked: bool | None = None) -> QToolButton:
    """``checked`` given: a checkable button and ``slot`` gets ``toggled``, else ``clicked``."""
    btn = QToolButton()
    btn.setText(text)
    btn.setToolTip(tip)
    if checked is None:
        btn.clicked.connect(slot)
    else:
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.toggled.connect(slot)
    return btn
