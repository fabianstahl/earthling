from datetime import datetime

import pytest

from earthling.core.properties import (
    PropertyDef,
    PropertyRegistry,
    PropertyStore,
    PType,
    bind_uniforms,
)
from earthling.render.parameters import build_registry

DEFS = [
    PropertyDef("a.f", "F", PType.FLOAT, 1.0, "A", 0.0, 10.0, uniform="u_f"),
    PropertyDef("a.i", "I", PType.INT, 2, "A", 0, 5),
    PropertyDef("b.b", "B", PType.BOOL, False, "B", uniform="u_b"),
    PropertyDef("b.e", "E", PType.ENUM, "x", "B", options=(("x", "X"), ("y", "Y")), uniform="u_e"),
    PropertyDef("b.c", "C", PType.COLOR, (1.0, 0.5, 0.0), "B", uniform="u_c"),
    PropertyDef("b.t", "T", PType.DATETIME, datetime(2026, 7, 1, 12, 0), "B"),
]


def make_store():
    return PropertyStore(PropertyRegistry(DEFS))


def test_coercion_and_clamping():
    s = make_store()
    s.set("a.f", "12.5")
    assert s["a.f"] == 10.0
    s.set("a.i", 3.6)
    assert s["a.i"] == 4
    s.set("b.b", "yes")
    assert s["b.b"] is True
    with pytest.raises(ValueError):
        s.set("b.e", "z")
    s.set("b.c", [2, -1, 0.25])
    assert s["b.c"] == (1.0, 0.0, 0.25)
    s.set("b.t", "2026-07-02T06:30:00")
    assert s["b.t"] == datetime(2026, 7, 2, 6, 30)


def test_listeners_and_change_detection():
    s = make_store()
    seen = []
    unsubscribe = s.subscribe(lambda pid, v: seen.append((pid, v)))
    assert s.set("a.f", 2.0)
    assert not s.set("a.f", 2.0)
    unsubscribe()
    s.set("a.f", 3.0)
    assert seen == [("a.f", 2.0)]


def test_json_roundtrip_and_problems():
    s = make_store()
    s.set("b.t", datetime(2030, 1, 1, 5, 0))
    s.set("b.c", (0.1, 0.2, 0.3))
    data = s.to_json()
    t = make_store()
    assert t.load_json(data) == []
    assert t.values == s.values
    problems = t.load_json({"nope": 1, "b.e": "zzz"})
    assert len(problems) == 2


def test_registry_rejects_duplicates_and_bad_defaults():
    with pytest.raises(ValueError):
        PropertyRegistry([DEFS[0], DEFS[0]])
    with pytest.raises(ValueError):
        PropertyRegistry([PropertyDef("x.e", "E", PType.ENUM, "q", options=(("a", "A"),))])


def test_sections_keep_declaration_order():
    assert [name for name, _ in PropertyRegistry(DEFS).sections()] == ["A", "B"]


class FakeUniform:
    def __init__(self):
        self.value = None


class FakeProgram(dict):
    pass


def test_bind_uniforms():
    s = make_store()
    s.set("b.e", "y")
    program = FakeProgram(u_f=FakeUniform(), u_e=FakeUniform(), u_c=FakeUniform())
    bind_uniforms(program, s)
    assert program["u_f"].value == 1.0
    assert program["u_e"].value == 1
    assert program["u_c"].value[1] == 0.5  # sRGB: the shaders linearise


def test_scene_registry_is_valid():
    registry = build_registry()
    assert "terrain.exaggeration" in registry
    PropertyStore(registry)


def test_property_panel_two_way_binding(qtbot):
    from earthling.ui.property_panel import PropertyPanel

    s = make_store()
    panel = PropertyPanel(s)
    qtbot.addWidget(panel)
    editor = panel.editors["a.f"]
    editor.commit(4.5)
    assert s["a.f"] == 4.5
    assert editor.field.fraction() == pytest.approx(0.45)
    assert panel.rows["a.f"].changed  # differs from the default
    s.set("a.f", 7.0)
    assert editor.field.value == 7.0
    panel.editors["b.e"].segments.buttons[1].click()  # two short choices: segments
    assert s["b.e"] == "y"
    panel.editors["b.b"].box.setChecked(True)
    assert s["b.b"] is True
    assert set(panel.sections) == {"A", "B"}


def test_drag_field_drags_types_and_steps(qtbot):
    from PyQt6.QtCore import QPoint, Qt

    from earthling.ui.property_panel import DragField

    field = DragField(0.0, 10.0, step=0.5, decimals=1, unit="m")
    qtbot.addWidget(field)
    field.resize(200, 24)
    field.set_value(2.0)
    seen = []
    field.edited.connect(lambda v, interactive: seen.append((v, interactive)))
    qtbot.mousePress(field, Qt.MouseButton.LeftButton, pos=QPoint(50, 12))
    qtbot.mouseMove(field, QPoint(90, 12))  # 40 of 200 px: a fifth of the range
    qtbot.mouseRelease(field, Qt.MouseButton.LeftButton, pos=QPoint(90, 12))
    assert seen[-2] == (4.0, True) and seen[-1] == (4.0, False)
    assert field.text() == "4.0 m"
    qtbot.keyClick(field, Qt.Key.Key_Up)
    assert field.value == 4.5
    field.start_typing()  # a click without moving types the value
    field._edit.setText("7,25")
    field._edit.editingFinished.emit()
    assert field.value == 7.2 or field.value == 7.3  # one decimal
    field.commit(99)
    assert field.value == 10.0  # clamped


def test_sections_follow_when_advanced_switch_and_tabs(qtbot):
    from earthling.core.properties import boolean, flt, section, shown_if
    from earthling.ui.property_panel import PropertyPanel

    defs = section(
        "Fog",
        boolean("fog.on", "Fog", False, switch=True),
        flt("fog.density", "Density", 0.3, 0.0, 1.0),
        flt("fog.glow", "Glow", 0.5, 0.0, 1.0, advanced=True),
    ) + section(
        "Clouds",
        boolean("low.on", "Low layer", True),
        flt("low.base", "Low layer: base", 900.0, 0.0, 9000.0, when=shown_if("low.on")),
        boolean("high.on", "High layer", False),
        flt("high.base", "High layer: base", 7000.0, 0.0, 9000.0, when=shown_if("high.on")),
        flt("cover", "Coverage", 0.5, 0.0, 1.0),
    )
    store = PropertyStore(PropertyRegistry(defs))
    panel = PropertyPanel(store)
    qtbot.addWidget(panel)
    panel.show()
    fog, clouds = panel.sections["Fog"], panel.sections["Clouds"]
    assert not fog.body.isVisible() and fog.off_note.isVisible()  # switched off
    fog.switch.click()
    assert store["fog.on"] and fog.body.isVisible()
    assert panel.rows["fog.density"].isVisible() and not panel.rows["fog.glow"].isVisible()
    fog.more.click()
    assert panel.rows["fog.glow"].isVisible()
    assert clouds.tabs == ["Low layer", "High layer"]
    assert panel.rows["low.base"].isVisible() and not panel.rows["high.base"].isVisible()
    assert panel.rows["low.base"].label.text() == "Base"
    clouds.tab_bar.buttons[1].click()
    assert not panel.rows["high.base"].isVisible()  # the high layer is off
    store.set("high.on", True)
    assert panel.rows["high.base"].isVisible() and panel.rows["cover"].isVisible()
    fog.reset_all()
    assert not store["fog.on"]
