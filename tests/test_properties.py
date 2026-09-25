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
    assert program["u_c"].value[1] == pytest.approx(0.5**2.2)


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
    editor.spin.setValue(4.5)
    assert s["a.f"] == 4.5
    assert editor.slider.value() == 450
    s.set("a.f", 7.0)
    assert editor.spin.value() == 7.0
    panel.editors["b.e"].combo.setCurrentIndex(1)
    assert s["b.e"] == "y"
    panel.editors["b.b"].box.setChecked(True)
    assert s["b.b"] is True
    assert set(panel.sections) == {"A", "B"}
