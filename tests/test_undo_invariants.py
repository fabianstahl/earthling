"""Undo and redo, tried with random sequences of real edits (Hypothesis): values, keys set,
deleted, moved and re-interpolated, looks, grouped edits. Undoing everything gives back the
scene exactly, redoing everything the edited one."""

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from earthling.core.animation import Interp

# a few settings of each kind (number, switch, choice, colour), animatable or not
PIDS = ["fog.density", "tracks.width", "sky.haze", "view.fov", "snow.cover", "labels.visible",
        "weather.enabled", "progress.mode", "tracks.color", "glow.intensity"]  # fmt: skip
TIMES = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]


@pytest.fixture(scope="module")
def window(qapp):
    from earthling.app.main_window import MainWindow

    w = MainWindow()
    w.scene.animation.duration = 6.0
    yield w
    w.close()


def value_of(d):
    """A strategy for valid values of the setting ``d``."""
    kind = type(d.default)
    if getattr(d, "options", None):
        return st.sampled_from([key for key, _ in d.options])
    if kind is bool:
        return st.booleans()
    if kind is tuple:
        return st.tuples(*[st.floats(0.0, 1.0).map(lambda v: round(v, 3))] * 3)
    low = d.minimum if d.minimum is not None else -100.0
    high = d.maximum if d.maximum is not None else 100.0
    return st.floats(low, high).map(lambda v: round(v, 4))


@st.composite
def edit(draw, registry):
    pid = draw(st.sampled_from([p for p in PIDS if p in registry]))
    d = registry[pid]
    kind = draw(st.sampled_from(["set", "key", "delete", "move", "interp", "look", "batch"]))
    return (kind, pid, draw(value_of(d)), draw(st.sampled_from(TIMES)),
            draw(st.sampled_from([-1.0, -0.5, 0.5, 1.0])), draw(st.sampled_from(list(Interp))),
            draw(st.integers(0, 50)))  # fmt: skip


def apply(window, step) -> None:
    kind, pid, value, t, dt, interp, n = step
    keys, store = window.keys, window.scene.store
    window.timeline.set_time(t)
    animatable = store.registry[pid].animatable
    existing = [(pid, k.time) for k in getattr(window.scene.animation.curves.get(pid), "keys", [])]
    if kind == "set":
        window.set_property(pid, value)
    elif kind == "key" and animatable:
        keys.set_key(pid, value)
    elif kind == "delete" and existing:
        keys.delete_keys([existing[n % len(existing)]])
    elif kind == "move" and existing:
        keys.move_keys([existing[n % len(existing)]], dt)
    elif kind == "interp" and existing:
        keys.set_interpolation(existing, interp)
    elif kind == "look":
        looks = [look for group in window.looks.values() for look in group]
        looks[n % len(looks)].apply()
    elif kind == "batch":
        window.undo_batch("two edits", lambda: (window.set_property(pid, value),
                                                window.set_property("sky.haze", 1.5)))  # fmt: skip


def state(window) -> str:
    window.timeline.set_time(0.0)  # (animated values are read at one time)
    return json.dumps(window.scene.to_json(), sort_keys=True, default=str)


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(data=st.data())
def test_undo_everything_gives_the_scene_back(window, data):
    steps = data.draw(st.lists(edit(window.scene.store.registry), min_size=1, max_size=10))
    stack = window.undo_stack
    stack.clear()
    before = state(window)
    for step in steps:
        apply(window, step)
    after = state(window)
    while stack.canUndo():
        stack.undo()
    assert state(window) == before
    while stack.canRedo():
        stack.redo()
    assert state(window) == after
    while stack.canUndo():  # (the next example starts from the same scene)
        stack.undo()


def test_undoing_the_first_key_gives_the_old_value_back(window):
    """The case Hypothesis found: the first key of a static setting, undone, left the setting
    at the keyed value (the curve went, the store kept its last value)."""
    store, stack = window.scene.store, window.undo_stack
    stack.clear()
    window.set_property("fog.density", 0.6)
    window.timeline.set_time(1.0)
    window.keys.set_key("fog.density", 3.0)
    assert store["fog.density"] == 3.0
    stack.undo()
    assert store["fog.density"] == 0.6 and not window.scene.animation.is_animated("fog.density")
    stack.redo()
    assert store["fog.density"] == 3.0
    stack.undo()
    stack.undo()
