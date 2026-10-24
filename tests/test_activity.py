"""The status bar says what is still going on: tiles to load (with the time left), labels,
unreachable servers - and nothing when all is done."""

from types import SimpleNamespace

from earthling.app import activity


def fake_viewport(tiles: int = 0, labels: bool = False):
    selection = SimpleNamespace(request=list(range(tiles)))
    terrain = SimpleNamespace(nodes=object(), last_selection=selection)
    return SimpleNamespace(
        renderer=SimpleNamespace(terrain=terrain, labels=SimpleNamespace(loading=labels))
    )


def test_tiles_with_the_time_left_labels_and_offline(qtbot, monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(activity.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(activity, "offline_hosts", lambda: [])
    viewport = fake_viewport()
    label = activity.ActivityLabel(viewport)
    qtbot.addWidget(label)
    label.refresh()
    assert label.text() == "" and label.isHidden()
    for tiles in (100, 90, 80, 70):  # 10 tiles per second
        viewport.renderer.terrain.last_selection.request = list(range(tiles))
        label.refresh()
        clock["now"] += 1.0
    assert label.text() == "Loading the view: 70 tiles, about 7 s"
    viewport.renderer.labels.loading = True
    monkeypatch.setattr(activity, "offline_hosts", lambda: ["server.arcgisonline.com"])
    label.refresh()
    assert label.text().endswith(" · preparing the labels · offline: server.arcgisonline.com"
                                 " cannot be reached")  # fmt: skip
    assert "30 s" in label.toolTip()
    viewport.renderer.terrain.last_selection.request = []
    viewport.renderer.labels.loading = False
    monkeypatch.setattr(activity, "offline_hosts", lambda: [])
    label.refresh()
    assert label.text() == "" and label.isHidden()


def test_no_estimate_while_the_view_changes(qtbot, monkeypatch):
    monkeypatch.setattr(activity, "offline_hosts", lambda: [])
    viewport = fake_viewport()
    label = activity.ActivityLabel(viewport)
    qtbot.addWidget(label)
    for tiles in (50, 60, 70, 80):  # (more and more: the camera moves)
        viewport.renderer.terrain.last_selection.request = list(range(tiles))
        label.refresh()
    assert label.text() == "Loading the view: 80 tiles"
    assert activity.duration_text(30.0) == "30 s" and activity.duration_text(600.0) == "10 min"
