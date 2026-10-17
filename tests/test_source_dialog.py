from earthling.core.config import Project
from earthling.data.check import CheckReport
from earthling.session import Session
from helpers import DEMO


def test_source_dialog_reports(qtbot, monkeypatch):
    from earthling.app import source_dialog

    session = Session(Project.load(DEMO))
    seen = []

    def fake_check(source, lon, lat, out_dir=None):
        seen.append((source.id, lon, lat))
        report = CheckReport()
        report.info("all good")
        return report

    monkeypatch.setattr(source_dialog, "check_imagery", fake_check)
    dialog = source_dialog.SourceDialog(session, lambda title, start: (7.0, 46.0))
    qtbot.addWidget(dialog)
    dialog.source.setCurrentIndex(dialog.source.findData("esri_world_imagery"))
    assert "all good" in dialog.run()
    assert seen[0][0] == "esri_world_imagery" and 45.0 < seen[0][2] < 47.0  # a track point
    dialog._choose()
    dialog.run()
    assert seen[-1][1:] == (7.0, 46.0)
