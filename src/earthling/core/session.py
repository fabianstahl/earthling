"""Moved to :mod:`earthling.session` (the session combines core, data and render; this alias
keeps existing scene scripts working)."""

from earthling.session import DEFAULT_FRAME, Session, TrackGroup, load_group_tracks

__all__ = ["DEFAULT_FRAME", "Session", "TrackGroup", "load_group_tracks"]
