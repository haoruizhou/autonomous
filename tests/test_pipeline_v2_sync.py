from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from pipeline_v2 import sync


def _pt(iso):
    return {"time": datetime.fromisoformat(iso.replace("Z", "+00:00"))}


def test_parse_iso_utc_handles_z_and_naive():
    assert sync.parse_iso_utc("2026-04-29T12:56:48Z") == datetime(2026, 4, 29, 12, 56, 48, tzinfo=timezone.utc)
    assert sync.parse_iso_utc("2026-04-29T12:56:48").tzinfo == timezone.utc


def test_gpx_time_window_reports_span_count_and_max_gap():
    pts = [_pt("2026-04-29T12:00:00Z"), _pt("2026-04-29T12:00:09Z"), _pt("2026-04-29T12:02:15Z")]
    win = sync.gpx_time_window(pts)
    assert win.t_first == datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
    assert win.t_last == datetime(2026, 4, 29, 12, 2, 15, tzinfo=timezone.utc)
    assert win.n_fixes == 3
    assert win.max_gap_s == pytest.approx(126.0)


def test_gpx_time_window_empty_raises():
    with pytest.raises(sync.AlignmentError):
        sync.gpx_time_window([])


def test_resolve_video_start_prefers_override():
    dt, source = sync.resolve_video_start(
        Path("clip.mp4"), gpx_date=date(2026, 4, 29), override="2026-04-29T12:56:48Z"
    )
    assert source == "override"
    assert dt == datetime(2026, 4, 29, 12, 56, 48, tzinfo=timezone.utc)


def test_resolve_video_start_uses_filename_when_no_override():
    dt, source = sync.resolve_video_start(
        Path("NH1_20260429125648Z.mp4"), gpx_date=date(2026, 4, 29)
    )
    assert source == "filename"
    assert dt == datetime(2026, 4, 29, 12, 56, 48, tzinfo=timezone.utc)


def test_resolve_video_start_falls_back_to_metadata(monkeypatch):
    monkeypatch.setattr(
        sync, "video_creation_time", lambda p: datetime(2026, 4, 29, 13, 0, 0, tzinfo=timezone.utc)
    )
    dt, source = sync.resolve_video_start(Path("VID_1234.mp4"), gpx_date=date(2026, 4, 29))
    assert source == "metadata"
    assert dt == datetime(2026, 4, 29, 13, 0, 0, tzinfo=timezone.utc)


def test_resolve_video_start_raises_when_nothing_resolves(monkeypatch):
    monkeypatch.setattr(sync, "video_creation_time", lambda p: None)
    with pytest.raises(sync.AlignmentError):
        sync.resolve_video_start(Path("VID_1234.mp4"), gpx_date=date(2026, 4, 29))


def test_resolve_video_start_invalid_override_raises():
    with pytest.raises(sync.AlignmentError):
        sync.resolve_video_start(Path("clip.mp4"), gpx_date=date(2026, 4, 29), override="not-a-time")
