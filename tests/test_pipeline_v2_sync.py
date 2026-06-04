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


def _win(first, last):
    return sync.GpxWindow(
        t_first=datetime.fromisoformat(first.replace("Z", "+00:00")),
        t_last=datetime.fromisoformat(last.replace("Z", "+00:00")),
        n_fixes=10, max_gap_s=9.0,
    )


def test_full_coverage_passes_no_warnings():
    start = datetime(2026, 4, 29, 12, 1, 0, tzinfo=timezone.utc)
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    report = sync.validate_alignment(start, "filename", 120.0, win)
    assert report.coverage == pytest.approx(1.0)
    assert report.warnings == []


def test_partial_coverage_warns_but_passes():
    # video starts 60s before GPS; 120s span → 60s inside window → 50% coverage
    start = datetime(2026, 4, 29, 11, 59, 0, tzinfo=timezone.utc)
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    report = sync.validate_alignment(start, "filename", 120.0, win, min_coverage=0.6, reject_coverage=0.1)
    assert report.coverage == pytest.approx(0.5)
    assert any("covers only" in w for w in report.warnings)


def test_below_floor_rejects():
    start = datetime(2026, 4, 29, 11, 50, 0, tzinfo=timezone.utc)  # mostly before window
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    with pytest.raises(sync.AlignmentError):
        sync.validate_alignment(start, "filename", 120.0, win, reject_coverage=0.1)


def test_disjoint_rejects():
    start = datetime(2026, 4, 29, 14, 0, 0, tzinfo=timezone.utc)
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    with pytest.raises(sync.AlignmentError):
        sync.validate_alignment(start, "filename", 120.0, win)


def test_metadata_source_adds_ambiguity_warning():
    start = datetime(2026, 4, 29, 12, 1, 0, tzinfo=timezone.utc)
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    report = sync.validate_alignment(start, "metadata", 120.0, win)
    assert any("AMBIGUOUS" in w for w in report.warnings)


def test_partial_below_floor_rejects():
    # video overlaps the window by only ~9% (6s of a 66s span) — below the 10% floor,
    # and distinct from the all-zero disjoint case
    start = datetime(2026, 4, 29, 11, 59, 0, tzinfo=timezone.utc)
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    assert sync.compute_coverage(start, 66.0, win) == pytest.approx(6 / 66)
    with pytest.raises(sync.AlignmentError):
        sync.validate_alignment(start, "filename", 66.0, win, reject_coverage=0.1)


def test_inverted_thresholds_raise_value_error():
    start = datetime(2026, 4, 29, 12, 1, 0, tzinfo=timezone.utc)
    win = _win("2026-04-29T12:00:00Z", "2026-04-29T12:10:00Z")
    with pytest.raises(ValueError):
        sync.validate_alignment(start, "filename", 120.0, win, min_coverage=0.3, reject_coverage=0.7)


def test_real_autocross_gpx_with_filename_start_passes():
    """Regression: the test_2fps_mac-style input must validate as pass."""
    from pipeline_v2.gps import parse_gpx
    win = sync.gpx_time_window(parse_gpx(Path("NHautocross/autocross.gpx")))
    # Synthetic start (12:40:00Z, embedded in the filename) sits inside the real
    # GPX window; an ~8 min clip is well within it.
    start, _ = sync.resolve_video_start(
        Path("NH1_20260429124000Z.mp4"), gpx_date=win.t_first.date()
    )
    report = sync.validate_alignment(start, "filename", 8 * 60.0, win)
    assert report.coverage > 0.6
