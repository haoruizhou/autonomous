"""Establish and validate the video↔GPS time relationship before SfM.

The GPX carries absolute UTC timestamps, so the only unknown is the video's
frame-0 UTC time. This module resolves that start time (filename → metadata →
override) and validates that the video span overlaps the GPS window, raising
AlignmentError when the two timelines do not agree.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline_v2.gps import video_start_time_from_filename


class AlignmentError(Exception):
    """Raised when the video and GPS timelines do not agree."""


@dataclass
class GpxWindow:
    t_first: datetime
    t_last: datetime
    n_fixes: int
    max_gap_s: float


def parse_iso_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def gpx_time_window(gpx_points: list[dict]) -> GpxWindow:
    if not gpx_points:
        raise AlignmentError("GPX has no timestamped trackpoints")
    times = sorted(p["time"] for p in gpx_points)
    gaps = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    return GpxWindow(
        t_first=times[0],
        t_last=times[-1],
        n_fixes=len(times),
        max_gap_s=max(gaps) if gaps else 0.0,
    )


def video_creation_time(video_path: Path) -> Optional[datetime]:
    """frame-0 UTC from MP4 metadata via ffprobe. AMBIGUOUS (start vs end)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        tags = json.loads(result.stdout or "{}").get("format", {}).get("tags", {})
        ct = tags.get("creation_time")
    except (ValueError, AttributeError):
        return None
    if not ct:
        return None
    try:
        return parse_iso_utc(ct)
    except ValueError:
        return None


def resolve_video_start(
    video_path: Path,
    *,
    gpx_date: date_cls,
    override: Optional[str] = None,
) -> tuple[datetime, str]:
    """Resolve the video's frame-0 UTC start time and its source.

    Priority: explicit ``override`` (ISO-8601) → filename convention
    (``video_start_time_from_filename``) → ffprobe metadata. Raises
    ``AlignmentError`` if none yields a time, or if ``override`` is not valid
    ISO-8601. ``gpx_date`` supplies the date for time-only filename patterns.
    Returns ``(start, source)`` with source in {"override","filename","metadata"}.
    """
    if override:
        try:
            return parse_iso_utc(override), "override"
        except ValueError as e:
            raise AlignmentError(
                f"--video-start value {override!r} is not valid ISO-8601: {e}"
            ) from e
    t = video_start_time_from_filename(video_path, date=gpx_date)
    if t is not None:
        return t, "filename"
    t = video_creation_time(video_path)
    if t is not None:
        return t, "metadata"
    raise AlignmentError(
        f"Cannot determine UTC start time for {video_path.name}. "
        f"Name it <stem>_HHMMSSZ.mp4 with the true UTC start, "
        f"or pass --video-start <ISO8601>."
    )


@dataclass
class AlignmentReport:
    video_start: datetime
    source: str
    video_span_s: float
    coverage: float
    warnings: list[str] = field(default_factory=list)


def compute_coverage(video_start: datetime, video_span_s: float, win: GpxWindow) -> float:
    if video_span_s <= 0:
        return 0.0
    t0 = video_start.timestamp()
    t1 = t0 + video_span_s
    overlap = min(t1, win.t_last.timestamp()) - max(t0, win.t_first.timestamp())
    return max(0.0, overlap) / video_span_s


def _disagree_message(name, video_start, source, span_s, win, coverage) -> str:
    t1 = datetime.fromtimestamp(video_start.timestamp() + span_s, tz=timezone.utc)
    return (
        f"video and GPS timelines do not agree for {name}.\n"
        f"  video start : {video_start.isoformat()}  (source: {source})\n"
        f"  video span  : {video_start.isoformat()} → {t1.isoformat()}  ({span_s/60:.1f} min)\n"
        f"  GPS window  : {win.t_first.isoformat()} → {win.t_last.isoformat()}  "
        f"({(win.t_last - win.t_first).total_seconds()/60:.1f} min)\n"
        f"  coverage    : {coverage*100:.0f}%\n"
        f"  fix         : name the video <stem>_HHMMSSZ.mp4 with its true UTC start, "
        f"or pass --video-start <ISO8601>."
    )


def validate_alignment(
    video_start: datetime,
    source: str,
    video_span_s: float,
    win: GpxWindow,
    *,
    video_name: str = "video",
    min_coverage: float = 0.6,
    reject_coverage: float = 0.1,
) -> AlignmentReport:
    if reject_coverage > min_coverage:
        raise ValueError(
            f"reject_coverage ({reject_coverage}) must be <= min_coverage ({min_coverage})"
        )
    coverage = compute_coverage(video_start, video_span_s, win)
    if coverage < reject_coverage:
        raise AlignmentError(_disagree_message(video_name, video_start, source, video_span_s, win, coverage))
    warnings: list[str] = []
    if source == "metadata":
        warnings.append(
            "video start from metadata creation_time is AMBIGUOUS (start vs end); "
            "prefer filename convention or --video-start"
        )
    if coverage < min_coverage:
        warnings.append(
            f"GPS covers only {coverage*100:.0f}% of the video; "
            f"frames outside the GPS window get no GPS prior"
        )
    return AlignmentReport(video_start, source, video_span_s, coverage, warnings)
