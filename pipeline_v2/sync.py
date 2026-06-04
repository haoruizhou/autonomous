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
