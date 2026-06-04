"""GPX parsing and video-frame timestamp alignment.

Extracted from the legacy ``track_pipeline.py`` orchestrator; these are the only
parts of that module still used by the active pipeline_v2 (see ``extract.py``).
"""

import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def parse_gpx(gpx_path: Path) -> list[dict]:
    """Parse GPX file → list of {lat, lon, alt, time (datetime UTC)}."""
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    points = []
    for pt in root.findall(".//gpx:trkpt", GPX_NS):
        t_el = pt.find("gpx:time", GPX_NS)
        if t_el is None:
            continue
        dt = datetime.fromisoformat(t_el.text.replace("Z", "+00:00"))
        ele_el = pt.find("gpx:ele", GPX_NS)
        alt = float(ele_el.text) if ele_el is not None else 0.0
        points.append({
            "lat":  float(pt.get("lat")),
            "lon":  float(pt.get("lon")),
            "alt":  alt,
            "time": dt,
        })
    return points


def video_start_time_from_filename(video_path: Path, date: date | None = None) -> Optional[datetime]:
    """Parse UTC start time from filename convention NH1_20260429125648Z.mp4.

    Supported patterns (last token before .ext):
      - HHMMSSZ         → e.g. NH1_125648Z.mp4  (date from GPX or today)
      - YYYYMMDDHHMMSSZ → e.g. NH1_20260429125648Z.mp4 (date embedded)

    Returns a timezone-aware UTC datetime, or None if no pattern matches.
    """
    stem = Path(video_path).stem  # e.g. "NH1_20260429125648Z"
    parts = stem.split("_")
    ts_part = parts[-1]           # e.g. "20260429125648Z"
    if not ts_part.endswith("Z"):
        return None

    if len(ts_part) == 15 and ts_part[:8].isdigit():
        # Embedded-date pattern: YYYYMMDDHHMMSSZ
        try:
            y = int(ts_part[0:4]); mo = int(ts_part[4:6]); d = int(ts_part[6:8])
            hh = int(ts_part[8:10]); mm = int(ts_part[10:12]); ss = int(ts_part[12:14])
            return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)
        except ValueError:
            return None
    elif len(ts_part) == 7 and ts_part[:6].isdigit():
        # Time-only pattern: HHMMSSZ — date must be provided
        if date is None:
            date = datetime.utcnow().date()
        try:
            hh = int(ts_part[0:2]); mm = int(ts_part[2:4]); ss = int(ts_part[4:6])
            return datetime(date.year, date.month, date.day, hh, mm, ss, tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def align_gpx_to_video(
    gpx_points: list[dict],
    video_path: Path,
    fps: float,
    t_start: Optional[datetime] = None,
) -> list[Optional[dict]]:
    """
    Returns one GPX point (or None) per video frame — interpolated by timestamp.

    Filename convention NH1_HHMMSSZ.mp4 supplies frame-0 UTC time; metadata
    creation_time is intentionally ignored.
    """
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if t_start is None:
        video_date = gpx_points[0]["time"].date()
        t_start = video_start_time_from_filename(video_path, date=video_date)
    if t_start is None:
        print(f"  [warn] Cannot parse UTC start time from filename {video_path.name}; skipping GPS alignment.")
        return [None] * n_frames
    # GPX times are UTC; video start is UTC per filename convention.
    # t_start is the UTC timestamp of frame 0.
    t0_ts = t_start.timestamp()

    aligned: list[Optional[dict]] = []
    for fi in range(n_frames):
        frame_t = t0_ts + fi / fps
        # Find surrounding GPX samples
        best = None
        for i in range(len(gpx_points) - 1):
            t_a = gpx_points[i]["time"].timestamp()
            t_b = gpx_points[i + 1]["time"].timestamp()
            if t_a <= frame_t <= t_b:
                alpha = (frame_t - t_a) / (t_b - t_a) if t_b > t_a else 0.0
                best = {
                    "lat": gpx_points[i]["lat"] + alpha * (gpx_points[i + 1]["lat"] - gpx_points[i]["lat"]),
                    "lon": gpx_points[i]["lon"] + alpha * (gpx_points[i + 1]["lon"] - gpx_points[i]["lon"]),
                    "alt": gpx_points[i]["alt"] + alpha * (gpx_points[i + 1]["alt"] - gpx_points[i]["alt"]),
                    "frame": fi,
                    "alpha": alpha,
                }
                break
        aligned.append(best)

    covered = sum(1 for x in aligned if x is not None)
    print(f"  GPX alignment: {covered}/{n_frames} frames have GPS ({video_path.name})")
    return aligned
