"""Stage A — video + GPX → OpenSfM project layout.

Writes:
    <project>/images/000000.jpg, 000001.jpg, ...
    <project>/exif_overrides.json   # per-image GPS prior + capture_time
    <project>/config.yaml           # OpenSfM config (sane phone-walk defaults)
    <project>/frame_index.json      # mapping image_name -> {video, src_frame, gps}

OpenSfM consumes `exif_overrides.json` natively and uses GPS in bundle adjustment,
so the trajectory output is GPS-anchored without a separate Procrustes step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import cv2

from pipeline_v2.gps import (
    align_gpx_to_video,
    parse_gpx,
)
from pipeline_v2 import sync


_PROCESSES = max(1, os.cpu_count() - 1)

DEFAULT_OPENSFM_CONFIG = f"""\
# Tuned for monocular phone walk-through with consumer GPS.
feature_type: SIFT
feature_min_frames: 4000
feature_process_size: 2048
sift_peak_threshold: 0.04

matcher_type: FLANN
# Wide GPS window (100m) captures revisited areas and improves connectivity
# on open parking lots where nearby frames may be far apart in GPS.
matching_gps_distance: 100
matching_gps_neighbors: 24
# 20 covers ±3-4 s at 6 fps — enough to bridge the gap between video clips when
# frames are globally sorted by capture_time (see write_project).
matching_time_neighbors: 20
matching_use_filters: yes
lowes_ratio: 0.85

# Looser triangulation/resection so weak-but-real tracks across turns survive.
robust_matching_threshold: 0.006
five_point_algo_threshold: 0.006
triangulation_threshold: 0.006
min_track_length: 2
resection_threshold: 0.008
resection_min_inliers: 8

# Force a single GPS-aligned frame for every reconstruction component.
align_method: auto
align_orientation_prior: vertical
bundle_use_gps: yes
bundle_compensate_gps_bias: yes
bundle_outlier_filtering_type: AUTO

# Dense reconstruction.
processes: {_PROCESSES}
depthmap_method: PATCH_MATCH_SAMPLE
depthmap_resolution: 640
depthmap_num_neighbors: 6
"""


def _gps_to_exif_override(gps: Optional[dict], dop_m: float = 5.0) -> Optional[dict]:
    """Convert an aligned GPS sample to an OpenSfM exif_override entry."""
    if gps is None:
        return None
    return {
        "gps": {
            "latitude": float(gps["lat"]),
            "longitude": float(gps["lon"]),
            "altitude": float(gps.get("alt", 0.0)),
            "dop": float(dop_m),  # phone GPS ~3-5m horizontal accuracy
        },
    }


def _collect_video_frames(
    video_path: Path,
    gpx_path: Path,
    sample_fps: float = 4.0,
    jpeg_quality: int = 92,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    video_start_override: Optional[str] = None,
    min_coverage: float = 0.6,
    reject_coverage: float = 0.1,
) -> tuple[list[dict], dict]:
    """Decode sampled frames from one video into memory.

    Returns:
        frames   — list of dicts:
                   {capture_time, frame_bgr, gps, video, src_frame, src_fps}
                   sorted by capture_time within this video.
        summary  — {video, total_frames, sampled, gps_covered, fps, stride}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(src_fps / sample_fps)))

    first_frame = int(start_sec * src_fps)
    last_frame = (n_total if duration_sec is None
                  else min(n_total, first_frame + int(duration_sec * src_fps)))
    if first_frame > 0 or duration_sec is not None:
        print(f"  clip: frames {first_frame}–{last_frame} "
              f"({(last_frame - first_frame) / src_fps / 60:.1f} min)")

    print(f"  {video_path.name}: {n_total} frames @ {src_fps:.2f} fps; "
          f"sample every {stride} → ~{(last_frame - first_frame) // stride} frames "
          f"({(last_frame - first_frame) / src_fps / 60:.1f} min)")

    # Resolve frame-0 UTC start, then validate against the GPS window.
    gpx_points = parse_gpx(gpx_path)
    win = sync.gpx_time_window(gpx_points)
    gpx_date = gpx_points[0]["time"].date()
    t_start, source = sync.resolve_video_start(
        video_path, gpx_date=gpx_date, override=video_start_override
    )
    span_s = n_total / src_fps
    report = sync.validate_alignment(
        t_start, source, span_s, win,
        video_name=video_path.name,
        min_coverage=min_coverage,
        reject_coverage=reject_coverage,
    )
    for w in report.warnings:
        print(f"  [warn] {w}")
    aligned = align_gpx_to_video(gpx_points, video_path, src_fps, t_start=t_start)
    creation_t = t_start

    frames: list[dict] = []
    gps_covered = 0
    for src_fi in range(first_frame, last_frame, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        gps = aligned[src_fi] if src_fi < len(aligned) else None
        capture_t = creation_t.timestamp() + src_fi / src_fps
        if gps is not None:
            gps_covered += 1
        frames.append({
            "capture_time": float(capture_t),
            "frame_bgr": frame_bgr,
            "gps": gps,
            "video": video_path.stem,
            "src_frame": src_fi,
            "src_fps": src_fps,
            "jpeg_quality": jpeg_quality,
        })

    cap.release()
    summary = {
        "video": video_path.stem,
        "total_frames": n_total,
        "sampled": len(frames),
        "gps_covered": gps_covered,
        "fps": src_fps,
        "stride": stride,
    }
    return frames, summary


def write_project(
    videos: Iterable[Path],
    gpx_path: Path,
    project_dir: Path,
    sample_fps: float = 4.0,
    config_yaml: Optional[str] = None,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    video_start_override: Optional[str] = None,
    min_coverage: float = 0.6,
    reject_coverage: float = 0.1,
) -> dict:
    """Build a single OpenSfM project from one or more videos sharing a GPX track.

    All frames across all videos are sorted globally by capture_time before being
    written with sequential zero-padded names (000000.jpg, 000001.jpg, …).  This
    ensures that frames from different clips are adjacent in the image sequence at
    the temporal seam, so OpenSfM's matching_time_neighbors walks right across the
    gap between clips and produces a single connected reconstruction.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # ── 1. Collect all frames from all videos ────────────────────────────────
    videos = list(videos)
    if video_start_override and len(videos) > 1:
        print("  [warn] --video-start ignored: applies only to a single video")
        video_start_override = None
    all_frames: list[dict] = []
    summaries: list[dict] = []
    for vp in videos:
        frames, summary = _collect_video_frames(
            vp, gpx_path,
            sample_fps=sample_fps,
            start_sec=start_sec,
            duration_sec=duration_sec,
            video_start_override=video_start_override,
            min_coverage=min_coverage,
            reject_coverage=reject_coverage,
        )
        all_frames.extend(frames)
        summaries.append(summary)

    # ── 2. Sort globally by capture_time ─────────────────────────────────────
    all_frames.sort(key=lambda f: f["capture_time"])

    # ── 3. Write images + build OpenSfM metadata ─────────────────────────────
    overrides_all: dict[str, dict] = {}
    index_all: dict[str, dict] = {}
    gps_total = 0

    for global_idx, f in enumerate(all_frames):
        name = f"{global_idx:06d}.jpg"
        cv2.imwrite(str(images_dir / name), f["frame_bgr"],
                    [cv2.IMWRITE_JPEG_QUALITY, int(f["jpeg_quality"])])

        entry: dict = {"capture_time": f["capture_time"]}
        gps_entry = _gps_to_exif_override(f["gps"])
        if gps_entry:
            entry.update(gps_entry)
            gps_total += 1
        overrides_all[name] = entry

        index_all[name] = {
            "video": f["video"],
            "src_frame": f["src_frame"],
            "src_fps": f["src_fps"],
            "gps": f["gps"],
        }

    (project_dir / "exif_overrides.json").write_text(
        json.dumps(overrides_all, indent=2, default=str)
    )
    (project_dir / "frame_index.json").write_text(
        json.dumps(index_all, indent=2, default=str)
    )
    (project_dir / "config.yaml").write_text(config_yaml or DEFAULT_OPENSFM_CONFIG)

    summary = {
        "project_dir": str(project_dir),
        "sample_fps": sample_fps,
        "videos": summaries,
        "n_images": len(all_frames),
        "n_with_gps": gps_total,
    }
    (project_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Project ready: {project_dir} "
          f"({summary['n_images']} images sorted by capture_time, "
          f"{gps_total} with GPS)")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract frames + GPS into an OpenSfM project")
    p.add_argument("--video", type=Path, action="append", required=True,
                   help="Video file (repeatable)")
    p.add_argument("--gpx", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True,
                   help="Output OpenSfM project directory")
    p.add_argument("--fps", type=float, default=6.0,
                   help="Sampling rate in frames/sec (default 6)")
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="Start offset in seconds into each video (default 0)")
    p.add_argument("--duration-sec", type=float, default=None,
                   help="Clip duration in seconds (default: full video)")
    args = p.parse_args()
    write_project(args.video, args.gpx, args.project, sample_fps=args.fps,
                  start_sec=args.start_sec, duration_sec=args.duration_sec)
