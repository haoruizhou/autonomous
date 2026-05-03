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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import cv2

# Reuse the v1 GPX/timestamp utilities — they're stable and well-tested.
from track_pipeline import (
    align_gpx_to_video,
    parse_gpx,
    video_creation_time,
)


DEFAULT_OPENSFM_CONFIG = """\
# Tuned for monocular phone walk-through with consumer GPS.
feature_type: SIFT
feature_min_frames: 2000
feature_process_size: 2048
sift_peak_threshold: 0.066

matcher_type: FLANN
matching_gps_distance: 100     # widened to bridge low-feature gaps
matching_gps_neighbors: 16
matching_time_neighbors: 8     # always match against +/- 8 temporal neighbors (loop closure)

# Bundle adjustment uses GPS as a soft prior (default behaviour).
# Phone GPS uncertainty: relax so SfM features dominate locally.
bundle_use_gps: yes
bundle_outlier_filtering_type: AUTO

# Dense reconstruction.
processes: 12
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


def extract_video(
    video_path: Path,
    gpx_path: Path,
    project_dir: Path,
    sample_fps: float = 4.0,
    image_prefix: Optional[str] = None,
    jpeg_quality: int = 92,
) -> dict:
    """Extract sampled frames + write OpenSfM project files for one video.

    Returns a summary dict:
        {video, total_frames, sampled, gps_covered, fps, project_dir}
    """
    project_dir = Path(project_dir)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(src_fps / sample_fps)))

    print(f"  {video_path.name}: {n_total} frames @ {src_fps:.2f} fps; "
          f"sample every {stride} → ~{n_total // stride} frames")

    gpx_points = parse_gpx(gpx_path)
    aligned = align_gpx_to_video(gpx_points, video_path, src_fps)
    t_end = video_creation_time(video_path) or datetime.now(timezone.utc)
    # creation_time is end-of-recording on iPhone/QuickTime — back out the start.
    creation_t = datetime.fromtimestamp(
        t_end.timestamp() - n_total / src_fps, tz=t_end.tzinfo
    )

    prefix = image_prefix or video_path.stem
    frame_index: dict[str, dict] = {}
    overrides: dict[str, dict] = {}

    written = 0
    gps_covered = 0
    for src_fi in range(0, n_total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        name = f"{prefix}_{src_fi:06d}.jpg"
        out_path = images_dir / name
        cv2.imwrite(str(out_path), frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])

        gps = aligned[src_fi] if src_fi < len(aligned) else None
        capture_t = creation_t.timestamp() + src_fi / src_fps

        entry: dict = {
            # OpenSfM ShotMeasurementDouble requires a numeric Unix timestamp.
            "capture_time": float(capture_t),
        }
        gps_entry = _gps_to_exif_override(gps)
        if gps_entry:
            entry.update(gps_entry)
            gps_covered += 1
        overrides[name] = entry

        frame_index[name] = {
            "video": video_path.stem,
            "src_frame": src_fi,
            "src_fps": src_fps,
            "gps": gps,
        }
        written += 1

    cap.release()

    return {
        "video": video_path.stem,
        "total_frames": n_total,
        "sampled": written,
        "gps_covered": gps_covered,
        "fps": src_fps,
        "stride": stride,
        "frame_index": frame_index,
        "exif_overrides": overrides,
    }


def write_project(
    videos: Iterable[Path],
    gpx_path: Path,
    project_dir: Path,
    sample_fps: float = 4.0,
    config_yaml: Optional[str] = None,
) -> dict:
    """Build a single OpenSfM project from one or more videos sharing a GPX track.

    Frame names are prefixed by video stem so multi-clip runs don't collide.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    overrides_all: dict[str, dict] = {}
    index_all: dict[str, dict] = {}

    for vp in videos:
        s = extract_video(vp, gpx_path, project_dir, sample_fps=sample_fps)
        summaries.append({k: v for k, v in s.items()
                          if k not in ("frame_index", "exif_overrides")})
        overrides_all.update(s["exif_overrides"])
        index_all.update(s["frame_index"])

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
        "n_images": sum(s["sampled"] for s in summaries),
        "n_with_gps": sum(s["gps_covered"] for s in summaries),
    }
    (project_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Project ready: {project_dir} "
          f"({summary['n_images']} images, {summary['n_with_gps']} with GPS)")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract frames + GPS into an OpenSfM project")
    p.add_argument("--video", type=Path, action="append", required=True,
                   help="Video file (repeatable)")
    p.add_argument("--gpx", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True,
                   help="Output OpenSfM project directory")
    p.add_argument("--fps", type=float, default=4.0,
                   help="Sampling rate in frames/sec (default 4)")
    args = p.parse_args()
    write_project(args.video, args.gpx, args.project, sample_fps=args.fps)
