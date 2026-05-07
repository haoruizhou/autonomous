#!/usr/bin/env python3
"""
FSAE Autocross Track Reconstruction Pipeline
=============================================
Stages:
  1. GPX clip & align timestamps to video frames
  2. Depth Anything 3 — per-frame monocular depth
  3. YOLOv8 segmentation — remove people/cars, keep road/cones
  4. Filtered 3D point cloud per frame
  5. Track reconstruction — ground plane, cone centroids, centerline
  6. OBJ export  (road mesh + cone markers)
  7. XODR export (OpenDRIVE for simulators)

Usage:
  conda run -n autonomous python track_pipeline.py \
      --nh1 NHautocross/NH1.mp4 \
      --nh2 NHautocross/NH2.mp4 \
      --gpx NHautocross/autocross.gpx \
      --output results/track

Run individual stages with --stage 1..7  (default: all)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Enable MPS CPU fallback for ops not yet implemented on Apple Silicon
# (e.g. upsample_bicubic2d used by DA3's DINOv2 backbone)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config / constants
# ─────────────────────────────────────────────────────────────────────────────

# Camera intrinsics — Xiaomi 15 Ultra, ~1080p video
# FoV ≈ 70° horizontal; fx = w / (2 * tan(FoV/2))
# Adjust if you know the actual focal length from EXIF.
CAM_HFOV_DEG = 70.0

# YOLO classes to REMOVE (dynamic / non-track objects)
YOLO_REMOVE_CLASSES = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    16,  # dog
    17,  # horse
}

# Cone HSV range (orange FSAE cones)
CONE_HSV_LOW1  = np.array([0,   130,  80], dtype=np.uint8)
CONE_HSV_HIGH1 = np.array([20,  255, 255], dtype=np.uint8)
CONE_HSV_LOW2  = np.array([165, 130,  80], dtype=np.uint8)
CONE_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)
CONE_MIN_AREA  = 60      # px²
CONE_MAX_AREA  = 15000   # px²

DA3_DIR = Path(__file__).parent / "Depth-Anything-3"
DA3_SRC = DA3_DIR / "src"

GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — GPX parsing & video frame timestamp alignment
# ─────────────────────────────────────────────────────────────────────────────

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


def _ffprobe_binary() -> str:
    """Return path to ffprobe, preferring system install then imageio-ffmpeg bundle."""
    import shutil
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffprobe:
        return sys_ffprobe
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        # imageio-ffmpeg ships ffmpeg; ffprobe lives alongside it
        import os
        ffprobe_bin = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe")
        if not os.path.exists(ffprobe_bin):
            # Some builds ship a single combined binary named ffmpeg
            ffprobe_bin = ffmpeg_bin.replace("ffmpeg", "ffprobe")
        if os.path.exists(ffprobe_bin):
            return ffprobe_bin
        # Fall back to the ffmpeg binary itself with -of json (ffprobe-compatible flags)
        return ffmpeg_bin
    except ImportError:
        pass
    return "ffprobe"


def video_creation_time(video_path: Path) -> Optional[datetime]:
    """Extract creation_time from MP4 metadata via ffprobe (or imageio-ffmpeg fallback)."""
    import subprocess, json as _json
    try:
        result = subprocess.run(
            [_ffprobe_binary(), "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        tags = _json.loads(result.stdout).get("format", {}).get("tags", {})
        ct = tags.get("creation_time")
        if ct:
            return datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except Exception as e:
        print(f"  [warn] ffprobe failed for {video_path.name}: {e}")
    return None


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def latlon_to_local_xy(points: list[dict]) -> np.ndarray:
    """Convert lat/lon list to local ENU metres (origin = first point). Returns Nx3."""
    if not points:
        return np.zeros((0, 3))
    lat0, lon0, alt0 = points[0]["lat"], points[0]["lon"], points[0]["alt"]
    R = 6_371_000.0
    out = []
    for p in points:
        x = math.radians(p["lon"] - lon0) * math.cos(math.radians(lat0)) * R
        y = math.radians(p["lat"] - lat0) * R
        z = p["alt"] - alt0
        out.append([x, y, z])
    return np.array(out, dtype=np.float64)


def align_gpx_to_video(
    gpx_points: list[dict],
    video_path: Path,
    fps: float,
) -> list[Optional[dict]]:
    """
    Returns one GPX point (or None) per video frame — interpolated by timestamp.

    NH1 creation_time ≈ 2026-04-29T13:00:00Z
    NH2 creation_time ≈ 2026-04-29T13:07:36Z
    GPX ends at                13:07:47Z  → NH2 barely covered
    """
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    t_end = video_creation_time(video_path)
    if t_end is None:
        print(f"  [warn] No creation_time for {video_path.name}; skipping GPS alignment.")
        return [None] * n_frames
    # iPhone/QuickTime convention: container creation_time = end of recording.
    duration_s = n_frames / fps if fps > 0 else 0.0
    t0_ts = t_end.timestamp() - duration_s

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


def run_stage1(args, out_dir: Path) -> dict:
    """Parse GPX, align to both videos, save JSON."""
    print("\n[Stage 1] GPX parsing & frame alignment")
    gpx_points = parse_gpx(args.gpx)
    print(f"  GPX points: {len(gpx_points)}  "
          f"({gpx_points[0]['time'].strftime('%H:%M:%S')} → "
          f"{gpx_points[-1]['time'].strftime('%H:%M:%S')} UTC)")

    results = {}
    for video_path in [args.nh1, args.nh2]:
        if video_path is None:
            continue
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        aligned = align_gpx_to_video(gpx_points, video_path, fps)
        key = video_path.stem  # "NH1" or "NH2"
        results[key] = {
            "fps":     fps,
            "aligned": [x for x in aligned if x is not None],
        }

    # Save all raw GPX points as local XY (for track map)
    local_xy = latlon_to_local_xy(gpx_points).tolist()
    gpx_out = out_dir / "gpx_local_xy.json"
    gpx_out.write_text(json.dumps({
        "origin_lat": gpx_points[0]["lat"],
        "origin_lon": gpx_points[0]["lon"],
        "points":     local_xy,
    }, indent=2))
    print(f"  Saved GPX local XY → {gpx_out}")

    for key, val in results.items():
        aligned_out = out_dir / f"{key}_gps_aligned.json"
        aligned_out.write_text(json.dumps(val["aligned"], indent=2))
        print(f"  Saved aligned GPS → {aligned_out}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Depth Anything 3 per-frame depth extraction
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_da3():
    """
    Add DA3 src to sys.path and import DepthAnything3.
    Stubs out depth_anything_3.utils.export before the import so we don't
    need moviepy / open3d / pycolmap just to run inference.
    """
    import types

    if str(DA3_SRC) not in sys.path:
        sys.path.insert(0, str(DA3_SRC))

    # Pre-register stub modules that api.py imports but we don't need for inference.
    # NOTE: parallel_utils is NOT stubbed — its real impl is needed; imageio stub
    #       satisfies its only top-level dep.
    _stub_names = [
        "depth_anything_3.utils.export",
        "depth_anything_3.utils.export.gs",
        "depth_anything_3.utils.export.mesh",
        "depth_anything_3.utils.export.npz",
        "depth_anything_3.utils.pose_align",
        "evo", "evo.core", "evo.core.trajectory",
        "imageio", "moviepy", "moviepy.editor",
        "trimesh", "open3d",
    ]
    for name in _stub_names:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.export = lambda *a, **kw: None
            stub.export_to_gs_ply = lambda *a, **kw: None
            stub.export_to_gs_video = lambda *a, **kw: None
            stub.align_poses_umeyama = lambda *a, **kw: (None, None, 1.0, None)
            stub.PosePath3D = object
            # imageio stub needs a few attributes accessed during import
            if name == "imageio":
                stub.imread  = lambda *a, **kw: None
                stub.mimread = lambda *a, **kw: []
                stub.plugins = types.ModuleType("imageio.plugins")
                sys.modules["imageio.plugins"] = stub.plugins
            sys.modules[name] = stub

    try:
        from depth_anything_3.api import DepthAnything3
        return DepthAnything3
    except ImportError:
        import subprocess
        print("  Installing depth_anything_3 package (no-deps)…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-q", "-e", str(DA3_DIR)],
            check=True,
        )
        from depth_anything_3.api import DepthAnything3
        return DepthAnything3


def run_stage2(args, out_dir: Path, sample_fps: int = 5) -> Path:
    """
    Extract frames from NH1+NH2, run DA3 depth inference.
    Saves per-frame depth .npy files to out_dir/depth_npy/.
    Returns path to that directory.
    """
    print("\n[Stage 2] Depth Anything 3 — monocular depth extraction")
    import torch

    if not DA3_DIR.exists():
        import subprocess
        print(f"  Cloning Depth-Anything-3 → {DA3_DIR}")
        subprocess.run(
            ["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(DA3_DIR)],
            check=True,
        )

    DepthAnything3 = _ensure_da3()

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"  Device: {device}")

    model_id = getattr(args, "model", "depth-anything/DA3MONO-LARGE")
    print(f"  Loading model: {model_id}")
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()

    depth_npy_dir = out_dir / "depth_npy"
    depth_vis_dir = out_dir / "depth_vis"
    frame_meta_dir = out_dir / "frame_meta"
    for d in [depth_npy_dir, depth_vis_dir, frame_meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("inferno")

    videos = [(v, v.stem) for v in [args.nh1, args.nh2] if v is not None]
    global_frame_idx = 0

    for video_path, stem in videos:
        cap = cv2.VideoCapture(str(video_path))
        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval  = max(1, int(round(src_fps / sample_fps)))

        print(f"  {stem}: {total_frm} frames @ {src_fps:.1f}fps → sample every {interval} (≈{sample_fps}fps)")

        tmp_frames = []
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % interval == 0:
                stem_name = f"{stem}_f{fi:06d}_g{global_frame_idx:06d}"
                npy_exists = (depth_npy_dir / f"{stem_name}.npy").exists()
                if not npy_exists:
                    tmp_path = out_dir / f"_tmp_{stem}_{fi:06d}.png"
                    cv2.imwrite(str(tmp_path), frame)
                    tmp_frames.append((fi, global_frame_idx, tmp_path))
                global_frame_idx += 1
            fi += 1
        cap.release()

        if not tmp_frames:
            print(f"  {stem}: all frames already computed, skipping.")
            continue

        # Batch inference
        chunk = 2
        for start in tqdm(range(0, len(tmp_frames), chunk),
                          desc=f"  DA3 {stem}", unit="batch"):
            batch = tmp_frames[start: start + chunk]
            paths = [str(b[2]) for b in batch]
            with __import__("torch").no_grad():
                pred = model.inference(paths)
            depths = pred.depth  # [N, H, W]

            for (src_fi, gfi, tmp_path), depth in zip(batch, depths):
                depth_arr = np.asarray(depth, dtype=np.float32)
                stem_name = f"{stem}_f{src_fi:06d}_g{gfi:06d}"

                # Raw depth
                np.save(str(depth_npy_dir / f"{stem_name}.npy"), depth_arr)

                # Colourmap vis
                lo, hi = np.percentile(depth_arr, 2), np.percentile(depth_arr, 98)
                norm = np.clip((depth_arr - lo) / max(hi - lo, 1e-6), 0, 1)
                colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
                __import__("PIL").Image.fromarray(colored).save(
                    str(depth_vis_dir / f"{stem_name}.png")
                )

                # Frame metadata (source video, frame index)
                (frame_meta_dir / f"{stem_name}.json").write_text(json.dumps({
                    "video": stem, "src_frame": src_fi, "global_frame": gfi,
                }))

                # Cleanup temp file
                tmp_path.unlink(missing_ok=True)

    n_saved = len(list(depth_npy_dir.glob("*.npy")))
    print(f"  Saved {n_saved} depth maps → {depth_npy_dir}")
    return depth_npy_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — YOLOv8 segmentation: remove dynamic objects, isolate road + cones
# ─────────────────────────────────────────────────────────────────────────────

def _select_debug_sample_frames(frames: list[int], sample_count: int, seed: int = 0) -> list[int]:
    unique_frames = sorted(set(int(frame) for frame in frames))
    if sample_count <= 0 or sample_count >= len(unique_frames):
        return unique_frames
    rng = random.Random(seed)
    return sorted(rng.sample(unique_frames, sample_count))


def _cone_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of orange cone pixels (HSV-based)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, CONE_HSV_LOW1, CONE_HSV_HIGH1),
        cv2.inRange(hsv, CONE_HSV_LOW2, CONE_HSV_HIGH2),
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # Filter by connected-component area
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if CONE_MIN_AREA <= a <= CONE_MAX_AREA:
            clean[labels == i] = 255
    return clean


def _cone_mask_hsv_debug(frame_bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.bitwise_or(
        cv2.inRange(hsv, CONE_HSV_LOW1, CONE_HSV_HIGH1),
        cv2.inRange(hsv, CONE_HSV_LOW2, CONE_HSV_HIGH2),
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k, iterations=2)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, k, iterations=2)

    n_raw, _, raw_stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_mask, connectivity=8)
    clean_mask = np.zeros_like(morph_mask)
    components = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if CONE_MIN_AREA <= area <= CONE_MAX_AREA:
            clean_mask[labels == i] = 255
            components.append({
                "cx": int(centroids[i][0]),
                "cy": int(centroids[i][1]),
                "area": area,
                "bbox": [
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]),
                ],
            })

    return {
        "raw_mask": raw_mask,
        "morph_mask": morph_mask,
        "clean_mask": clean_mask,
        "raw_count": max(0, n_raw - 1),
        "raw_areas": [int(raw_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_raw)],
        "clean_count": len(components),
        "components": components,
    }


def _cone_boxes_to_centers(boxes: list[dict], min_conf: float = 0.25) -> list[dict]:
    centers = []
    for box in boxes:
        conf = float(box.get("conf", 0.0))
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        centers.append({
            "cx": int(round((x1 + x2) / 2.0)),
            "cy": int(round(y2)),
            "area": int(round(w * h)),
            "conf": round(conf, 3),
            "bbox": [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))],
            "source": "model",
            "label": str(box.get("label", "traffic cone")),
        })
    return centers


def _yolo_boxes_to_dicts(yolo_res, names: Optional[dict] = None) -> list[dict]:
    if yolo_res.boxes is None:
        return []
    xyxy = yolo_res.boxes.xyxy.cpu().numpy()
    confs = yolo_res.boxes.conf.cpu().numpy()
    cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)
    boxes = []
    for coords, conf, cls_id in zip(xyxy, confs, cls_ids):
        boxes.append({
            "xyxy": [float(v) for v in coords],
            "conf": float(conf),
            "cls": int(cls_id),
            "label": str(names.get(int(cls_id), "traffic cone")) if names else "traffic cone",
        })
    return boxes


def _gps_offset_for_frame(gps_list: list[dict], src_frame: int) -> np.ndarray:
    if not gps_list:
        return np.zeros(3, dtype=np.float32)

    for g in gps_list:
        if g.get("frame") == src_frame:
            lat0 = gps_list[0]["lat"]
            lon0 = gps_list[0]["lon"]
            ox = math.radians(g["lon"] - lon0) * math.cos(math.radians(lat0)) * 6_371_000.0
            oz = math.radians(g["lat"] - lat0) * 6_371_000.0
            return np.array([ox, 0.0, oz], dtype=np.float32)

    return np.zeros(3, dtype=np.float32)


def _cone_projection_records(
    frame_name: str,
    video_stem: str,
    src_frame: int,
    cone_centers_px: list[dict],
    depth_arr: np.ndarray,
    gps_offset: np.ndarray,
) -> list[dict]:
    h, w = depth_arr.shape
    fov_rad = math.radians(CAM_HFOV_DEG)
    fx = fy = w / (2 * math.tan(fov_rad / 2))
    cx, cy = w / 2.0, h / 2.0
    records = []

    for cc in cone_centers_px:
        px, py = int(cc["cx"]), int(cc["cy"])
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        d_rel = float(depth_arr[py, px])
        if d_rel <= 0 or not math.isfinite(d_rel):
            continue
        d = 1.0 / (d_rel + 1e-6)
        camera_xyz = np.array([
            (px - cx) / fx * d,
            (py - cy) / fy * d,
            d,
        ], dtype=np.float32)
        world_xyz = camera_xyz + gps_offset
        records.append({
            "frame": frame_name,
            "video": video_stem,
            "src_frame": int(src_frame),
            "px": px,
            "py": py,
            "area": int(cc.get("area", 0)),
            "depth_rel": round(d_rel, 6),
            "camera_xyz": [round(float(v), 6) for v in camera_xyz],
            "gps_offset": [round(float(v), 6) for v in gps_offset],
            "world_xyz": [round(float(v), 6) for v in world_xyz],
        })

    return records


def _write_debug_html_report(debug_dir: Path, title: str = "Track Pipeline Debug Report") -> Path:
    frames_dir = debug_dir / "frames"
    world_dir = debug_dir / "world"
    frame_panels = sorted(frames_dir.glob("*_debug.png")) if frames_dir.exists() else []
    world_panels = sorted(world_dir.glob("*_world.png")) if world_dir.exists() else []
    world_by_stem = {p.name.replace("_world.png", ""): p for p in world_panels}
    summary = world_dir / "cone_world_summary.png"
    records_path = debug_dir / "cone_projection_records.json"

    rows = []
    for panel in frame_panels:
        stem = panel.name.replace("_debug.png", "")
        world_panel = world_by_stem.get(stem)
        world_cell = f'<a href="world/{world_panel.name}">world plot</a>' if world_panel else "missing"
        rows.append(
            f"<tr><td>{stem}</td><td><a href=\"frames/{panel.name}\">frame panel</a></td>"
            f"<td>{world_cell}</td></tr>"
        )

    summary_html = f'<p><a href="world/{summary.name}">Aggregate world-space cone summary</a></p>' if summary.exists() else ""
    records_html = f'<p><a href="{records_path.name}">Cone projection records JSON</a></p>' if records_path.exists() else ""
    body_rows = "\n".join(rows) if rows else '<tr><td colspan="3">No frame panels generated.</td></tr>'
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; line-height: 1.4; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {summary_html}
  {records_html}
  <table>
    <thead><tr><th>Frame</th><th>Image-space debug</th><th>World-space debug</th></tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</body>
</html>
"""
    html_path = debug_dir / "index.html"
    html_path.write_text(html)
    return html_path


# ── SegFormer (Cityscapes) 3-class semantic segmenter ───────────────────────
# Cityscapes class ids: road=0, sidewalk=1, vegetation=8, terrain=9.
_CS_ROAD_IDS = (0, 1)
_CS_GRASS_IDS = (8, 9)
_SEGFORMER_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def _load_segformer_cityscapes():
    """Lazy-load SegFormer-B0 fine-tuned on Cityscapes. Returns (model, processor, device) or None on failure."""
    try:
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    except Exception as e:
        print(f"  [warn] transformers not available ({e}); SegFormer disabled")
        return None
    try:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  Loading SegFormer ({_SEGFORMER_NAME}) on {device}...")
        processor = SegformerImageProcessor.from_pretrained(_SEGFORMER_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(_SEGFORMER_NAME).to(device).eval()
        return model, processor, device
    except Exception as e:
        print(f"  [warn] SegFormer load failed: {e}")
        return None


def _label_map_segformer(seg, frame_bgr: np.ndarray) -> np.ndarray:
    """Run SegFormer once and return a uint8 HxW label map: 0=other, 1=road, 2=grass."""
    import torch
    model, processor, device = seg
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, 19, h', w']
    logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    label = np.zeros((h, w), dtype=np.uint8)
    for cid in _CS_ROAD_IDS:
        label[pred == cid] = 1
    for cid in _CS_GRASS_IDS:
        label[pred == cid] = 2
    return label


def _yolop_drivable_prob(seg_model, frame_bgr: np.ndarray) -> np.ndarray:
    """Run YOLOP at its trained 640x384 input and return per-pixel drivable prob (float32, HxW)."""
    import torch
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    in_w, in_h = 640, 384
    img_in = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    t = seg_model.transform(img_in).to(seg_model.device).unsqueeze(0)
    with torch.no_grad():
        _, da_seg_out, _ = seg_model.model(t)  # [1, 2, h, w]
    prob = torch.softmax(da_seg_out, dim=1)[:, 1:2]  # foreground (drivable) channel
    prob = torch.nn.functional.interpolate(prob, size=(h, w), mode="bilinear", align_corners=True)
    return prob.squeeze().cpu().numpy().astype(np.float32)


def _road_mask_yolop(
    seg_model,
    frame_bgr: np.ndarray,
    cone_mask: np.ndarray,
    threshold: float = 0.35,
    union_heuristic: bool = True,
    heuristic_gate: float = 0.10,
) -> np.ndarray:
    """Drivable-area mask via YOLOP, with a heuristic-augmented YOLOP-gated union.

    YOLOP (BDD100K) under-segments wide paved lots/paddocks. We add heuristic
    pixels where YOLOP probability ≥ ``heuristic_gate`` (low confidence is OK,
    but pure-grass pixels — prob ~ 0 — stay excluded). This keeps grass out
    while filling in YOLOP's confidence valleys on open paddocks.
    """
    prob = _yolop_drivable_prob(seg_model, frame_bgr)
    road = (prob >= threshold).astype(np.uint8) * 255

    if union_heuristic:
        heur = _road_mask_heuristic(frame_bgr, cone_mask)
        gate = (prob >= heuristic_gate).astype(np.uint8) * 255
        heur_gated = cv2.bitwise_and(heur, gate)
        road = cv2.bitwise_or(road, heur_gated)

    road = cv2.bitwise_and(road, cv2.bitwise_not(cone_mask))

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k9, iterations=3)
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN,  k9, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        road = np.where(labels == largest, 255, 0).astype(np.uint8)
    return road


def _road_mask_heuristic(frame_bgr: np.ndarray, cone_mask: np.ndarray) -> np.ndarray:
    """Estimate drivable-surface mask (dark asphalt below horizon)."""
    h, w = frame_bgr.shape[:2]
    horizon_y = int(h * 0.42)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]
    a_ch = lab[:, :, 1].astype(np.int16) - 128
    b_ch = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a_ch**2 + b_ch**2).astype(np.float32)

    road_zone = np.zeros((h, w), dtype=np.uint8)
    road_zone[horizon_y:, :] = 255

    dark_gray = ((L >= 18) & (L <= 160) & (chroma < 32)).astype(np.uint8) * 255
    road_raw  = cv2.bitwise_and(dark_gray, road_zone)
    road_raw  = cv2.bitwise_and(road_raw, cv2.bitwise_not(cone_mask))

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road_raw = cv2.morphologyEx(road_raw, cv2.MORPH_CLOSE, k9, iterations=4)
    road_raw = cv2.morphologyEx(road_raw, cv2.MORPH_OPEN,  k9, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(road_raw, connectivity=8)
    road_mask = np.zeros_like(road_raw)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        road_mask[labels == largest] = 255
    return road_mask


def _mask_to_bgr(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def _annotate_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _resize_panel(image: np.ndarray, size: tuple[int, int] = (480, 270)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _save_frame_debug_panel(
    path: Path,
    frame_bgr: np.ndarray,
    depth_arr: np.ndarray,
    cone_debug: dict,
    remove_mask: np.ndarray,
    road_mask: np.ndarray,
    keep_mask: np.ndarray,
    cone_centers_px: list[dict],
    grass_mask: Optional[np.ndarray] = None,
) -> None:
    original = frame_bgr.copy()
    for cc in cone_centers_px:
        cv2.drawMarker(original, (cc["cx"], cc["cy"]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
        if "bbox" in cc:
            x, y, bw, bh = cc["bbox"]
            cv2.rectangle(original, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"{cc.get('source', 'cone')} {cc.get('conf', 0):.2f}"
        cv2.putText(original, label, (cc["cx"] + 6, cc["cy"]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    lo, hi = np.percentile(depth_arr, 2), np.percentile(depth_arr, 98)
    depth_norm = (np.clip((depth_arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    raw_vis = cv2.cvtColor(cone_debug["raw_mask"], cv2.COLOR_GRAY2BGR)
    clean_vis = cv2.cvtColor(cone_debug["clean_mask"], cv2.COLOR_GRAY2BGR)
    road_layer   = _mask_to_bgr(road_mask, (80, 80, 255))
    remove_layer = _mask_to_bgr(remove_mask, (255, 0, 0))
    if grass_mask is not None:
        grass_layer = _mask_to_bgr(grass_mask, (60, 200, 80))
        road_remove_vis = cv2.addWeighted(cv2.addWeighted(road_layer, 1.0, grass_layer, 1.0, 0), 1.0, remove_layer, 1.0, 0)
        seg_title = "Road (red) / grass (green) / removed (blue)"
    else:
        road_remove_vis = cv2.addWeighted(road_layer, 1.0, remove_layer, 1.0, 0)
        seg_title = "Road mask + removed objects"
    keep_cone_vis = cv2.addWeighted(_mask_to_bgr(keep_mask, (0, 180, 0)), 1.0, _mask_to_bgr(cone_debug["clean_mask"], (0, 140, 255)), 1.0, 0)

    panels = [
        _annotate_panel(_resize_panel(original), "Original + cone centroids"),
        _annotate_panel(_resize_panel(depth_vis), "DA3 depth visualization"),
        _annotate_panel(_resize_panel(raw_vis), f"Raw HSV cone mask ({cone_debug['raw_count']})"),
        _annotate_panel(_resize_panel(clean_vis), f"Clean cone components ({cone_debug['clean_count']})"),
        _annotate_panel(_resize_panel(road_remove_vis), seg_title),
        _annotate_panel(_resize_panel(keep_cone_vis), "Final keep mask + cones"),
    ]
    canvas = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _world_to_canvas(points_xy: np.ndarray, width: int = 900, height: int = 900) -> tuple[np.ndarray, float, float, float]:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(points_xy) == 0:
        return canvas, 1.0, width / 2, height / 2
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 80) / span[0], (height - 80) / span[1])
    ox = 40 - min_xy[0] * scale
    oy = height - 40 + min_xy[1] * scale
    return canvas, scale, ox, oy


def _draw_world_points(canvas: np.ndarray, points_xy: np.ndarray, scale: float, ox: float, oy: float, color: tuple[int, int, int], radius: int) -> None:
    for x, y in points_xy:
        px = int(round(x * scale + ox))
        py = int(round(oy - y * scale))
        cv2.circle(canvas, (px, py), radius, color, -1)


def _save_frame_world_plot(
    path: Path,
    records: list[dict],
    centerline_xy: np.ndarray,
    gps_offset: np.ndarray,
) -> None:
    cone_xy = np.array([[r["world_xyz"][0], r["world_xyz"][2]] for r in records], dtype=np.float32) if records else np.zeros((0, 2), dtype=np.float32)
    gps_xy = np.array([[gps_offset[0], gps_offset[2]]], dtype=np.float32)
    all_points = np.vstack([p for p in [centerline_xy, cone_xy, gps_xy] if len(p) > 0])
    canvas, scale, ox, oy = _world_to_canvas(all_points)
    if len(centerline_xy) > 1:
        pts = np.array([[int(round(x * scale + ox)), int(round(oy - y * scale))] for x, y in centerline_xy], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (200, 200, 200), 2)
    _draw_world_points(canvas, gps_xy, scale, ox, oy, (255, 0, 0), 7)
    _draw_world_points(canvas, cone_xy, scale, ox, oy, (0, 85, 255), 6)
    cv2.putText(canvas, path.name.replace("_world.png", ""), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blue=phone GPS, orange=projected cones, gray=GPS centerline", (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _load_centerline_xy(out_dir: Path) -> np.ndarray:
    gpx_xy_path = out_dir / "gpx_local_xy.json"
    if not gpx_xy_path.exists():
        return np.zeros((0, 2), dtype=np.float32)
    gpx_data = json.loads(gpx_xy_path.read_text())
    pts = np.array(gpx_data.get("points", []), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return pts[:, :2]


def run_stage3(args, out_dir: Path, depth_npy_dir: Path) -> Path:
    """
    For each depth frame:
      - Reload original video frame (by metadata)
      - Run YOLOv8n-seg → build 'remove mask' for people/cars
      - Run HSV cone detector
      - Run heuristic road detector
      - Save keep_mask.npy (255 = valid for 3D reconstruction)
      - Save cone_centroids.json per frame
    Returns path to masks directory.
    """
    print("\n[Stage 3] YOLOv8 segmentation — filtering dynamic objects")
    from ultralytics import YOLO

    yolo = YOLO("yolov8n-seg.pt")   # auto-downloads on first run (~6 MB)
    print("  YOLOv8n-seg loaded")

    cone_model = YOLO(getattr(args, "cone_model", "yolov8s-world.pt"))
    if hasattr(cone_model, "set_classes"):
        cone_model.set_classes(["traffic cone", "cone"])
    cone_conf = float(getattr(args, "cone_conf", 0.25))
    use_hsv_fallback = bool(getattr(args, "cone_hsv_fallback", False))
    print(f"  Cone detector loaded: {getattr(args, 'cone_model', 'yolov8s-world.pt')} (conf ≥ {cone_conf:.2f})")

    seg_model_choice = str(getattr(args, "seg_model", "segformer-cityscapes")).lower()
    road_method = str(getattr(args, "road_method", "yolop")).lower()
    segformer = None
    yolop_seg = None
    if seg_model_choice == "segformer-cityscapes":
        segformer = _load_segformer_cityscapes()
        if segformer is None:
            print("  [warn] SegFormer unavailable; falling back to YOLOP+heuristic")
            seg_model_choice = "yolop-heuristic"
    if seg_model_choice != "segformer-cityscapes" and road_method == "yolop":
        try:
            from monoslam import MultiClassSegmentation
            yolop_seg = MultiClassSegmentation()
            print("  YOLOP drivable-area segmenter loaded")
        except Exception as e:
            print(f"  [warn] YOLOP load failed ({e}); falling back to heuristic road mask")
            yolop_seg = None
            road_method = "heuristic"
    elif seg_model_choice != "segformer-cityscapes":
        print("  Road mask: heuristic (LAB chroma + horizon)")

    mask_dir    = out_dir / "keep_masks"
    cone_dir    = out_dir / "cone_data"
    seg_vis_dir = out_dir / "seg_vis"
    label_dir   = out_dir / "label_maps"
    mask_dir.mkdir(exist_ok=True)
    cone_dir.mkdir(exist_ok=True)
    seg_vis_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    debug_enabled = bool(getattr(args, "debug_vis", False))
    debug_every = max(1, int(getattr(args, "debug_every", 10)))
    debug_dir = out_dir / "debug_vis"
    debug_frames_dir = debug_dir / "frames"
    debug_world_dir = debug_dir / "world"
    if debug_enabled:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)
        debug_world_dir.mkdir(parents=True, exist_ok=True)
    centerline_xy = _load_centerline_xy(out_dir) if debug_enabled else np.zeros((0, 2), dtype=np.float32)
    gps_aligned = {}
    if debug_enabled:
        for jpath in out_dir.glob("*_gps_aligned.json"):
            gps_aligned[jpath.name.replace("_gps_aligned.json", "")] = json.loads(jpath.read_text())
    projection_records: list[dict] = []

    frame_meta_dir = out_dir / "frame_meta"
    depth_files    = sorted(depth_npy_dir.glob("*.npy"))
    debug_sample_count = int(getattr(args, "debug_sample_frames", 0))
    debug_sample_set: Optional[set[tuple[str, int]]] = None
    if debug_sample_count > 0:
        sample_candidates = []
        for depth_file in depth_files:
            meta_path = frame_meta_dir / f"{depth_file.stem}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                sample_candidates.append((meta["video"], int(meta["src_frame"])))
        sampled_indices = _select_debug_sample_frames(list(range(len(sample_candidates))), debug_sample_count, int(getattr(args, "debug_sample_seed", 0)))
        debug_sample_set = {sample_candidates[i] for i in sampled_indices}
        print(f"  Debug sample mode: {len(debug_sample_set)}/{len(sample_candidates)} frames")

    # Pre-open video captures (reuse across frames)
    video_caps: dict[str, cv2.VideoCapture] = {}
    for vp in [args.nh1, args.nh2]:
        if vp is not None:
            video_caps[vp.stem] = cv2.VideoCapture(str(vp))

    cone_summary: list[dict] = []

    for depth_file in tqdm(depth_files, desc="  Seg+filter", unit="frame"):
        stem_name = depth_file.stem
        meta_path = frame_meta_dir / f"{stem_name}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        video_stem = meta["video"]
        src_frame  = meta["src_frame"]
        if debug_sample_set is not None and (video_stem, int(src_frame)) not in debug_sample_set:
            continue

        cap = video_caps.get(video_stem)
        if cap is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        h, w = frame_bgr.shape[:2]

        # ── YOLO remove mask ─────────────────────────────────────────────────
        remove_mask = np.zeros((h, w), dtype=np.uint8)
        yolo_res = yolo(frame_bgr, verbose=False, conf=0.35)[0]
        if yolo_res.masks is not None:
            for seg_mask, cls_id in zip(
                yolo_res.masks.data.cpu().numpy(),
                yolo_res.boxes.cls.cpu().numpy().astype(int),
            ):
                if cls_id in YOLO_REMOVE_CLASSES:
                    m = cv2.resize(
                        (seg_mask * 255).astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    remove_mask = cv2.bitwise_or(remove_mask, m)

        # Dilate remove mask slightly to avoid fringe artefacts
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        remove_mask = cv2.dilate(remove_mask, k7, iterations=2)

        # ── Cone detection ───────────────────────────────────────────────────
        cone_debug = _cone_mask_hsv_debug(frame_bgr)
        cone_debug = {**cone_debug, "clean_mask": cv2.bitwise_and(cone_debug["clean_mask"], cv2.bitwise_not(remove_mask))}
        cone_res = cone_model(frame_bgr, verbose=False, conf=cone_conf)[0]
        cone_boxes = _yolo_boxes_to_dicts(cone_res, getattr(cone_res, "names", None))
        cone_centers_px = _cone_boxes_to_centers(cone_boxes, min_conf=cone_conf)

        if use_hsv_fallback and not cone_centers_px:
            n_cones, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cone_debug["clean_mask"], connectivity=8
            )
            for i in range(1, n_cones):
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                area = int(stats[i, cv2.CC_STAT_AREA])
                cone_centers_px.append({"cx": cx, "cy": cy, "area": area, "source": "hsv", "conf": 0.0})

        cone_mask = np.zeros((h, w), dtype=np.uint8)
        for cc in cone_centers_px:
            x, y, bw, bh = cc.get("bbox", [max(0, cc["cx"] - 3), max(0, cc["cy"] - 3), 6, 6])
            cv2.rectangle(cone_mask, (x, y), (min(w - 1, x + bw), min(h - 1, y + bh)), 255, -1)
        cone_mask = cv2.bitwise_and(cone_mask, cv2.bitwise_not(remove_mask))

        # ── Road / grass / cone label map ────────────────────────────────────
        grass_mask = np.zeros((h, w), dtype=np.uint8)
        if segformer is not None:
            label_map = _label_map_segformer(segformer, frame_bgr)
            road_mask = ((label_map == 1).astype(np.uint8)) * 255
            grass_mask = ((label_map == 2).astype(np.uint8)) * 255
            road_mask  = cv2.bitwise_and(road_mask,  cv2.bitwise_not(remove_mask))
            grass_mask = cv2.bitwise_and(grass_mask, cv2.bitwise_not(remove_mask))
            road_mask  = cv2.bitwise_and(road_mask,  cv2.bitwise_not(cone_mask))
            grass_mask = cv2.bitwise_and(grass_mask, cv2.bitwise_not(cone_mask))
        else:
            if yolop_seg is not None:
                road_mask = _road_mask_yolop(yolop_seg, frame_bgr, cone_mask)
            else:
                road_mask = _road_mask_heuristic(frame_bgr, cone_mask)
            road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(remove_mask))
            label_map = np.zeros((h, w), dtype=np.uint8)
            label_map[road_mask > 0] = 1

        # Stamp cones (label 3) and ensure removed regions are 0
        label_map[cone_mask > 0]   = 3
        label_map[remove_mask > 0] = 0

        # ── Combined keep mask (road + cones, no dynamic objects) ────────────
        keep_mask = cv2.bitwise_or(road_mask, cone_mask)

        np.save(str(mask_dir  / f"{stem_name}.npy"), keep_mask)
        np.save(str(label_dir / f"{stem_name}.npy"), label_map)

        cone_data = {
            "frame": stem_name,
            "video": video_stem,
            "src_frame": src_frame,
            "cone_count": len(cone_centers_px),
            "cone_centers_px": cone_centers_px,
        }
        (cone_dir / f"{stem_name}.json").write_text(json.dumps(cone_data))
        cone_summary.append(cone_data)

        if debug_enabled and src_frame % debug_every == 0:
            depth_arr = np.load(str(depth_file))
            if depth_arr.shape != (h, w):
                depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            gps_offset = _gps_offset_for_frame(gps_aligned.get(video_stem, []), src_frame)
            records = _cone_projection_records(
                frame_name=stem_name,
                video_stem=video_stem,
                src_frame=src_frame,
                cone_centers_px=cone_centers_px,
                depth_arr=depth_arr,
                gps_offset=gps_offset,
            )
            projection_records.extend(records)
            _save_frame_debug_panel(
                debug_frames_dir / f"{stem_name}_debug.png",
                frame_bgr,
                depth_arr,
                cone_debug,
                remove_mask,
                road_mask,
                keep_mask,
                cone_centers_px,
                grass_mask=grass_mask,
            )
            _save_frame_world_plot(
                debug_world_dir / f"{stem_name}_world.png",
                records,
                centerline_xy,
                gps_offset,
            )

        # ── Segmentation visualisation (save every 10th frame) ───────────────
        if src_frame % 50 == 0:
            vis = frame_bgr.copy()
            vis[remove_mask > 0] = (vis[remove_mask > 0] * 0.2).astype(np.uint8)
            vis[road_mask > 0]   = (vis[road_mask > 0]   * np.array([0.5, 0.5, 1.0])).astype(np.uint8)
            vis[cone_mask > 0]   = (vis[cone_mask > 0]   * np.array([0.2, 0.5, 1.5]).clip(0,255)).astype(np.uint8)
            cv2.imwrite(str(seg_vis_dir / f"{stem_name}_seg.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

    for cap in video_caps.values():
        cap.release()

    (out_dir / "cone_summary.json").write_text(json.dumps(cone_summary, indent=2))
    print(f"  Saved {len(depth_files)} keep masks → {mask_dir}")
    print(f"  Saved cone summary → {out_dir / 'cone_summary.json'}")
    if debug_enabled:
        records_path = debug_dir / "cone_projection_records.json"
        records_path.write_text(json.dumps(projection_records, indent=2))
        report_path = _write_debug_html_report(debug_dir)
        print(f"  Debug visualization report → {report_path}")
    return mask_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Filtered 3D point cloud from depth × keep_mask
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(
    depth_arr: np.ndarray,
    keep_mask: np.ndarray,
    frame_bgr: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project masked depth pixels to 3D camera-space points.
    Returns (points Nx3, colors Nx3 uint8).
    DA3 outputs *relative* inverse depth; we treat it as metric-ish
    (scale is arbitrary — will be normalised globally later).
    """
    h, w = depth_arr.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    valid = (keep_mask > 0) & (depth_arr > 0) & np.isfinite(depth_arr)

    # Invert relative depth → approximate metric (larger value = farther)
    d = depth_arr[valid].astype(np.float64)
    d = 1.0 / (d + 1e-6)  # relative → disparity-like metric proxy
    d *= depth_scale

    X = (u[valid] - cx) / fx * d
    Y = (v[valid] - cy) / fy * d
    Z = d

    pts = np.stack([X, Y, Z], axis=1)

    # Colors (RGB)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    colors = rgb[valid]

    return pts.astype(np.float32), colors


def run_stage4(args, out_dir: Path, depth_npy_dir: Path, mask_dir: Path) -> Path:
    """
    Merge per-frame point clouds into a single global cloud.
    Uses GPS heading (where available) to rotate frames into world coordinates.
    Saves cloud.npz with keys: xyz (Nx3 float32), rgb (Nx3 uint8).
    """
    print("\n[Stage 4] Building filtered 3D point cloud")

    depth_files = sorted(depth_npy_dir.glob("*.npy"))
    frame_meta_dir = out_dir / "frame_meta"

    # Camera intrinsics (estimated from video resolution + HFoV)
    # We'll derive fx/fy per frame from actual image size
    fov_rad = math.radians(CAM_HFOV_DEG)

    video_caps: dict[str, cv2.VideoCapture] = {}
    for vp in [args.nh1, args.nh2]:
        if vp is not None:
            video_caps[vp.stem] = cv2.VideoCapture(str(vp))

    all_pts: list[np.ndarray]    = []
    all_rgb: list[np.ndarray]    = []

    # Load GPS aligned data for heading computation
    gps_aligned: dict[str, list] = {}
    for vp in [args.nh1, args.nh2]:
        if vp is None:
            continue
        p = out_dir / f"{vp.stem}_gps_aligned.json"
        if p.exists():
            gps_aligned[vp.stem] = json.loads(p.read_text())

    for depth_file in tqdm(depth_files, desc="  Back-project", unit="frame"):
        stem_name = depth_file.stem
        mask_path = mask_dir / f"{stem_name}.npy"
        meta_path = frame_meta_dir / f"{stem_name}.json"
        if not mask_path.exists() or not meta_path.exists():
            continue

        meta       = json.loads(meta_path.read_text())
        video_stem = meta["video"]
        src_frame  = meta["src_frame"]

        cap = video_caps.get(video_stem)
        if cap is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        h, w      = frame_bgr.shape[:2]
        fx = fy   = w / (2 * math.tan(fov_rad / 2))
        cx, cy    = w / 2.0, h / 2.0

        depth_arr = np.load(str(depth_file))
        keep_mask = np.load(str(mask_path))

        if depth_arr.shape != (h, w):
            depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)

        # Subsample to keep cloud manageable (max 5 k pts per frame)
        pts, rgb = depth_to_pointcloud(depth_arr, keep_mask, frame_bgr, fx, fy, cx, cy)
        if len(pts) > 5000:
            idx  = np.random.choice(len(pts), 5000, replace=False)
            pts  = pts[idx]
            rgb  = rgb[idx]

        if len(pts) == 0:
            continue

        # Simple forward translation using GPS (if available) — no rotation yet
        # Full pose estimation would require VO; GPS gives coarse XY offset
        gps_list = gps_aligned.get(video_stem, [])
        offset = np.zeros(3, dtype=np.float32)
        for g in gps_list:
            if g["frame"] == src_frame:
                # Local ENU metres from first GPS point
                lat0 = gps_list[0]["lat"]
                lon0 = gps_list[0]["lon"]
                R    = 6_371_000.0
                ox   = math.radians(g["lon"] - lon0) * math.cos(math.radians(lat0)) * R
                oy   = math.radians(g["lat"] - lat0) * R
                offset = np.array([ox, 0.0, oy], dtype=np.float32)
                break

        all_pts.append(pts + offset)
        all_rgb.append(rgb)

    for cap in video_caps.values():
        cap.release()

    if not all_pts:
        print("  [warn] No point cloud data generated.")
        cloud_path = out_dir / "cloud.npz"
        np.savez(str(cloud_path), xyz=np.zeros((0, 3), dtype=np.float32),
                 rgb=np.zeros((0, 3), dtype=np.uint8))
        return cloud_path

    xyz = np.concatenate(all_pts, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    # Remove statistical outliers (simple percentile clip)
    for axis in range(3):
        lo, hi = np.percentile(xyz[:, axis], 1), np.percentile(xyz[:, axis], 99)
        keep   = (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
        xyz    = xyz[keep]
        rgb    = rgb[keep]

    cloud_path = out_dir / "cloud.npz"
    np.savez(str(cloud_path), xyz=xyz, rgb=rgb)
    print(f"  Point cloud: {len(xyz):,} points → {cloud_path}")
    return cloud_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Track reconstruction: ground plane, cone 3D positions, centerline
# ─────────────────────────────────────────────────────────────────────────────

def _save_cone_world_summary(
    path: Path,
    centerline_pts: list[list],
    raw_cone_candidates: np.ndarray,
    cone_positions: list[np.ndarray],
) -> None:
    cl = np.array(centerline_pts, dtype=np.float32)
    centerline_xy = cl[:, :2] if cl.ndim == 2 and len(cl) > 0 and cl.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float32)
    raw_xy = raw_cone_candidates[:, [0, 2]] if len(raw_cone_candidates) > 0 else np.zeros((0, 2), dtype=np.float32)
    clustered_arr = np.array(cone_positions, dtype=np.float32) if cone_positions else np.zeros((0, 3), dtype=np.float32)
    clustered_xy = clustered_arr[:, [0, 2]] if len(clustered_arr) > 0 else np.zeros((0, 2), dtype=np.float32)
    all_points = np.vstack([p for p in [centerline_xy, raw_xy, clustered_xy] if len(p) > 0]) if any(len(p) > 0 for p in [centerline_xy, raw_xy, clustered_xy]) else np.zeros((0, 2), dtype=np.float32)
    canvas, scale, ox, oy = _world_to_canvas(all_points, width=1200, height=1200)
    if len(centerline_xy) > 1:
        pts = np.array([[int(round(x * scale + ox)), int(round(oy - y * scale))] for x, y in centerline_xy], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (210, 210, 210), 3)
    _draw_world_points(canvas, raw_xy, scale, ox, oy, (128, 190, 255), 3)
    _draw_world_points(canvas, clustered_xy, scale, ox, oy, (0, 85, 255), 8)
    cv2.putText(canvas, "World-space cone projection summary", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "gray=GPS centerline, pale orange=raw projected cones, orange=clustered cones", (25, canvas.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def run_stage5(args, out_dir: Path, cloud_path: Path) -> dict:
    """
    From point cloud + cone data:
      - RANSAC ground plane fit
      - Project cloud onto ground plane → 2D
      - Cluster cone pixel centroids → 3D cone positions via ground plane
      - Fit centerline as midpoint spline between left/right cone rows
    """
    print("\n[Stage 5] Track reconstruction")
    from scipy.spatial import cKDTree
    from scipy.signal import savgol_filter

    data = np.load(str(cloud_path))
    xyz  = data["xyz"]

    if len(xyz) == 0:
        print("  [warn] Empty point cloud, skipping reconstruction.")
        return {}

    # ── RANSAC ground plane (Ax + By + Cz + D = 0) ───────────────────────────
    best_plane  = None
    best_inlier = 0
    thresh      = 0.15   # 15 cm tolerance
    n_iter      = 200

    for _ in range(n_iter):
        idx = np.random.choice(len(xyz), 3, replace=False)
        p1, p2, p3 = xyz[idx]
        n  = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n /= nn
        D  = -np.dot(n, p1)
        dist       = np.abs(xyz @ n + D)
        inlier_cnt = int((dist < thresh).sum())
        if inlier_cnt > best_inlier:
            best_inlier = inlier_cnt
            best_plane  = (n, D)

    if best_plane is None:
        print("  [warn] Ground plane RANSAC failed.")
        return {}

    normal, D = best_plane
    inliers   = xyz[np.abs(xyz @ normal + D) < thresh]
    print(f"  Ground plane: normal={normal.round(3)}, inliers={best_inlier}/{len(xyz)}")

    # ── Project cloud to ground-plane 2D ─────────────────────────────────────
    # Build local 2D coordinate system on the ground plane
    up  = np.array([0., 1., 0.])
    e1  = np.cross(normal, up)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1., 0., 0.])
    e1 /= np.linalg.norm(e1)
    e2  = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)

    proj2d = np.stack([inliers @ e1, inliers @ e2], axis=1)

    # ── Cone 3D positions from depth + pixel centroids ────────────────────────
    cone_summary_path = out_dir / "cone_summary.json"
    cone_3d: list[np.ndarray] = []

    if cone_summary_path.exists():
        cone_summary = json.loads(cone_summary_path.read_text())
        depth_npy_dir = out_dir / "depth_npy"

        gps_aligned = {}
        for jpath in out_dir.glob("*_gps_aligned.json"):
            vname = jpath.name.replace("_gps_aligned.json", "")
            gps_aligned[vname] = json.loads(jpath.read_text())

        for entry in cone_summary:
            stem_name = entry["frame"]
            depth_file = depth_npy_dir / f"{stem_name}.npy"
            if not depth_file.exists():
                continue
            depth_arr = np.load(str(depth_file))
            h, w = depth_arr.shape
            fov_rad = math.radians(CAM_HFOV_DEG)
            fx = fy = w / (2 * math.tan(fov_rad / 2))
            cx, cy = w / 2.0, h / 2.0

            src_frame = entry.get("src_frame", 0)
            video_stem = entry.get("video", stem_name.split("_")[0])
            offset = _gps_offset_for_frame(gps_aligned.get(video_stem, []), src_frame)

            for cc in entry["cone_centers_px"]:
                px, py = cc["cx"], cc["cy"]
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue
                d_rel = float(depth_arr[py, px])
                if d_rel <= 0 or not math.isfinite(d_rel):
                    continue
                d = 1.0 / (d_rel + 1e-6)
                X = (px - cx) / fx * d
                Y = (py - cy) / fy * d
                Z = d
                cone_3d.append(np.array([X, Y, Z], dtype=np.float32) + offset)

    cone_3d_arr = np.array(cone_3d) if cone_3d else np.zeros((0, 3), dtype=np.float32)
    print(f"  Raw cone 3D candidates: {len(cone_3d_arr)}")

    # Cluster cone candidates (DBSCAN-lite via grid)
    cone_positions: list[np.ndarray] = []
    if len(cone_3d_arr) > 0:
        from scipy.spatial import cKDTree as KDTree
        tree = KDTree(cone_3d_arr)
        visited = np.zeros(len(cone_3d_arr), dtype=bool)
        for i, pt in enumerate(cone_3d_arr):
            if visited[i]:
                continue
            neighbors = tree.query_ball_point(pt, r=0.8)
            for j in neighbors:
                visited[j] = True
            cluster = cone_3d_arr[neighbors]
            cone_positions.append(cluster.mean(axis=0))
        print(f"  Clustered cone positions: {len(cone_positions)}")

    # ── Centerline from GPS track ─────────────────────────────────────────────
    gpx_xy_path = out_dir / "gpx_local_xy.json"
    centerline_pts: list[list] = []
    if gpx_xy_path.exists():
        gpx_data = json.loads(gpx_xy_path.read_text())
        pts_raw  = np.array(gpx_data["points"])   # Nx3 (x=East, y=North, z=alt)
        if len(pts_raw) > 10:
            # Smooth with Savitzky-Golay
            win = min(15, len(pts_raw) - (1 if len(pts_raw) % 2 == 0 else 0))
            if win >= 3 and win % 2 == 1:
                xs = savgol_filter(pts_raw[:, 0], win, 3)
                ys = savgol_filter(pts_raw[:, 1], win, 3)
            else:
                xs, ys = pts_raw[:, 0], pts_raw[:, 1]
            for x, y, row in zip(xs, ys, pts_raw):
                centerline_pts.append([float(x), float(y), float(row[2])])
        print(f"  Centerline: {len(centerline_pts)} points (from GPS track)")

    result = {
        "ground_plane_normal": normal.tolist(),
        "ground_plane_D": float(D),
        "cone_positions": [c.tolist() for c in cone_positions],
        "centerline": centerline_pts,
        "proj2d_shape": list(proj2d.shape),
    }
    out_path = out_dir / "track_data.json"
    out_path.write_text(json.dumps(result, indent=2))
    if bool(getattr(args, "debug_vis", False)):
        debug_dir = out_dir / "debug_vis"
        summary_path = debug_dir / "world" / "cone_world_summary.png"
        _save_cone_world_summary(summary_path, centerline_pts, cone_3d_arr, cone_positions)
        report_path = _write_debug_html_report(debug_dir)
        print(f"  Debug world summary → {summary_path}")
        print(f"  Debug visualization report → {report_path}")
    print(f"  Track data → {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — OBJ export
# ─────────────────────────────────────────────────────────────────────────────

def run_stage6(args, out_dir: Path, track_data: dict) -> Path:
    """
    Export track as Wavefront OBJ:
      - Ground mesh (quad strip along centerline ± half-width)
      - Cone markers (small pyramids at each cone position)
    """
    print("\n[Stage 6] OBJ export")
    obj_path = out_dir / "track.obj"
    mtl_path = out_dir / "track.mtl"

    centerline = np.array(track_data.get("centerline", []))
    cone_positions = track_data.get("cone_positions", [])
    half_width = 4.5  # metres — typical FSAE track half-width

    lines = [
        "# FSAE Autocross Track — generated by track_pipeline.py",
        f"mtllib {mtl_path.name}",
        "",
    ]
    verts = []
    faces = []

    # ── Road mesh ─────────────────────────────────────────────────────────────
    if len(centerline) >= 2:
        lines.append("# Road surface")
        lines.append("usemtl road")
        cl = centerline[:, :2] if centerline.ndim == 2 and centerline.shape[1] >= 2 else centerline

        for i, pt in enumerate(cl):
            if i == 0:
                tangent = cl[1] - cl[0]
            elif i == len(cl) - 1:
                tangent = cl[-1] - cl[-2]
            else:
                tangent = cl[i + 1] - cl[i - 1]
            tn = np.linalg.norm(tangent)
            if tn < 1e-6:
                continue
            tangent /= tn
            perp = np.array([-tangent[1], tangent[0]])
            z = float(centerline[i][2]) if centerline.shape[1] > 2 else 0.0

            left  = pt + perp * half_width
            right = pt - perp * half_width
            verts.append(f"v {left[0]:.3f}  {z:.3f}  {left[1]:.3f}")
            verts.append(f"v {right[0]:.3f} {z:.3f}  {right[1]:.3f}")

        # Quad faces (two triangles each)
        for i in range(len(cl) - 1):
            base = i * 2 + 1
            a, b, c, d = base, base + 1, base + 2, base + 3
            faces.append(f"f {a} {b} {d}")
            faces.append(f"f {a} {d} {c}")

    lines += verts
    lines.append("")
    lines += faces
    lines.append("")

    # ── Cone markers (small vertical pyramids) ────────────────────────────────
    if cone_positions:
        lines.append("# Cone markers")
        lines.append("usemtl cone")
        v_offset = len(verts) + 1  # 1-indexed OBJ
        cone_size = 0.35   # metres radius

        for cp in cone_positions:
            cx, cy = float(cp[0]), float(cp[2]) if len(cp) > 2 else 0.0
            cz_base = 0.0
            cz_tip  = 0.75  # 75 cm cone height
            # Base square
            v_offset_now = len(lines) - len(verts) - 3  # rough; we track manually
            base_verts = [
                f"v {cx-cone_size:.3f} {cz_base:.3f} {cy-cone_size:.3f}",
                f"v {cx+cone_size:.3f} {cz_base:.3f} {cy-cone_size:.3f}",
                f"v {cx+cone_size:.3f} {cz_base:.3f} {cy+cone_size:.3f}",
                f"v {cx-cone_size:.3f} {cz_base:.3f} {cy+cone_size:.3f}",
                f"v {cx:.3f} {cz_tip:.3f} {cy:.3f}",
            ]
            lines += base_verts

    (obj_path).write_text("\n".join(lines))

    # ── MTL material file ──────────────────────────────────────────────────────
    mtl_path.write_text(
        "newmtl road\n"
        "Kd 0.35 0.35 0.35\n"
        "Ka 0.1 0.1 0.1\n\n"
        "newmtl cone\n"
        "Kd 1.0 0.45 0.0\n"
        "Ka 0.2 0.1 0.0\n"
    )

    print(f"  OBJ → {obj_path}")
    print(f"  MTL → {mtl_path}")
    return obj_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — OpenDRIVE (XODR) export
# ─────────────────────────────────────────────────────────────────────────────

def _chord_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a polyline (metres)."""
    diffs = np.diff(pts, axis=0)
    segs  = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def run_stage7(args, out_dir: Path, track_data: dict) -> Path:
    """
    Export minimal OpenDRIVE 1.6 XML.
    Encodes the GPS centerline as a poly3 reference line,
    and places one 'driving' lane of width 9 m (FSAE nominal).
    """
    print("\n[Stage 7] OpenDRIVE (XODR) export")

    centerline = np.array(track_data.get("centerline", []))
    if len(centerline) < 2:
        print("  [warn] Not enough centerline points for XODR — skipping.")
        return out_dir / "track.xodr"

    cl_xy = centerline[:, :2]
    s_arr = _chord_lengths(cl_xy)
    total_length = float(s_arr[-1])

    # Build geometry list: one line segment per GPS interval
    geom_elems = []
    for i in range(len(cl_xy) - 1):
        dx = cl_xy[i + 1][0] - cl_xy[i][0]
        dy = cl_xy[i + 1][1] - cl_xy[i][1]
        hdg = math.atan2(dy, dx)
        seg_len = float(s_arr[i + 1] - s_arr[i])
        if seg_len < 1e-4:
            continue
        geom_elems.append(
            f'        <geometry s="{s_arr[i]:.4f}" x="{cl_xy[i][0]:.4f}" '
            f'y="{cl_xy[i][1]:.4f}" hdg="{hdg:.6f}" length="{seg_len:.4f}">\n'
            f'          <line/>\n'
            f'        </geometry>'
        )

    # Lane width — FSAE track ≥ 3 m per side, nominal total 9 m
    lane_width = 9.0

    # Cone objects extraction
    cones = np.array(track_data.get("cone_positions", []))
    object_elems = []
    if len(cones) > 0 and len(cl_xy) >= 2:
        object_elems.append("    <objects>")
        for obj_id, cone in enumerate(cones):
            cx, cy, cz = cone
            min_dist = float('inf')
            best_s, best_t, best_hdg = 0.0, 0.0, 0.0
            
            for i in range(len(cl_xy) - 1):
                p1 = cl_xy[i]
                p2 = cl_xy[i+1]
                v = p2 - p1
                w = np.array([cx, cy]) - p1
                v_sq = np.dot(v, v)
                if v_sq < 1e-6:
                    continue
                proj = np.dot(w, v) / v_sq
                proj_clamped = max(0.0, min(1.0, float(proj)))
                closest_pt = p1 + proj_clamped * v
                dist = float(np.linalg.norm(np.array([cx, cy]) - closest_pt))
                
                if dist < min_dist:
                    min_dist = dist
                    best_s = float(s_arr[i]) + proj_clamped * float(np.sqrt(v_sq))
                    cross = float(v[0] * w[1] - v[1] * w[0])
                    best_t = dist if cross > 0 else -dist
                    best_hdg = math.atan2(v[1], v[0])
            
            object_elems.append(
                f'      <object id="{obj_id+1}" name="cone" s="{best_s:.4f}" t="{best_t:.4f}" '
                f'zOffset="{cz:.4f}" type="obstacle" validLength="0.3" validWidth="0.3" '
                f'height="0.75" hdg="{best_hdg:.4f}">\n'
                f'        <outline>\n'
                f'          <cornerLocal u="0.15" v="0.15" z="0.0"/>\n'
                f'          <cornerLocal u="-0.15" v="0.15" z="0.0"/>\n'
                f'          <cornerLocal u="-0.15" v="-0.15" z="0.0"/>\n'
                f'          <cornerLocal u="0.15" v="-0.15" z="0.0"/>\n'
                f'        </outline>\n'
                f'      </object>'
            )
        object_elems.append("    </objects>")

    xodr = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="6" name="NHAutocross" version="1.0"
          date="{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}"
          north="0.0" south="0.0" east="0.0" west="0.0">
    <geoReference><![CDATA[+proj=latlong +datum=WGS84]]></geoReference>
  </header>

  <road name="AutocrossTrack" length="{total_length:.4f}" id="1" junction="-1">
    <link/>
    <planView>
{chr(10).join(geom_elems)}
    </planView>
    <elevationProfile>
      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
    </elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false">
            <roadMark sOffset="0.0" type="solid" weight="standard"
                      color="white" width="0.12"/>
          </lane>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="{lane_width/2:.2f}" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="solid" weight="standard"
                      color="white" width="0.12"/>
          </lane>
        </right>
        <left>
          <lane id="1" type="driving" level="false">
            <width sOffset="0.0" a="{lane_width/2:.2f}" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="solid" weight="standard"
                      color="white" width="0.12"/>
          </lane>
        </left>
      </laneSection>
    </lanes>
{chr(10).join(object_elems)}
  </road>
</OpenDRIVE>
"""

    xodr_path = out_dir / "track.xodr"
    xodr_path.write_text(xodr)
    print(f"  XODR → {xodr_path}  (road length: {total_length:.1f} m)")
    return xodr_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_args(argv=None):
    p = argparse.ArgumentParser(description="FSAE Track Reconstruction Pipeline")
    p.add_argument("--nh1",    type=Path, default=Path("NHautocross/NH1.mp4"))
    p.add_argument("--nh2",    type=Path, default=Path("NHautocross/NH2.mp4"))
    p.add_argument("--gpx",    type=Path, default=Path("NHautocross/autocross.gpx"))
    p.add_argument("--output", type=Path, default=Path("results/track"))
    p.add_argument("--model",  default="depth-anything/DA3MONO-LARGE",
                   help="DA3 HuggingFace model ID")
    p.add_argument("--fps",    type=int,  default=5,
                   help="Depth sample rate (frames/sec, default 5)")
    p.add_argument("--stage",  type=int,  default=0,
                   help="Run only this stage (1-7); 0 = all")
    p.add_argument("--debug-vis", action="store_true",
                   help="Save per-frame and world-space debug visualization report")
    p.add_argument("--debug-every", type=int, default=50,
                   help="Save one debug visualization panel every N source frames")
    p.add_argument("--cone-model", default="yolov8s-world.pt",
                   help="YOLO/YOLO-World model for cone detection")
    p.add_argument("--cone-conf", type=float, default=0.25,
                   help="Minimum confidence for model-based cone detections")
    p.add_argument("--cone-hsv-fallback", action="store_true",
                   help="Use HSV cone components only when the cone model finds no cones")
    p.add_argument("--debug-sample-frames", type=int, default=0,
                   help="Run Stage 3 on a deterministic random subset of N depth frames")
    p.add_argument("--debug-sample-seed", type=int, default=0,
                   help="Seed for --debug-sample-frames selection")
    p.add_argument("--road-method", choices=["yolop", "heuristic"], default="yolop",
                   help="Fallback road segmentation when --seg-model is yolop-heuristic")
    p.add_argument("--seg-model", choices=["segformer-cityscapes", "yolop-heuristic"],
                   default="segformer-cityscapes",
                   help="Per-pixel road/grass segmenter (default: segformer-cityscapes)")
    return p.parse_args(argv)


def main():
    args = build_args()

    # Validate inputs
    for attr, label in [("nh1", "NH1"), ("nh2", "NH2"), ("gpx", "GPX")]:
        p = getattr(args, attr)
        if p and not p.exists():
            print(f"  [warn] {label} file not found: {p}")
            setattr(args, attr, None)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  FSAE Track Reconstruction")
    print(f"  Output → {out_dir.resolve()}")
    print(f"{'='*60}")

    run_all = args.stage == 0

    # Stage 1 — GPX alignment
    if run_all or args.stage == 1:
        if args.gpx:
            run_stage1(args, out_dir)
        else:
            print("\n[Stage 1] Skipped — no GPX file.")

    # Stage 2 — Depth extraction
    if run_all or args.stage == 2:
        depth_npy_dir = run_stage2(args, out_dir, sample_fps=args.fps)
    else:
        depth_npy_dir = out_dir / "depth_npy"

    # Stage 3 — Segmentation + filtering
    if run_all or args.stage == 3:
        mask_dir = run_stage3(args, out_dir, depth_npy_dir)
    else:
        mask_dir = out_dir / "keep_masks"

    # Stage 4 — Point cloud
    if run_all or args.stage == 4:
        cloud_path = run_stage4(args, out_dir, depth_npy_dir, mask_dir)
    else:
        cloud_path = out_dir / "cloud.npz"

    # Stage 5 — Track reconstruction
    if run_all or args.stage == 5:
        track_data = run_stage5(args, out_dir, cloud_path)
    else:
        td_path = out_dir / "track_data.json"
        track_data = json.loads(td_path.read_text()) if td_path.exists() else {}

    # Stage 6 — OBJ
    if run_all or args.stage == 6:
        if track_data:
            run_stage6(args, out_dir, track_data)
        else:
            print("\n[Stage 6] Skipped — no track data.")

    # Stage 7 — XODR
    if run_all or args.stage == 7:
        if track_data:
            run_stage7(args, out_dir, track_data)
        else:
            print("\n[Stage 7] Skipped — no track data.")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.  Results → {out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
