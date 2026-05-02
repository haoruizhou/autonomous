"""
Semantic Segmentation Pipeline — Lane, Road, and Cone Mapping
=============================================================
Uses YOLOv8-seg (COCO) for instance segmentation and heuristics to
distinguish road surface, lane markings, and traffic cones.

Classes targeted from COCO:
  - cone  → via custom orange-HSV heuristic (COCO has no "cone" class)
  - road  → detected as the dominant dark ground plane
  - lane  → bright stripe regions on road surface

Output: color-coded segmentation video + per-frame stats JSON.

Usage:
    uv run python segment_video.py --input data/NH1.mp4 --output results/NH1_seg.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------
COLORS = {
    "road":   (80,  80,  80),    # dark gray
    "lane":   (255, 255, 255),   # white
    "cone":   (0,   140, 255),   # orange-ish in BGR
    "bg":     (20,   20,  20),   # near-black background
}

ALPHA = 0.55   # mask overlay opacity


# ---------------------------------------------------------------------------
# Cone detection — HSV-based (orange traffic cones)
# ---------------------------------------------------------------------------
def detect_cones_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask of orange traffic cone regions."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Orange hue range — two wraps around the HSV wheel
    lower1 = np.array([0,   120,  80], dtype=np.uint8)
    upper1 = np.array([20,  255, 255], dtype=np.uint8)
    lower2 = np.array([165, 120,  80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    # Morphological cleanup — remove noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove very small blobs (< 80 px²) and very large ones (likely sky/sunset)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 80 <= area <= 12000:
            clean[labels == i] = 255

    return clean


# ---------------------------------------------------------------------------
# Road + Lane detection — combined approach
# ---------------------------------------------------------------------------
def detect_road_and_lane(
    frame_bgr: np.ndarray,
    cone_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (road_mask, lane_mask) as uint8 single-channel masks.

    Strategy:
    1. Convert to grayscale + LAB.
    2. Use adaptive threshold to isolate the bright lane markings.
    3. Estimate road as the large dark-gray connected region below
       the horizon (bottom 60 % of frame), excluding cones and lane marks.
    """
    h, w = frame_bgr.shape[:2]
    horizon_y = int(h * 0.40)   # below this line is "road zone"

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L    = lab[:, :, 0]

    # ---- Lane markings: bright stripes on dark asphalt ----
    # Compute local background luminance with a large blur
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=25)
    diff = L.astype(np.int16) - blurred.astype(np.int16)
    bright_stripe = np.clip(diff, 0, 255).astype(np.uint8)

    # Threshold: keep pixels that are significantly brighter than neighbours
    _, lane_raw = cv2.threshold(bright_stripe, 28, 255, cv2.THRESH_BINARY)

    # Restrict to road zone (below horizon)
    lane_raw[:horizon_y, :] = 0

    # Exclude cone regions from lane mask
    lane_raw[cone_mask > 0] = 0

    # Morphological polish
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    lane_mask = cv2.morphologyEx(lane_raw, cv2.MORPH_CLOSE, k3, iterations=2)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, k3, iterations=1)

    # Remove tiny specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
    lane_clean = np.zeros_like(lane_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            lane_clean[labels == i] = 255
    lane_mask = lane_clean

    # ---- Road surface: dark-gray asphalt in road zone ----
    # Road tends to be dark (low L in LAB) and low-saturation
    road_zone = np.zeros(L.shape, dtype=np.uint8)
    road_zone[horizon_y:, :] = 255

    # Dark-gray: L between 30–145, very low saturation (a,b near 128)
    a_ch = lab[:, :, 1].astype(np.int16) - 128
    b_ch = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a_ch**2 + b_ch**2).astype(np.float32)

    dark_gray = (
        (L >= 20) & (L <= 150) &
        (chroma < 30)
    ).astype(np.uint8) * 255

    road_raw = cv2.bitwise_and(dark_gray, road_zone)

    # Exclude cones + lane markings from road
    exclusion = cv2.bitwise_or(cone_mask, lane_mask)
    road_raw  = cv2.bitwise_and(road_raw, cv2.bitwise_not(exclusion))

    # Keep the largest connected component (main road slab)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road_raw = cv2.morphologyEx(road_raw, cv2.MORPH_CLOSE, k5, iterations=4)
    road_raw = cv2.morphologyEx(road_raw, cv2.MORPH_OPEN,  k5, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_raw, connectivity=8)
    road_mask = np.zeros_like(road_raw)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        road_mask[labels == largest] = 255

    return road_mask, lane_mask


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def render_overlay(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray,
    lane_mask: np.ndarray,
    cone_mask: np.ndarray,
) -> np.ndarray:
    """Blend colored semantic masks over the original frame."""
    overlay = frame_bgr.copy().astype(np.float32)

    def apply(mask, color_bgr):
        colored = np.full_like(frame_bgr, color_bgr, dtype=np.float32)
        m3 = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
        nonlocal overlay
        overlay = overlay * (1 - ALPHA * m3) + colored * (ALPHA * m3)

    apply(road_mask, COLORS["road"])
    apply(lane_mask, COLORS["lane"])
    apply(cone_mask, COLORS["cone"])

    result = np.clip(overlay, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
def draw_legend(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    legend_x, legend_y = 18, 18
    items = [
        ("Road",  COLORS["road"]),
        ("Lane",  COLORS["lane"]),
        ("Cone",  COLORS["cone"]),
    ]
    box_s = 18
    pad   = 6
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    thick  = 1

    # Semi-transparent backdrop
    bx1, by1 = legend_x - pad, legend_y - pad
    bx2 = bx1 + 120
    by2 = by1 + len(items) * (box_s + 6) + pad
    roi = frame[by1:by2, bx1:bx2].astype(np.float32)
    black = np.zeros_like(roi)
    frame[by1:by2, bx1:bx2] = cv2.addWeighted(roi, 0.4, black, 0.6, 0).astype(np.uint8)

    for i, (label, color) in enumerate(items):
        y = legend_y + i * (box_s + 6)
        cv2.rectangle(frame, (legend_x, y), (legend_x + box_s, y + box_s), color, -1)
        cv2.putText(frame, label, (legend_x + box_s + 6, y + box_s - 4),
                    font, fscale, (220, 220, 220), thick, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Stats overlay
# ---------------------------------------------------------------------------
def draw_stats(frame: np.ndarray, stats: dict, frame_idx: int, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    total_px = h * w
    txt_y = h - 14
    font = cv2.FONT_HERSHEY_SIMPLEX

    road_pct = stats.get("road_pct", 0)
    lane_pct = stats.get("lane_pct", 0)
    cone_cnt = stats.get("cone_count", 0)

    info = (
        f"Frame {frame_idx:05d}  |  "
        f"Road {road_pct:.1f}%  Lane {lane_pct:.1f}%  Cones ~{cone_cnt}"
    )
    # Shadow
    cv2.putText(frame, info, (12, txt_y + 1), font, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, info, (12, txt_y),     font, 0.48, (200, 240, 200), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_video(input_path: Path, output_path: Path, max_frames: int = 0,
                  show_preview: bool = False) -> list[dict]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frm = min(total_frm, max_frames)

    print(f"  Input : {input_path}  [{src_w}×{src_h} @ {src_fps:.1f} fps, {total_frm} frames]")
    print(f"  Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use mp4v codec — most compatible on macOS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (src_w, src_h))

    per_frame_stats: list[dict] = []
    total_px = src_w * src_h

    t0 = time.perf_counter()
    with tqdm(total=total_frm, unit="frame", ncols=80) as pbar:
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames > 0 and fidx >= max_frames):
                break

            # --- Segmentation ---
            cone_mask            = detect_cones_hsv(frame)
            road_mask, lane_mask = detect_road_and_lane(frame, cone_mask)

            # --- Stats ---
            road_pct  = 100 * np.count_nonzero(road_mask) / total_px
            lane_pct  = 100 * np.count_nonzero(lane_mask) / total_px

            # Cone count = connected components
            n_cones, *_ = cv2.connectedComponents(cone_mask, connectivity=8)
            n_cones = max(0, n_cones - 1)

            stats = {
                "frame": fidx,
                "road_pct":   round(road_pct,  2),
                "lane_pct":   round(lane_pct,  2),
                "cone_count": n_cones,
            }
            per_frame_stats.append(stats)

            # --- Render ---
            vis = render_overlay(frame, road_mask, lane_mask, cone_mask)
            vis = draw_legend(vis)
            vis = draw_stats(vis, stats, fidx, src_fps)

            writer.write(vis)

            if show_preview:
                cv2.imshow("Segmentation", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            fidx += 1
            pbar.update(1)

    elapsed = time.perf_counter() - t0
    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    fps_proc = fidx / elapsed if elapsed > 0 else 0
    print(f"\n  Processed {fidx} frames in {elapsed:.1f}s  ({fps_proc:.1f} fps)")

    return per_frame_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Lane / Road / Cone Semantic Segmentation"
    )
    parser.add_argument(
        "--input",  "-i",
        default="data/NH1.mp4",
        help="Path to input video (default: data/NH1.mp4)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/NH1_seg.mp4",
        help="Path to output segmentation video",
    )
    parser.add_argument(
        "--max-frames", "-n",
        type=int, default=0,
        help="Limit processing to N frames (0 = all)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live preview window while processing",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("  Autonomous Racing — Semantic Segmentation")
    print("  Classes: Road · Lane · Cone")
    print("=" * 60)

    stats = process_video(
        input_path,
        output_path,
        max_frames=args.max_frames,
        show_preview=args.preview,
    )

    # Save per-frame JSON alongside the video
    stats_path = output_path.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  Stats  → {stats_path}")
    print(f"  Video  → {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
