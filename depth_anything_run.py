#!/usr/bin/env -S uv run --python 3.12 --script
# /// script
# requires-python = ">=3.11, <3.13"
# dependencies = [
#   "torch>=2",
#   "torchvision",
#   "numpy<2",
#   "opencv-python",
#   "pillow",
#   "einops",
#   "huggingface_hub",
#   "imageio",
#   "trimesh",
#   "omegaconf",
#   "typer",
#   "safetensors",
#   "e3nn",
#   "moviepy<2",
#   "plyfile",
#   "pillow-heif",
#   "requests",
#   "tqdm",
#   "matplotlib",
#   "addict",
#   "open3d",
# ]
# ///
"""
Depth Anything 3 – Monocular Depth from Video
Runs on Apple Silicon (M2) using MPS backend.

Usage:
  uv run depth_anything_run.py [VIDEO_PATH] [--fps 5] [--model DA3MONO-LARGE] [--output ./depth_output]

Requirements:
  - Depth-Anything-3 must be cloned in the same directory (or set DA3_DIR env var).
  - Run from /Users/hz/GitHub/autonomous/
"""

import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path
from datetime import datetime

# ── Ensure DA3 is on sys.path ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
DA3_DIR = Path(os.environ.get("DA3_DIR", SCRIPT_DIR / "Depth-Anything-3"))
DA3_SRC = DA3_DIR / "src"


def clone_da3():
    if not DA3_DIR.exists():
        print(f"[setup] Cloning Depth-Anything-3 → {DA3_DIR}")
        subprocess.run(
            ["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(DA3_DIR)],
            check=True,
        )
    else:
        print(f"[setup] Using existing DA3 clone at {DA3_DIR}")


def install_da3():
    """Install DA3 into the current uv script venv (no-deps since all deps are in the script header)."""
    print(f"[setup] Installing depth_anything_3 package from {DA3_DIR}...")
    # UV script envs don't include pip; use 'uv pip' instead.
    # --no-deps: all DA3 dependencies are already declared in the inline script header.
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable,
         "--no-deps", "--quiet", "-e", str(DA3_DIR)],
        check=True,
    )



def main():
    parser = argparse.ArgumentParser(description="Depth Anything 3 – Video Depth Extraction")
    parser.add_argument(
        "video",
        nargs="?",
        default="/Users/hz/GitHub/autonomous/NHautocross/NH1.mp4",
        help="Path to input MP4 video",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Frames per second to process (default: 5, lower = faster)",
    )
    parser.add_argument(
        "--model", default="depth-anything/DA3MONO-LARGE",
        help="HuggingFace model ID (default: DA3MONO-LARGE, good for M2)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ./depth_output/TIMESTAMP)",
    )
    parser.add_argument(
        "--chunk", type=int, default=2,
        help="Batch chunk size (reduce to 1 if OOM, default: 2)",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else SCRIPT_DIR / "depth_output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────────
    clone_da3()

    # Add DA3 src to path so we can import it (must happen before import attempt)
    if str(DA3_SRC) not in sys.path:
        sys.path.insert(0, str(DA3_SRC))

    try:
        from depth_anything_3.api import DepthAnything3
        print("[setup] depth_anything_3 already importable ✓")
    except ImportError:
        install_da3()
        # Re-add after install in case reload is needed
        if str(DA3_SRC) not in sys.path:
            sys.path.insert(0, str(DA3_SRC))
        from depth_anything_3.api import DepthAnything3

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    # ── Device selection ──────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  Depth Anything 3 – Video Depth Extraction")
    print(f"{'='*60}")
    print(f"  Video   : {video_path}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"  FPS     : {args.fps}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}\n")

    # ── Frame extraction ──────────────────────────────────────────────────────
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(src_fps / args.fps)))

    print(f"[1/3] Extracting frames (src={src_fps:.1f}fps → target={args.fps}fps, every {frame_interval} frames)...")

    frame_idx = 0
    saved = 0
    pbar = tqdm(total=total_frames, desc="Extracting", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_path = frame_dir / f"frame_{saved:06d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved += 1
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"  → Extracted {saved} frames to {frame_dir}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[2/3] Loading model: {args.model}")
    print("  (First run will download model weights from HuggingFace...)")
    model = DepthAnything3.from_pretrained(args.model).to(device)
    model.eval()
    print("  → Model loaded ✓\n")

    # ── Depth inference ───────────────────────────────────────────────────────
    depth_npy_dir = output_dir / "depth_npy"
    depth_vis_dir = output_dir / "depth_vis"
    depth_npy_dir.mkdir(exist_ok=True)
    depth_vis_dir.mkdir(exist_ok=True)

    image_paths = sorted(frame_dir.glob("*.png"))
    print(f"[3/3] Running depth inference on {len(image_paths)} frames (chunk={args.chunk})...")

    def normalize_depth(depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32)
        valid = np.isfinite(d)
        if valid.sum() == 0:
            return np.zeros(d.shape, dtype=np.uint8)
        lo, hi = np.percentile(d[valid], 2), np.percentile(d[valid], 98)
        if hi <= lo:
            hi = lo + 1e-6
        vis = np.clip((d - lo) / (hi - lo), 0, 1)
        return (vis * 255).astype(np.uint8)

    # Apply a colormap for nicer visualization
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("inferno")

    chunk = args.chunk
    for start in tqdm(range(0, len(image_paths), chunk), desc="Depth inference", unit="batch"):
        batch_paths = [str(p) for p in image_paths[start : start + chunk]]
        with torch.no_grad():
            prediction = model.inference(batch_paths)
        depths = prediction.depth  # [N, H, W]
        for i, depth in enumerate(depths):
            idx = start + i
            stem = f"frame_{idx:06d}"
            depth_array = np.asarray(depth, dtype=np.float32)

            # Raw depth map
            np.save(str(depth_npy_dir / f"{stem}_depth.npy"), depth_array)

            # Colormap visualization
            norm = normalize_depth(depth_array) / 255.0
            colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(colored).save(str(depth_vis_dir / f"{stem}_depth_vis.png"))

    print(f"\n✅ Depth inference complete!")
    print(f"   Raw depth maps (.npy) → {depth_npy_dir}")
    print(f"   Visualizations (.png) → {depth_vis_dir}")

    # ── Quick preview summary ─────────────────────────────────────────────────
    vis_count = len(list(depth_vis_dir.glob("*.png")))
    print(f"\n   Generated {vis_count} depth visualization frames")
    print(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    main()
