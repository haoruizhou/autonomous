"""Stage C — per-frame semantic masks via Mask2Former-Cityscapes (safetensors).

Runs once per undistorted image and writes:
    <project>/labels/<image_stem>.npy        # uint8 HxW, project-defined class ids
    <project>/labels/<image_stem>_vis.png    # quick colourised visualisation

Project class ids (kept tiny on purpose; downstream meshing keys off these):
    0 = other / sky / unknown
    1 = road      (Cityscapes road, sidewalk)
    2 = grass     (Cityscapes terrain, vegetation)
    3 = cone      (stamped later by cones.py — never set here)
    4 = removed   (Cityscapes person, rider, car, truck, bus, motorcycle, bicycle)

Cityscapes id reference (Mask2Former config.id2label):
    0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole,
    6 traffic light, 7 traffic sign, 8 vegetation, 9 terrain, 10 sky,
    11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train,
    17 motorcycle, 18 bicycle.

Mask2Former checkpoint: facebook/mask2former-swin-tiny-cityscapes-semantic
(ships safetensors → loads cleanly on torch < 2.6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

OTHER, ROAD, GRASS, CONE, REMOVED = 0, 1, 2, 3, 4

_CS_ROAD = (0, 1)
_CS_GRASS = (8, 9)
_CS_REMOVED = (11, 12, 13, 14, 15, 17, 18)

_DEFAULT_MODEL = "facebook/mask2former-swin-tiny-cityscapes-semantic"

# Visualisation palette (BGR for cv2)
_VIS = {
    OTHER:   (0, 0, 0),
    ROAD:    (160, 160, 160),
    GRASS:   (60, 200, 80),
    CONE:    (0, 140, 255),
    REMOVED: (180, 60, 60),
}


def _load_model(name: str = _DEFAULT_MODEL):
    """Load Mask2Former + image processor on the best available device."""
    import torch
    from transformers import (
        AutoImageProcessor,
        Mask2FormerForUniversalSegmentation,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Loading {name} on {device} …")
    processor = AutoImageProcessor.from_pretrained(name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(device).eval()
    return model, processor, device


def _cityscapes_to_project(pred: np.ndarray) -> np.ndarray:
    """Map a Cityscapes semantic prediction to the project's 5-class scheme."""
    out = np.zeros_like(pred, dtype=np.uint8)
    for cid in _CS_ROAD:
        out[pred == cid] = ROAD
    for cid in _CS_GRASS:
        out[pred == cid] = GRASS
    for cid in _CS_REMOVED:
        out[pred == cid] = REMOVED
    return out


def label_image(model, processor, device, frame_bgr: np.ndarray) -> np.ndarray:
    """Run Mask2Former and return a HxW uint8 label map in project class ids."""
    import torch
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(h, w)]
    )[0].cpu().numpy().astype(np.int16)
    return _cityscapes_to_project(pred)


def label_images_batch(
    model, processor, device, frames_bgr: list, batch_size: int = 8
) -> list:
    """Process multiple frames in batched GPU forward passes.

    Returns a list of HxW uint8 numpy arrays (project class ids), one per input frame.
    Falls back to single-image processing per frame if a batch fails (e.g. OOM).
    """
    import torch

    results: list = [None] * len(frames_bgr)

    # Build (index, PIL image, (H, W)) tuples for all frames.
    items = []
    for i, frame_bgr in enumerate(frames_bgr):
        h, w = frame_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        items.append((i, pil, (h, w)))

    # Process in chunks of batch_size.
    for chunk_start in range(0, len(items), batch_size):
        chunk = items[chunk_start: chunk_start + batch_size]
        indices = [c[0] for c in chunk]
        pil_images = [c[1] for c in chunk]
        target_sizes = [c[2] for c in chunk]

        try:
            inputs = processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            preds = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            for idx, pred in zip(indices, preds):
                arr = pred.cpu().numpy().astype(np.int16)
                results[idx] = _cityscapes_to_project(arr)

        except RuntimeError as exc:
            # Likely OOM — fall back to single-image processing for this chunk.
            print(f"  [warn] batch inference failed ({exc}); falling back to single-image for chunk")
            for orig_idx, pil_img, (h, w) in chunk:
                frame_bgr = frames_bgr[orig_idx]
                try:
                    results[orig_idx] = label_image(model, processor, device, frame_bgr)
                except Exception as inner_exc:
                    print(f"  [warn] single-image fallback also failed for frame {orig_idx}: {inner_exc}")
                    results[orig_idx] = np.zeros(
                        (frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8
                    )

    return results


def colorise(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cid, color in _VIS.items():
        out[label_map == cid] = color
    return out


def label_project(
    project_dir: Path,
    images_subdir: str = "undistorted/images",
    out_subdir: str = "labels",
    model_name: str = _DEFAULT_MODEL,
    image_names: Optional[Iterable[str]] = None,
    save_vis: bool = True,
    batch_size: int = 8,
) -> dict:
    """Label every undistorted image in the project. Skips files that already exist."""
    project_dir = Path(project_dir)
    img_dir = project_dir / images_subdir
    if not img_dir.is_dir():
        # Fall back to original frames if undistort wasn't run yet.
        img_dir = project_dir / "images"
        print(f"  [info] {project_dir/'undistorted/images'} missing — using {img_dir}")

    out_dir = project_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = _load_model(model_name)

    if image_names is None:
        image_names = sorted(p.name for p in img_dir.glob("*.jpg"))
    image_names = list(image_names)

    summary = {"n_images": len(image_names), "n_labelled": 0, "skipped": 0}

    # Filter to unprocessed names only — do NOT pre-load all frames.
    pending_names = [
        n for n in image_names
        if not (out_dir / f"{Path(n).stem}.npy").exists()
    ]
    summary["skipped"] = len(image_names) - len(pending_names)

    # Stream in batches — load from disk, process, write, free immediately.
    for batch_start in range(0, len(pending_names), batch_size):
        batch_names = pending_names[batch_start: batch_start + batch_size]
        batch_frames = []
        for name in batch_names:
            frame_bgr = cv2.imread(str(img_dir / name))
            if frame_bgr is not None:
                batch_frames.append(frame_bgr)

        if not batch_frames:
            continue

        labels = label_images_batch(model, processor, device, batch_frames, batch_size=batch_size)

        for name, label in zip(batch_names, labels):
            stem = Path(name).stem
            out_npy = out_dir / f"{stem}.npy"
            np.save(str(out_npy), label)
            if save_vis:
                cv2.imwrite(str(out_dir / f"{stem}_vis.png"), colorise(label))
            summary["n_labelled"] += 1
            if summary["n_labelled"] % 20 == 0:
                print(f"    {summary['n_labelled']}/{len(pending_names)} labelled")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Labels written to {out_dir} "
          f"(labelled={summary['n_labelled']}, skipped={summary['skipped']})")
    return summary


def generate_opensfm_masks(
    project_dir: Path,
    images_subdir: str = "images",
    out_subdir: str = "masks",
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 8,
) -> dict:
    """Generate binary OpenSfM feature masks from semantic segmentation.

    Writes <project>/masks/<stem>.png for each image: white (255) pixels are
    masked out (people, riders, vehicles, bikes) so OpenSfM skips keypoints
    there during detect_features.  Requires ``use_masks: yes`` in config.yaml
    (added automatically by extract.write_project).

    Skips images whose mask file already exists.
    """
    project_dir = Path(project_dir)
    img_dir = project_dir / images_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    out_dir = project_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = _load_model(model_name)

    image_names = sorted(p.name for p in img_dir.glob("*.jpg"))
    summary = {"n_images": len(image_names), "n_written": 0, "skipped": 0}

    pending_names = [n for n in image_names if not (out_dir / f"{n}.png").exists()]
    summary["skipped"] = len(image_names) - len(pending_names)

    for batch_start in range(0, len(pending_names), batch_size):
        batch_names = pending_names[batch_start: batch_start + batch_size]
        batch_frames = [cv2.imread(str(img_dir / n)) for n in batch_names]
        batch_frames = [f for f in batch_frames if f is not None]
        if not batch_frames:
            continue

        labels = label_images_batch(model, processor, device, batch_frames, batch_size=batch_size)

        for name, label in zip(batch_names, labels):
            # REMOVED class → 255 (masked), everything else → 0 (valid)
            mask = np.where(label == REMOVED, np.uint8(255), np.uint8(0))
            cv2.imwrite(str(out_dir / f"{name}.png"), mask)
            summary["n_written"] += 1
            if summary["n_written"] % 20 == 0:
                print(f"    {summary['n_written']}/{len(pending_names)} masks written")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Masks written to {out_dir} "
          f"(written={summary['n_written']}, skipped={summary['skipped']})")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run Mask2Former-Cityscapes on a project")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--mode", choices=("labels", "masks"), default="labels",
                   help="labels: label undistorted images; masks: generate OpenSfM feature masks")
    p.add_argument("--images-subdir", default=None,
                   help="Override images subdirectory (default: undistorted/images for labels, images for masks)")
    p.add_argument("--out-subdir", default=None,
                   help="Override output subdirectory (default: labels or masks)")
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--model", default=_DEFAULT_MODEL)
    args = p.parse_args()
    if args.mode == "masks":
        generate_opensfm_masks(
            args.project,
            images_subdir=args.images_subdir or "images",
            out_subdir=args.out_subdir or "masks",
            model_name=args.model,
        )
    else:
        label_project(args.project,
                      images_subdir=args.images_subdir or "undistorted/images",
                      out_subdir=args.out_subdir or "labels",
                      model_name=args.model,
                      save_vis=not args.no_vis)
