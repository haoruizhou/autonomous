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
    for name in image_names:
        stem = Path(name).stem
        out_npy = out_dir / f"{stem}.npy"
        if out_npy.exists():
            summary["skipped"] += 1
            continue
        frame_bgr = cv2.imread(str(img_dir / name))
        if frame_bgr is None:
            continue
        label = label_image(model, processor, device, frame_bgr)
        np.save(str(out_npy), label)
        if save_vis:
            cv2.imwrite(str(out_dir / f"{stem}_vis.png"), colorise(label))
        summary["n_labelled"] += 1
        if summary["n_labelled"] % 20 == 0:
            print(f"    {summary['n_labelled']}/{len(image_names)} labelled")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Labels written to {out_dir} "
          f"(labelled={summary['n_labelled']}, skipped={summary['skipped']})")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run Mask2Former-Cityscapes on a project")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--images-subdir", default="undistorted/images")
    p.add_argument("--out-subdir", default="labels")
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--model", default=_DEFAULT_MODEL)
    args = p.parse_args()
    label_project(args.project,
                  images_subdir=args.images_subdir,
                  out_subdir=args.out_subdir,
                  model_name=args.model,
                  save_vis=not args.no_vis)
