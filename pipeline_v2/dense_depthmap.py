"""Stage D2 — generate per-frame depthmaps from the OpenMVS dense PLY.

For each image that has YOLO cone detections, projects the full dense PLY
into the image to produce a dense depthmap (minimum-depth per pixel).
cones.py then uses these depthmaps to backproject YOLO detections to 3D.

Output: <project>/undistorted/depthmaps/<name>.dense.npz  (depth, mask)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
# PLY loading via Open3D (handles list properties correctly)
# --------------------------------------------------------------------------- #

def load_ply_vertices(ply_path: Path) -> np.ndarray:
    """Load xyz from a binary OpenMVS PLY using Open3D.
    Returns (N, 3) float64 array.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    xyz = np.asarray(pcd.points, dtype=np.float64)
    return xyz


# --------------------------------------------------------------------------- #
# Camera helpers (same convention as cones.py)
# --------------------------------------------------------------------------- #

def _angle_axis_to_R(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _intrinsics(cam: dict, W: int, H: int) -> np.ndarray:
    f = float(cam["focal"]) * max(W, H)
    return np.array([[f, 0, W / 2.0],
                     [0, f, H / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_depthmaps(
    project_dir: Path,
    ply_path: Path,
    undist_subdir: str = "undistorted",
    model_name: str = "yolov8s-world.pt",
    conf: float = 0.12,
    max_depth_m: float = 45.0,
) -> tuple[int, int]:
    """
    1. Load YOLO cone detector.
    2. For each image, run YOLO → find detection bounding boxes.
    3. If detections exist, project the full dense PLY into the image and
       write a dense depthmap (minimum depth per pixel).
    4. cones.py (ply_mode=True) then uses these .dense.npz files.
    """
    project_dir = Path(project_dir)
    ply_path = Path(ply_path)
    undist = project_dir / undist_subdir
    img_dir = undist / "images"
    out_dir = project_dir / "undistorted" / "depthmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reconstruction
    rec = json.loads((undist / "reconstruction.json").read_text())[0]
    cameras = rec["cameras"]
    shots = rec["shots"]

    # Load PLY once
    print(f"  Loading PLY from {ply_path}")
    xyz = load_ply_vertices(ply_path)
    n_pts = len(xyz)
    print(f"  PLY: {n_pts:,} vertices, shape {xyz.shape}")

    # Load YOLO model
    from ultralytics import YOLO
    model = YOLO(str(project_dir / model_name))
    if hasattr(model, 'set_classes'):
        model.set_classes(['traffic cone', 'orange traffic cone', 'cone'])
    print(f"  Cone detector: {model_name} (conf ≥ {conf:.2f})")

    shot_names = sorted(shots.keys())
    total_dets = 0
    frames_with_dets = 0

    for name in shot_names:
        img_path = img_dir / name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        Hi, Wi = img.shape[:2]

        # YOLO inference
        results = model(img, verbose=False, conf=conf)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        # Parse detections
        dets = []
        for xyxy, score, cls_id in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            dets.append({
                "xyxy": [x1, y1, x2, y2],
                "cx": (x1 + x2) / 2.0,
                "cy": y2,
                "conf": float(score),
                "label": results[0].names.get(int(cls_id), "cone"),
            })

        if not dets:
            continue

        frames_with_dets += 1
        total_dets += len(dets)

        # Camera projection matrix
        shot = shots[name]
        cam = cameras[shot["camera"]]
        K = _intrinsics(cam, Wi, Hi)
        R = _angle_axis_to_R(np.array(shot["rotation"], dtype=np.float64))
        t = np.array(shot["translation"], dtype=np.float64)

        # Project full PLY: world → camera → image
        # PLY is in OpenMVS frame: X_cam = X_world @ R.T + t
        # (different transpose convention from sparse camera formula)
        cam_xyz = xyz @ R.T + t           # Nx3 camera coords
        uv_hom = cam_xyz @ K.T             # Nx3 homogeneous coords (3rd col = depth)
        depth = uv_hom[:, 2]               # depth in camera frame
        uv = uv_hom[:, :2] / depth[:, None] # perspective divide → pixel coords

        # Filter: in front of camera, within depth range, in image bounds
        valid = (
            (depth > 0) &
            (depth <= max_depth_m) &
            (uv[:, 0] >= 0) & (uv[:, 0] < Wi) &
            (uv[:, 1] >= 0) & (uv[:, 1] < Hi)
        )

        uv_v = np.round(uv[valid]).astype(int)
        depth_v = depth[valid]

        # Dense depthmap: minimum depth per pixel (nearest surface wins)
        depthmap = np.full((Hi, Wi), np.inf, dtype=np.float32)
        np.minimum.at(depthmap, (uv_v[:, 1], uv_v[:, 0]), depth_v)

        # Confidence: 1 where a PLY point projects to this pixel
        confmap = np.zeros((Hi, Wi), dtype=np.float32)
        confmap[uv_v[:, 1], uv_v[:, 0]] = 1.0

        out_path = out_dir / f"{name}.dense.npz"
        np.savez_compressed(out_path, depth=depthmap, confidence=confmap)
        print(f"    {name}: {len(dets)} dets, {out_path.name} ({np.isfinite(depthmap).sum():,} valid pixels)")

    print(f"  Total: {total_dets} detections in {frames_with_dets} frames")
    return total_dets, frames_with_dets


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--model", default="yolov8s-world.pt")
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--max-depth", type=float, default=45.0)
    p.add_argument("--undist-subdir", default="undistorted")
    args = p.parse_args()

    generate_depthmaps(
        project_dir=args.project,
        ply_path=args.ply,
        model_name=args.model,
        conf=args.conf,
        max_depth_m=args.max_depth,
        undist_subdir=args.undist_subdir,
    )
