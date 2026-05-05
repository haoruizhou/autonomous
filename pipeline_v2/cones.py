"""Stage E — detect traffic cones and anchor them in the SfM world frame.

Runs YOLO-World on undistorted images, backprojects each detection through the
OpenSfM clean depthmap, clusters repeated observations, and writes:
    <project>/cones.json
    <project>/cone_detections.json
    <project>/cloud_with_cones.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree

OTHER, ROAD, GRASS, CONE, REMOVED = 0, 1, 2, 3, 4


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


def _load_model(model_name: str, conf: float):
    from ultralytics import YOLO
    model = YOLO(model_name)
    if hasattr(model, "set_classes"):
        model.set_classes(["traffic cone", "orange traffic cone", "cone"])
    print(f"  Loading cone detector {model_name} (conf ≥ {conf:.2f})")
    return model


def _detections(model, img_bgr: np.ndarray, conf: float) -> list[dict]:
    res = model(img_bgr, verbose=False, conf=conf)[0]
    if res.boxes is None:
        return []
    names = getattr(res, "names", {}) or {}
    boxes = []
    for xyxy, score, cls_id in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
        res.boxes.cls.cpu().numpy().astype(int),
    ):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "bbox": [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))],
            "cx": int(round((x1 + x2) / 2.0)),
            "cy": int(round(y2)),
            "conf": float(score),
            "cls": int(cls_id),
            "label": str(names.get(int(cls_id), "traffic cone")),
        })
    return boxes


def _detections_batch(model, imgs: list[np.ndarray], conf: float) -> list[list[dict]]:
    """Run YOLO inference on a batch of images at once.

    Returns a list of detection lists, one per image, in the same format as
    _detections.  Falls back to one-by-one if batch inference raises an error.
    """
    try:
        results = model(imgs, verbose=False, conf=conf)
        out = []
        for res in results:
            if res.boxes is None:
                out.append([])
                continue
            names = getattr(res, "names", {}) or {}
            boxes = []
            for xyxy, score, cls_id in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
            ):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                boxes.append({
                    "xyxy": [x1, y1, x2, y2],
                    "bbox": [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))],
                    "cx": int(round((x1 + x2) / 2.0)),
                    "cy": int(round(y2)),
                    "conf": float(score),
                    "cls": int(cls_id),
                    "label": str(names.get(int(cls_id), "traffic cone")),
                })
            out.append(boxes)
        return out
    except Exception:
        # Fall back to single-image inference
        return [_detections(model, img, conf) for img in imgs]


def _median_depth(depth: np.ndarray, x: int, y: int, radius: int = 3) -> float:
    H, W = depth.shape
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    vals = depth[y0:y1, x0:x1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return 0.0
    return float(np.median(vals))


def _backproject(px: int, py: int, depth_m: float, Kinv: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pix = np.array([px, py, 1.0], dtype=np.float64)
    ray = Kinv @ pix
    X_cam = ray * depth_m
    return (X_cam - t) @ R


def _cluster_points(records: list[dict], radius_m: float, min_observations: int) -> list[dict]:
    clusters: list[dict] = []
    centers_2d: list[list[float]] = []  # parallel to clusters, xy only
    tree: KDTree | None = None

    for rec in sorted(records, key=lambda r: -float(r["conf"])):
        p = np.array(rec["xyz"], dtype=np.float64)
        best_i, best_d = -1, float("inf")
        if tree is not None:
            best_d, best_i = tree.query(p[:2], k=1)
            best_d = float(best_d)
            best_i = int(best_i)
        if best_i >= 0 and best_d <= radius_m:
            cl = clusters[best_i]
            cl["points"].append(p)
            cl["detections"].append(rec)
            cl["center"] = np.median(np.stack(cl["points"]), axis=0).tolist()
            centers_2d[best_i] = cl["center"][:2]
            tree = KDTree(centers_2d)
        else:
            clusters.append({"center": p.tolist(), "points": [p], "detections": [rec]})
            centers_2d.append(p[:2].tolist())
            tree = KDTree(centers_2d)

    out = []
    for i, cl in enumerate(clusters):
        if len(cl["detections"]) < min_observations:
            continue
        confs = [float(d["conf"]) for d in cl["detections"]]
        frames = sorted({d["image"] for d in cl["detections"]})
        out.append({
            "id": len(out),
            "xyz": [round(float(v), 4) for v in cl["center"]],
            "n_observations": len(cl["detections"]),
            "mean_conf": round(float(np.mean(confs)), 4),
            "max_conf": round(float(np.max(confs)), 4),
            "frames": frames,
        })
    return out


def detect_component(
    project_dir: Path,
    undist_subdir: str = "undistorted",
    model_name: str = "yolov8s-world.pt",
    conf: float = 0.12,
    max_depth_m: float = 45.0,
) -> list[dict]:
    project_dir = Path(project_dir)
    undist = project_dir / undist_subdir
    img_dir = undist / "images"
    dm_dir = undist / "depthmaps"
    rec = json.loads((undist / "reconstruction.json").read_text())[0]
    cameras = rec["cameras"]
    model = _load_model(model_name, conf)

    BATCH_SIZE = 8

    # Collect all valid (src_idx, name, shot, img_path, dm_path) tuples first.
    shot_list = []
    for src_idx, (name, shot) in enumerate(rec["shots"].items()):
        img_path = img_dir / name
        dm_path = dm_dir / f"{name}.clean.npz"
        if img_path.exists() and dm_path.exists():
            shot_list.append((src_idx, name, shot, img_path, dm_path))

    records = []
    # Process in batches: load images for the batch, run batch inference, then
    # load depth maps only for shots that had detections.
    for batch_start in range(0, len(shot_list), BATCH_SIZE):
        batch = shot_list[batch_start: batch_start + BATCH_SIZE]

        # Load images for the batch.
        batch_imgs: list[np.ndarray] = []
        valid_batch: list[tuple] = []  # (src_idx, name, shot, dm_path, img_shape)
        for src_idx, name, shot, img_path, dm_path in batch:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            batch_imgs.append(img)
            valid_batch.append((src_idx, name, shot, dm_path, img.shape[:2]))

        if not batch_imgs:
            continue

        # Batch YOLO inference.
        batch_dets = _detections_batch(model, batch_imgs, conf)

        for (src_idx, name, shot, dm_path, (Hi, Wi)), dets in zip(valid_batch, batch_dets):
            if not dets:
                continue
            # Load depth map only when there are detections to process.
            depth = np.load(dm_path)["depth"]
            Hd, Wd = depth.shape
            sx, sy = Wd / Wi, Hd / Hi
            cam = cameras[shot["camera"]]
            Kinv = np.linalg.inv(_intrinsics(cam, Wd, Hd))
            R = _angle_axis_to_R(np.array(shot["rotation"], dtype=np.float64))
            t = np.array(shot["translation"], dtype=np.float64)

            for det in dets:
                px = int(round(det["cx"] * sx))
                py = int(round(det["cy"] * sy))
                if px < 0 or py < 0 or px >= Wd or py >= Hd:
                    continue
                d = _median_depth(depth, px, py)
                if d <= 0 or d > max_depth_m:
                    continue
                xyz = _backproject(px, py, d, Kinv, R, t)
                records.append({
                    "component": undist_subdir,
                    "image": name,
                    "src": int(src_idx),
                    "conf": round(float(det["conf"]), 4),
                    "label": det["label"],
                    "bbox": det["bbox"],
                    "px_depth": [int(px), int(py)],
                    "depth_m": round(float(d), 4),
                    "xyz": [float(v) for v in xyz],
                })
    print(f"  {undist_subdir}: {len(records)} cone detections with valid depth")
    return records


def detect_project(
    project_dir: Path,
    components: tuple[str, ...] = ("undistorted", "undistorted_rec1"),
    model_name: str = "yolov8s-world.pt",
    conf: float = 0.12,
    cluster_radius_m: float = 0.75,
    min_observations: int = 2,
    max_depth_m: float = 45.0,
) -> dict:
    project_dir = Path(project_dir)
    detections = []
    for undist_subdir in components:
        if not (project_dir / undist_subdir / "reconstruction.json").exists():
            print(f"  [skip] {undist_subdir}: no reconstruction.json")
            continue
        detections.extend(detect_component(
            project_dir, undist_subdir, model_name=model_name,
            conf=conf, max_depth_m=max_depth_m,
        ))

    cones = _cluster_points(detections, cluster_radius_m, min_observations)
    summary = {
        "n_detections": len(detections),
        "n_cones": len(cones),
        "cluster_radius_m": cluster_radius_m,
        "min_observations": min_observations,
        "cones": cones,
    }
    (project_dir / "cone_detections.json").write_text(json.dumps(detections, indent=2))
    (project_dir / "cones.json").write_text(json.dumps(summary, indent=2))
    print(f"  Clustered {len(detections)} detections → {len(cones)} cones")
    return summary


def stamp_cloud(project_dir: Path, in_name: str = "cloud.npz", out_name: str = "cloud_with_cones.npz", radius_m: float = 0.35) -> Path:
    project_dir = Path(project_dir)
    cloud = np.load(project_dir / in_name)
    xyz = cloud["xyz"]
    cls = cloud["cls"].copy()
    cones = json.loads((project_dir / "cones.json").read_text()).get("cones", [])
    for cone in cones:
        p = np.array(cone["xyz"], dtype=np.float32)
        dxy = np.linalg.norm(xyz[:, :2] - p[:2], axis=1)
        cls[dxy <= radius_m] = CONE
    out = project_dir / out_name
    data = {k: cloud[k] for k in cloud.files if k != "cls"}
    data["cls"] = cls
    np.savez_compressed(out, **data)
    print(f"  Stamped {len(cones)} cones into {out}: cone points={int((cls == CONE).sum()):,}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Detect and anchor cones in an OpenSfM project")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model", default="yolov8s-world.pt")
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--cluster-radius", type=float, default=0.75)
    p.add_argument("--min-observations", type=int, default=2)
    p.add_argument("--max-depth", type=float, default=45.0)
    p.add_argument("--stamp-cloud", action="store_true")
    p.add_argument("--cloud-in", default="cloud.npz")
    p.add_argument("--cloud-out", default="cloud_with_cones.npz")
    args = p.parse_args()
    detect_project(
        args.project,
        model_name=args.model,
        conf=args.conf,
        cluster_radius_m=args.cluster_radius,
        min_observations=args.min_observations,
        max_depth_m=args.max_depth,
    )
    if args.stamp_cloud:
        stamp_cloud(args.project, in_name=args.cloud_in, out_name=args.cloud_out)
