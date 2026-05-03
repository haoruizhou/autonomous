"""Stage D — fuse SfM poses + per-frame labels + depthmaps → labeled point cloud.

Input layout (per reconstruction component):
    <project>/<undist>/reconstruction.json
    <project>/<undist>/images/<stem>.jpg          (undistorted RGB)
    <project>/<undist>/depthmaps/<stem>.jpg.clean.npz
    <project>/<labels>/<stem>.npy                 (uint8 HxW project class ids)

Output:
    <project>/cloud.npz with arrays:
        xyz (N,3) float32  — world coordinates (OpenSfM ENU)
        rgb (N,3) uint8    — sampled from undistorted image
        cls (N,)  uint8    — project class id (0=other, 1=road, 2=grass, 3=cone, 4=removed)
        src (N,)  int32    — index of source frame (for traceback)

Notes:
    - Pixel coordinates are in DEPTHMAP space (typically 360x640) since depthmap
      resolution < image resolution. Labels are at image resolution and are
      downsampled to depth resolution by nearest-neighbor.
    - Sampling stride keeps the cloud manageable (default 4 px → ~14k pts/frame
      pre-filter, ~50% survive valid-depth filter).
    - Class 0 (other/sky) and 4 (removed) points are dropped — they have no
      role in the final mesh.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

OTHER, ROAD, GRASS, CONE, REMOVED = 0, 1, 2, 3, 4
KEEP_CLASSES = (ROAD, GRASS, CONE)


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
    """OpenSfM perspective camera → 3x3 K for an image of size (W,H)."""
    f_norm = float(cam["focal"])
    f = f_norm * max(W, H)
    return np.array([[f, 0, W / 2.0],
                     [0, f, H / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def assemble_component(
    project_dir: Path,
    undist_subdir: str,
    labels_subdir: str,
    pixel_stride: int = 4,
    max_depth_m: float = 40.0,
) -> dict:
    project_dir = Path(project_dir)
    undist = project_dir / undist_subdir
    img_dir = undist / "images"
    dm_dir = undist / "depthmaps"
    lbl_dir = project_dir / labels_subdir

    rec = json.loads((undist / "reconstruction.json").read_text())[0]
    cameras = rec["cameras"]

    xs, rgbs, cs, ss = [], [], [], []
    n_frames = 0
    for src_idx, (name, shot) in enumerate(rec["shots"].items()):
        stem = Path(name).stem
        dm_path = dm_dir / f"{name}.clean.npz"
        lbl_path = lbl_dir / f"{stem}.npy"
        img_path = img_dir / name
        if not dm_path.exists() or not lbl_path.exists() or not img_path.exists():
            continue

        depth = np.load(dm_path)["depth"]                  # (Hd, Wd) float32
        Hd, Wd = depth.shape
        label_full = np.load(lbl_path)                     # (Hi, Wi) uint8
        img = cv2.imread(str(img_path))                    # (Hi, Wi, 3) BGR
        if img is None:
            continue
        Hi, Wi = img.shape[:2]

        # Resize label and image to depth grid (nearest for label, area for img).
        if (Hi, Wi) != (Hd, Wd):
            label = cv2.resize(label_full, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
            img_d = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
        else:
            label, img_d = label_full, img

        # Sample pixels on a stride grid; keep only valid + interesting classes.
        vs, us = np.mgrid[0:Hd:pixel_stride, 0:Wd:pixel_stride]
        vs = vs.reshape(-1); us = us.reshape(-1)
        d = depth[vs, us]
        cls = label[vs, us]
        mask = (d > 0) & (d < max_depth_m) & np.isin(cls, KEEP_CLASSES)
        if not mask.any():
            continue
        us, vs, d, cls = us[mask], vs[mask], d[mask], cls[mask]

        cam = cameras[shot["camera"]]
        K = _intrinsics(cam, Wd, Hd)
        Kinv = np.linalg.inv(K)

        # Pixel → camera ray (in depthmap intrinsics).
        pix = np.stack([us, vs, np.ones_like(us)], axis=1).astype(np.float64)  # (N,3)
        rays_cam = pix @ Kinv.T                                                # (N,3)
        # OpenSfM perspective convention: depth = z-component along ray ⇒ X_cam = ray * (depth/ray_z).
        # Equivalently: X_cam = depth * ray (since ray.z == 1 for normalized image plane).
        X_cam = rays_cam * d[:, None]

        # Camera→world: X_w = R^T (X_c - t)
        R = _angle_axis_to_R(np.array(shot["rotation"]))
        t = np.array(shot["translation"], dtype=np.float64)
        X_w = (X_cam - t) @ R     # since R^T applied on the right == row-vector form

        bgr = img_d[vs, us]
        rgb = bgr[:, ::-1]        # BGR → RGB

        xs.append(X_w.astype(np.float32))
        rgbs.append(rgb.astype(np.uint8))
        cs.append(cls.astype(np.uint8))
        ss.append(np.full(len(d), src_idx, dtype=np.int32))
        n_frames += 1

    if not xs:
        raise RuntimeError(f"No points produced for {undist}")

    xyz = np.concatenate(xs)
    rgb = np.concatenate(rgbs)
    cls = np.concatenate(cs)
    src = np.concatenate(ss)
    print(f"  {undist.name}: {n_frames} frames → {len(xyz):,} points "
          f"(road={int((cls==ROAD).sum()):,} grass={int((cls==GRASS).sum()):,} "
          f"cone={int((cls==CONE).sum()):,})")
    return {"xyz": xyz, "rgb": rgb, "cls": cls, "src": src}


def assemble_project(
    project_dir: Path,
    components: Iterable[tuple[str, str]] = (
        ("undistorted", "labels"),
        ("undistorted_rec1", "labels_rec1"),
    ),
    pixel_stride: int = 4,
    max_depth_m: float = 40.0,
) -> dict:
    project_dir = Path(project_dir)
    parts = []
    for undist_sub, lbl_sub in components:
        if not (project_dir / undist_sub / "reconstruction.json").exists():
            print(f"  [skip] {undist_sub}: no reconstruction.json")
            continue
        parts.append(assemble_component(
            project_dir, undist_sub, lbl_sub,
            pixel_stride=pixel_stride, max_depth_m=max_depth_m,
        ))

    cloud = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].keys()}
    out = project_dir / "cloud.npz"
    np.savez_compressed(out, **cloud)
    print(f"  Saved {out}: {len(cloud['xyz']):,} points")
    return cloud


def export_ply(project_dir: Path, name: str = "cloud.ply") -> Path:
    """Write a colourised PLY for MeshLab/Blender inspection."""
    project_dir = Path(project_dir)
    cloud = np.load(project_dir / "cloud.npz")
    xyz, rgb, cls = cloud["xyz"], cloud["rgb"], cloud["cls"]

    # Class-tinted color so you can see road vs grass at a glance.
    tint = {
        ROAD:  np.array([180, 180, 180], np.uint8),
        GRASS: np.array([60, 200, 80], np.uint8),
        CONE:  np.array([0, 140, 255], np.uint8),
    }
    show = rgb.copy()
    for c, col in tint.items():
        m = cls == c
        show[m] = (0.5 * show[m] + 0.5 * col).astype(np.uint8)

    out = project_dir / name
    with out.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar class\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b), c in zip(xyz, show, cls):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)} {int(c)}\n")
    print(f"  PLY: {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build labeled point cloud from SfM + labels")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-depth", type=float, default=40.0)
    p.add_argument("--ply", action="store_true", help="Also write cloud.ply")
    args = p.parse_args()
    assemble_project(args.project, pixel_stride=args.stride, max_depth_m=args.max_depth)
    if args.ply:
        export_ply(args.project)
