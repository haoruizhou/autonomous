"""Stage D — fuse SfM poses + per-frame labels + depthmaps → labeled point cloud.

Dense path (default, requires compute_depthmaps):
    <project>/<undist>/reconstruction.json
    <project>/<undist>/images/<stem>.jpg          (undistorted RGB)
    <project>/<undist>/depthmaps/<stem>.jpg.clean.npz
    <project>/<labels>/<stem>.npy                 (uint8 HxW project class ids)

OpenMVS GPU path (when scene_dense.ply exists from DensifyPointCloud):
    <project>/<undist>/openmvs/scene_dense.ply    (dense XYZ+RGB from OpenMVS)
    <project>/<undist>/reconstruction.json        (camera poses for label projection)
    <project>/<labels>/<stem>.npy                 (labels projected via k-NN camera voting)

Sparse fallback (when no depthmaps directory exists):
    <project>/<undist>/reconstruction.json        (SfM bundle points used directly)
    <project>/<undist>/images/<stem>.jpg          (for RGB color)
    <project>/<labels>/<stem>.npy                 (labels projected via camera poses)

Output:
    <project>/cloud.npz with arrays:
        xyz (N,3) float32  — world coordinates (OpenSfM ENU)
        rgb (N,3) uint8    — sampled from undistorted image
        cls (N,)  uint8    — project class id (0=other, 1=road, 2=grass, 3=cone, 4=removed)
        src (N,)  int32    — index of source frame (for traceback)

Notes:
    Dense: pixel stride keeps cloud manageable (~14k pts/frame pre-filter).
    OpenMVS: projects label masks onto the dense PLY via k-NN camera voting (~600MB RAM peak).
    Sparse: uses ~1-2M bundle-adjustment points; labels assigned by majority vote
    across all observing cameras. Lower density but no depthmap dependency.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

OTHER, ROAD, GRASS, CONE, REMOVED = 0, 1, 2, 3, 4
KEEP_CLASSES = (ROAD, GRASS, CONE)
_N_CLASSES = 5


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


def _read_ply_xyz_rgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ and RGB from a binary or ASCII PLY file.

    Handles OpenMVS color property names (red/green/blue or diffuse_red/…).
    Returns xyz float32 (N,3) and rgb uint8 (N,3).
    """
    with open(ply_path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break

        n_verts = 0
        is_binary = False
        props: list[tuple[str, str]] = []
        for line in header:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif "binary_little_endian" in line:
                is_binary = True
            elif line.startswith("property") and not line.startswith("property list"):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)

        _type_map = {
            "float": "f4", "float32": "f4",
            "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1",
            "char": "i1", "int8": "i1",
            "short": "i2", "int16": "i2",
            "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4",
            "uint": "u4", "uint32": "u4",
        }
        dt = np.dtype([(name, _type_map.get(tp, "f4")) for tp, name in props])
        raw = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt) if is_binary else \
              np.array([f.readline().split() for _ in range(n_verts)], dtype=dt)

    names = {name for _, name in props}
    x = raw["x"].astype(np.float32)
    y = raw["y"].astype(np.float32)
    z = raw["z"].astype(np.float32)
    xyz = np.stack([x, y, z], axis=1)

    r_name = "red" if "red" in names else "diffuse_red"
    g_name = "green" if "green" in names else "diffuse_green"
    b_name = "blue" if "blue" in names else "diffuse_blue"
    rgb = np.stack([raw[r_name], raw[g_name], raw[b_name]], axis=1).astype(np.uint8)

    valid = np.isfinite(xyz).all(axis=1)
    if not valid.all():
        n_bad = int((~valid).sum())
        print(f"    dropping {n_bad:,} NaN/Inf points from PLY")
        xyz, rgb = xyz[valid], rgb[valid]

    return xyz, rgb


def _assemble_component_openmvs(
    project_dir: Path,
    undist_subdir: str,
    labels_subdir: str,
    max_depth_m: float = 40.0,
    k_cameras: int = 6,
) -> dict:
    """Label an OpenMVS dense PLY by projecting each point onto its k nearest cameras.

    For each dense point we find the k spatially nearest cameras, project the point
    into each camera's label map, and assign the majority-vote label. RAM peak ~600 MB
    for a 5M-point cloud with 2400 cameras.
    """
    from scipy.spatial import KDTree

    undist = project_dir / undist_subdir
    ply_path = undist / "openmvs" / "scene_dense.ply"
    lbl_dir = project_dir / labels_subdir

    print(f"  {undist_subdir}: loading OpenMVS dense PLY …")
    xyz, rgb_ply = _read_ply_xyz_rgb(ply_path)
    print(f"    {len(xyz):,} points loaded")

    recs = json.loads((undist / "reconstruction.json").read_text())

    shot_list: list[tuple] = []
    cam_origins: list[np.ndarray] = []
    for rec in recs:
        cameras = rec["cameras"]
        for sname, shot in rec["shots"].items():
            lp = lbl_dir / f"{Path(sname).stem}.npy"
            if not lp.exists():
                continue
            R = _angle_axis_to_R(np.array(shot["rotation"]))
            t = np.array(shot["translation"], dtype=np.float64)
            origin = (-R.T @ t).astype(np.float32)
            shot_list.append((R, t, cameras[shot["camera"]], lp))
            cam_origins.append(origin)

    # Filter PLY points to camera bounding box + generous margin.
    # OpenMVS DensifyPointCloud produces extreme coordinate outliers (finite but
    # ~1e33) from degenerate triangulations. These pass np.isfinite() but land
    # nowhere near the scene, causing k-NN to assign wrong cameras to 99%+ of points.
    if cam_origins:
        origins_arr = np.array(cam_origins, dtype=np.float64)
        margin = 100.0  # metres beyond camera bbox
        lo = origins_arr.min(axis=0) - margin
        hi = origins_arr.max(axis=0) + margin
        in_bbox = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        n_before = len(xyz)
        xyz, rgb_ply = xyz[in_bbox], rgb_ply[in_bbox]
        n_removed = n_before - len(xyz)
        if n_removed:
            print(f"    removed {n_removed:,} out-of-bbox outlier points ({len(xyz):,} remain)")

    if not shot_list:
        raise RuntimeError(f"No labeled shots found for {undist_subdir}")

    N = len(xyz)
    n_cams = len(shot_list)
    tree = KDTree(np.array(cam_origins, dtype=np.float64))

    # For each point, query its k nearest cameras. Sort by camera so each
    # label file is loaded exactly once.
    k = min(k_cameras, n_cams)
    _, cam_idx_arr = tree.query(xyz.astype(np.float64), k=k)  # (N, k)
    if cam_idx_arr.ndim == 1:
        cam_idx_arr = cam_idx_arr[:, None]

    # Flatten to (N*k,) pairs, sort by camera index
    all_pt_idx = np.repeat(np.arange(N, dtype=np.int32), k)
    all_cam_idx = cam_idx_arr.ravel().astype(np.int32)
    order = np.argsort(all_cam_idx, kind="stable")
    sorted_pts = all_pt_idx[order]
    sorted_cams = all_cam_idx[order]

    boundaries = np.where(np.diff(sorted_cams))[0] + 1
    pt_groups = np.split(sorted_pts, boundaries)
    unique_cams = sorted_cams[np.concatenate([[0], boundaries])]

    # vote_counts[i, c] = number of cameras that labelled point i as class c
    vote_counts = np.zeros((N, _N_CLASSES), dtype=np.uint16)

    for ci, pt_indices in zip(unique_cams.tolist(), pt_groups):
        R, t, cam, lp = shot_list[ci]
        lbl = np.load(lp)
        H, W = lbl.shape
        f = float(cam["focal"]) * max(W, H)

        pts = xyz[pt_indices].astype(np.float64)
        X_cam = pts @ R.T + t
        valid_depth = (X_cam[:, 2] > 0) & (X_cam[:, 2] < max_depth_m)
        u = np.round(X_cam[:, 0] / X_cam[:, 2] * f + W / 2).astype(np.int32)
        v = np.round(X_cam[:, 1] / X_cam[:, 2] * f + H / 2).astype(np.int32)
        valid = valid_depth & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not valid.any():
            continue
        labels = lbl[v[valid], u[valid]].astype(np.int32)
        np.add.at(vote_counts, (pt_indices[valid], labels), 1)

    labeled = vote_counts.sum(axis=1) > 0
    cls_all = vote_counts[labeled].argmax(axis=1).astype(np.uint8)
    keep = np.isin(cls_all, list(KEEP_CLASSES))

    xyz_out = xyz[labeled][keep]
    rgb_out = rgb_ply[labeled][keep]
    cls_out = cls_all[keep]
    src_out = np.zeros(len(xyz_out), dtype=np.int32)

    print(f"  {undist_subdir} (openmvs): {len(xyz_out):,} points "
          f"(road={int((cls_out == ROAD).sum()):,} "
          f"grass={int((cls_out == GRASS).sum()):,} "
          f"cone={int((cls_out == CONE).sum()):,})")
    return {"xyz": xyz_out, "rgb": rgb_out, "cls": cls_out, "src": src_out}


def _assemble_component_sparse(
    project_dir: Path,
    undist_subdir: str,
    labels_subdir: str,
    max_depth_m: float = 40.0,
) -> dict:
    """Sparse fallback: label bundle-adjustment points via camera projection."""
    undist = project_dir / undist_subdir
    img_dir = undist / "images"
    lbl_dir = project_dir / labels_subdir

    recs = json.loads((undist / "reconstruction.json").read_text())
    all_xyz, all_rgb, all_cls, all_src = [], [], [], []

    for rec_idx, rec in enumerate(recs):
        cameras = rec["cameras"]
        shots = rec["shots"]
        points = rec.get("points", {})
        if not points:
            continue

        # Pre-build shot lookup: name → (R, t, cam_dict, label_path, img_path)
        shot_info: dict[str, tuple] = {}
        for sname, shot in shots.items():
            stem = Path(sname).stem
            lp = lbl_dir / f"{stem}.npy"
            ip = img_dir / sname
            if not lp.exists():
                continue
            R = _angle_axis_to_R(np.array(shot["rotation"]))
            t = np.array(shot["translation"], dtype=np.float64)
            shot_info[sname] = (R, t, cameras[shot["camera"]], lp, ip)

        lbl_cache: dict = {}
        img_cache: dict = {}

        # --- Vectorized sparse path ---
        # Invert the index: build per-shot point lists
        shot_to_points: dict = defaultdict(list)  # shot_name -> list of pt_idx
        point_xyz: dict = {}        # pt_idx -> xyz float32
        point_color: dict = {}      # pt_idx -> rgb uint8 (fallback from bundle point)

        for pt_idx, (pt_id, pt) in enumerate(points.items()):
            xyz = np.array(pt["coordinates"], dtype=np.float32)
            point_xyz[pt_idx] = xyz
            point_color[pt_idx] = np.array(pt.get("color", [128, 128, 128]), dtype=np.uint8)
            for sname in pt.get("observations", {}):
                if sname in shot_info:
                    shot_to_points[sname].append(pt_idx)

        # Accumulators per point
        point_label_votes: dict = defaultdict(list)   # pt_idx -> list of label ints
        point_img_color: dict = {}                     # pt_idx -> rgb from image (first valid)

        # Per-shot batch projection
        for sname, pt_indices in shot_to_points.items():
            R, t, cam, lp, ip = shot_info[sname]

            # Load label map (cached)
            if lp not in lbl_cache:
                lbl_cache[lp] = np.load(lp)
            lbl = lbl_cache[lp]
            H, W = lbl.shape
            f = float(cam["focal"]) * max(W, H)

            # Stack all point xyz for this shot: (N, 3)
            xyzs = np.stack([point_xyz[i] for i in pt_indices], axis=0).astype(np.float64)

            # Batch project: X_cam = xyzs @ R.T + t  (shape: N, 3)
            X_cam = xyzs @ R.T + t

            # Filter by depth
            valid_depth = (X_cam[:, 2] > 0) & (X_cam[:, 2] < max_depth_m)

            # Batch pixel coords
            u = np.round(X_cam[:, 0] / X_cam[:, 2] * f + W / 2).astype(np.int32)
            v = np.round(X_cam[:, 1] / X_cam[:, 2] * f + H / 2).astype(np.int32)

            # Filter in-bounds
            valid_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            valid = valid_depth & valid_bounds

            if not valid.any():
                continue

            pt_indices_arr = np.array(pt_indices)
            valid_pt_indices = pt_indices_arr[valid]
            valid_u = u[valid]
            valid_v = v[valid]

            # Vectorized label lookup
            labels = lbl[valid_v, valid_u]
            for pt_idx, label in zip(valid_pt_indices.tolist(), labels.tolist()):
                point_label_votes[pt_idx].append(int(label))

            # Vectorized color lookup (load image once per shot)
            if ip.exists():
                if ip not in img_cache:
                    img_cache[ip] = cv2.imread(str(ip))
                img = img_cache[ip]
                if img is not None:
                    Hi, Wi = img.shape[:2]
                    # Re-project with image dimensions (may differ from label map)
                    fi = float(cam["focal"]) * max(Wi, Hi)
                    ui = np.round(X_cam[:, 0] / X_cam[:, 2] * fi + Wi / 2).astype(np.int32)
                    vi = np.round(X_cam[:, 1] / X_cam[:, 2] * fi + Hi / 2).astype(np.int32)
                    valid_img_bounds = (ui >= 0) & (ui < Wi) & (vi >= 0) & (vi < Hi)
                    valid_img = valid_depth & valid_img_bounds
                    if valid_img.any():
                        valid_img_pts = pt_indices_arr[valid_img]
                        colors = img[vi[valid_img], ui[valid_img], ::-1]  # BGR → RGB
                        for pt_idx, color in zip(valid_img_pts.tolist(), colors):
                            if pt_idx not in point_img_color:
                                point_img_color[pt_idx] = color

        # Majority vote and filter
        for pt_idx in range(len(points)):
            if pt_idx not in point_label_votes or not point_label_votes[pt_idx]:
                continue
            cls = Counter(point_label_votes[pt_idx]).most_common(1)[0][0]
            if cls not in KEEP_CLASSES:
                continue
            all_xyz.append(point_xyz[pt_idx])
            # Prefer image-sampled color, fall back to bundle-adjustment color
            all_rgb.append(point_img_color.get(pt_idx, point_color[pt_idx]))
            all_cls.append(cls)
            all_src.append(rec_idx)

        # Free caches between reconstruction components
        lbl_cache.clear()
        img_cache.clear()

    if not all_xyz:
        raise RuntimeError(f"No sparse points produced for {undist_subdir}")

    xyz = np.stack(all_xyz)
    rgb = np.stack(all_rgb).astype(np.uint8)
    cls = np.array(all_cls, dtype=np.uint8)
    src = np.array(all_src, dtype=np.int32)
    print(f"  {undist_subdir} (sparse): {len(xyz):,} points "
          f"(road={int((cls==ROAD).sum()):,} grass={int((cls==GRASS).sum()):,} "
          f"cone={int((cls==CONE).sum()):,})")
    return {"xyz": xyz, "rgb": rgb, "cls": cls, "src": src}


def _load_frame(args):
    """Load depth, label, and image for a single frame. Returns None if any file is missing."""
    dm_path, lbl_path, img_path = args
    if not dm_path.exists() or not lbl_path.exists() or not img_path.exists():
        return None
    depth = np.load(dm_path)["depth"]
    label_full = np.load(lbl_path)
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return depth, label_full, img


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

    # OpenMVS GPU dense PLY takes priority over CPU depthmaps when present.
    openmvs_ply = undist / "openmvs" / "scene_dense.ply"
    if openmvs_ply.exists():
        print(f"  {undist_subdir}: OpenMVS dense PLY found — using GPU dense path")
        return _assemble_component_openmvs(
            project_dir, undist_subdir, labels_subdir, max_depth_m=max_depth_m
        )

    # Auto-fallback: if no depthmaps directory, use sparse SfM points.
    if not dm_dir.is_dir() or not any(dm_dir.glob("*.npz")):
        print(f"  {undist_subdir}: no depthmaps found — using sparse SfM fallback")
        return _assemble_component_sparse(
            project_dir, undist_subdir, labels_subdir, max_depth_m=max_depth_m
        )

    rec = json.loads((undist / "reconstruction.json").read_text())[0]
    cameras = rec["cameras"]

    # Prefetch all frames in parallel (I/O bound)
    shot_list = list(rec["shots"].items())
    frame_args = [
        (
            dm_dir / f"{name}.clean.npz",
            lbl_dir / f"{Path(name).stem}.npy",
            img_dir / name,
        )
        for name, shot in shot_list
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        frame_data_list = list(executor.map(_load_frame, frame_args))

    xs, rgbs, cs, ss = [], [], [], []
    n_frames = 0
    for src_idx, ((name, shot), frame_data) in enumerate(zip(shot_list, frame_data_list)):
        if frame_data is None:
            continue
        depth, label_full, img = frame_data

        Hd, Wd = depth.shape
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
