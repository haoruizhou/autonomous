"""GPS-guided terrain correction for v2 road/grass clouds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

ROAD, GRASS, CONE = 1, 2, 3
GROUND_CLASSES = (ROAD, GRASS)


def clean_gps_path(points: np.ndarray, *, min_step_m: float = 0.5, z_window: int = 31) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 GPS path, got {pts.shape}")
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return pts.reshape(0, 3)

    keep = [0]
    last = pts[0, :2]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i, :2] - last)) >= min_step_m:
            keep.append(i)
            last = pts[i, :2]
    pts = pts[keep]

    if len(pts) >= 3:
        win = min(int(z_window), len(pts))
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            out = pts.copy()
            out[:, 2] = median_filter(pts[:, 2], size=win, mode="nearest")
            pts = out
    return pts


def terrain_z_at_xy(xy: np.ndarray, gps_path: np.ndarray, *, k: int = 8) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    gps = np.asarray(gps_path, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected Nx2 query xy, got {xy.shape}")
    if gps.ndim != 2 or gps.shape[1] != 3 or len(gps) == 0:
        raise ValueError(f"Expected non-empty Nx3 GPS path, got {gps.shape}")

    tree = cKDTree(gps[:, :2])
    kk = max(1, min(int(k), len(gps)))
    dist, idx = tree.query(xy, k=kk)
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if kk == 1:
        return gps[idx, 2].astype(np.float64)

    exact = dist[:, 0] < 1e-9
    weights = 1.0 / np.maximum(dist, 1e-3) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    z = np.sum(gps[idx, 2] * weights, axis=1)
    z[exact] = gps[idx[exact, 0], 2]
    return z.astype(np.float64)


def _ground_mask(cls: np.ndarray) -> np.ndarray:
    return np.isin(cls, GROUND_CLASSES)


def correct_cloud_z(
    xyz: np.ndarray,
    cls: np.ndarray,
    gps_path: np.ndarray,
    *,
    residual_m: float = 0.5,
    k: int = 8,
) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    labels = np.asarray(cls)
    mask = _ground_mask(labels)
    if not mask.any():
        return out.astype(np.float32)

    terrain = terrain_z_at_xy(out[mask, :2], gps_path, k=k)
    residual = out[mask, 2] - terrain
    out[mask, 2] = terrain + np.clip(residual, -float(residual_m), float(residual_m))
    return out.astype(np.float32)


def limit_local_z_jumps(
    xyz: np.ndarray,
    cls: np.ndarray,
    *,
    cell_m: float = 1.0,
    max_jump_m: float = 0.25,
    iterations: int = 3,
) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    labels = np.asarray(cls)
    ground_idx = np.flatnonzero(_ground_mask(labels))
    if len(ground_idx) < 2:
        return out.astype(np.float32)

    tree = cKDTree(out[ground_idx, :2])
    pairs = np.asarray(list(tree.query_pairs(float(cell_m))), dtype=np.int64)
    if pairs.size == 0:
        return out.astype(np.float32)

    max_jump = float(max_jump_m)
    max_iters = max(int(iterations), 1) * 4
    for _ in range(max_iters):
        changed = False
        for a_local, b_local in pairs:
            a = ground_idx[int(a_local)]
            b = ground_idx[int(b_local)]
            dz = out[b, 2] - out[a, 2]
            if abs(dz) <= max_jump:
                continue
            excess = (abs(dz) - max_jump) / 2.0
            sign = 1.0 if dz > 0 else -1.0
            out[a, 2] += sign * excess
            out[b, 2] -= sign * excess
            changed = True
        if not changed:
            break
    return out.astype(np.float32)


def load_camera_gps_path(project_dir: Path, undist_subdir: str = "undistorted") -> np.ndarray:
    rec_path = Path(project_dir) / undist_subdir / "reconstruction.json"
    recs = json.loads(rec_path.read_text())
    pts = []
    for rec in recs:
        shots = sorted(rec.get("shots", {}).values(), key=lambda s: float(s.get("capture_time", 0.0)))
        pts.extend([shot["gps_position"] for shot in shots if "gps_position" in shot])
    if not pts:
        raise RuntimeError(f"No gps_position entries found in {rec_path}")
    return np.asarray(pts, dtype=np.float64)


def _z_stats(xyz: np.ndarray, cls: np.ndarray) -> dict:
    mask = _ground_mask(cls)
    z = np.asarray(xyz)[mask, 2]
    return {
        "min": float(z.min()),
        "max": float(z.max()),
        "span": float(z.max() - z.min()),
        "median": float(np.median(z)),
        "count": int(len(z)),
    }


def correct_project_cloud(
    project_dir: Path,
    *,
    cloud_name: str = "cloud.npz",
    out_name: str = "cloud_terrain.npz",
    residual_m: float = 0.5,
    max_jump_m: float = 0.25,
    cell_m: float = 1.0,
    iterations: int = 3,
    gps_min_step_m: float = 0.5,
    gps_z_window: int = 31,
) -> Path:
    project_dir = Path(project_dir)
    cloud = np.load(project_dir / cloud_name)
    xyz = np.asarray(cloud["xyz"], dtype=np.float64)
    cls = np.asarray(cloud["cls"])

    raw_gps = load_camera_gps_path(project_dir)
    gps = clean_gps_path(raw_gps, min_step_m=gps_min_step_m, z_window=gps_z_window)
    corrected = correct_cloud_z(xyz, cls, gps, residual_m=residual_m)
    corrected = limit_local_z_jumps(
        corrected,
        cls,
        cell_m=cell_m,
        max_jump_m=max_jump_m,
        iterations=iterations,
    )

    out_path = project_dir / out_name
    arrays = {key: cloud[key] for key in cloud.files}
    arrays["xyz"] = corrected.astype(np.float32)
    np.savez_compressed(out_path, **arrays)

    summary = {
        "input_cloud": cloud_name,
        "output_cloud": out_name,
        "raw_gps_points": int(len(raw_gps)),
        "clean_gps_points": int(len(gps)),
        "gps_z": {
            "min": float(gps[:, 2].min()),
            "max": float(gps[:, 2].max()),
            "span": float(gps[:, 2].max() - gps[:, 2].min()),
        },
        "residual_m": float(residual_m),
        "max_jump_m": float(max_jump_m),
        "before_ground_z": _z_stats(xyz, cls),
        "after_ground_z": _z_stats(corrected, cls),
    }
    (project_dir / "terrain_correction.json").write_text(json.dumps(summary, indent=2))
    print(f"  terrain cloud: {out_path}")
    print(
        "  ground Z before: "
        f"{summary['before_ground_z']['min']:.2f}–{summary['before_ground_z']['max']:.2f} "
        f"span={summary['before_ground_z']['span']:.2f}m"
    )
    print(
        "  ground Z after:  "
        f"{summary['after_ground_z']['min']:.2f}–{summary['after_ground_z']['max']:.2f} "
        f"span={summary['after_ground_z']['span']:.2f}m"
    )
    return out_path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Correct v2 cloud Z using cleaned GPS terrain")
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--cloud", default="cloud.npz")
    ap.add_argument("--out", default="cloud_terrain.npz")
    ap.add_argument("--residual", type=float, default=0.5)
    ap.add_argument("--max-jump", type=float, default=0.25)
    ap.add_argument("--cell", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--gps-min-step", type=float, default=0.5)
    ap.add_argument("--gps-z-window", type=int, default=31)
    args = ap.parse_args()

    correct_project_cloud(
        args.project,
        cloud_name=args.cloud,
        out_name=args.out,
        residual_m=args.residual,
        max_jump_m=args.max_jump,
        cell_m=args.cell,
        iterations=args.iterations,
        gps_min_step_m=args.gps_min_step,
        gps_z_window=args.gps_z_window,
    )


if __name__ == "__main__":
    main()
