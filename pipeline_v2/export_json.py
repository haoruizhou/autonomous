"""Export measured v2 reconstruction as lightweight JSON for browser simulators."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation
from scipy.signal import savgol_filter

ROAD, GRASS, CONE = 1, 2, 3


def _make_grid(xyz: np.ndarray, cell_m: float, margin_m: float) -> dict:
    x0 = float(xyz[:, 0].min() - margin_m)
    y0 = float(xyz[:, 1].min() - margin_m)
    W = int(np.ceil((xyz[:, 0].max() + margin_m - x0) / cell_m)) + 1
    H = int(np.ceil((xyz[:, 1].max() + margin_m - y0) / cell_m)) + 1
    return {"x0": x0, "y0": y0, "width": W, "height": H, "cell_m": cell_m}


def _accumulate(grid: dict, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor((xyz[:, 0] - grid["x0"]) / grid["cell_m"]).astype(np.int32)
    iy = np.floor((xyz[:, 1] - grid["y0"]) / grid["cell_m"]).astype(np.int32)
    valid = (ix >= 0) & (iy >= 0) & (ix < grid["width"]) & (iy < grid["height"])
    ix, iy = ix[valid], iy[valid]
    z = xyz[valid, 2]
    count = np.zeros((grid["height"], grid["width"]), dtype=np.int32)
    zsum = np.zeros((grid["height"], grid["width"]), dtype=np.float64)
    np.add.at(count, (iy, ix), 1)
    np.add.at(zsum, (iy, ix), z)
    return count, zsum


def _cells(mask: np.ndarray, zgrid: np.ndarray, max_cells: int | None = None) -> list[list[float]]:
    idx = np.argwhere(mask)
    if max_cells and len(idx) > max_cells:
        step = int(np.ceil(len(idx) / max_cells))
        idx = idx[::step]
    return [[int(i), int(j), round(float(zgrid[j, i]), 3)] for j, i in idx]


def _camera_centerline(project_dir: Path) -> list[list[float]]:
    rec_paths = [project_dir / "undistorted" / "reconstruction.json", project_dir / "undistorted_rec1" / "reconstruction.json"]
    pts = []
    for rec_path in rec_paths:
        if not rec_path.exists():
            continue
        for rec in json.loads(rec_path.read_text()):
            shots = sorted(
                rec["shots"].values(),
                key=lambda s: float(s.get("capture_time", 0.0)),
            )
            pts.extend([s["gps_position"] for s in shots if "gps_position" in s])
    if len(pts) < 3:
        return []
    arr = np.array(pts, dtype=np.float64)
    keep = np.ones(len(arr), dtype=bool)
    step = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    keep[1:] = step < 8.0
    arr = arr[keep]
    if len(arr) >= 9:
        win = min(31, len(arr) - (1 - len(arr) % 2))
        if win >= 5 and win % 2 == 1:
            arr[:, 0] = savgol_filter(arr[:, 0], win, 3)
            arr[:, 1] = savgol_filter(arr[:, 1], win, 3)
    return [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for x, y, z in arr]


def _centerline_mask(grid: dict, centerline: list[list[float]], radius_m: float) -> np.ndarray:
    mask = np.zeros((grid["height"], grid["width"]), dtype=bool)
    if not centerline:
        return mask
    radius_cells = max(1, int(np.ceil(radius_m / grid["cell_m"])))
    # Mark each centerline point as a single cell in a sparse boolean grid
    for x, y, _ in centerline:
        ci = int(np.floor((x - grid["x0"]) / grid["cell_m"]))
        cj = int(np.floor((y - grid["y0"]) / grid["cell_m"]))
        if 0 <= ci < grid["width"] and 0 <= cj < grid["height"]:
            mask[cj, ci] = True
    # Build a disk-shaped structuring element that matches the original radius check
    r = radius_cells
    coords = np.arange(-r, r + 1)
    di, dj = np.meshgrid(coords, coords)
    disk = np.hypot(di, dj) * grid["cell_m"] <= radius_m
    mask = binary_dilation(mask, structure=disk)
    return mask


def _patch_centerline_road_holes(
    road_mask: np.ndarray,
    grass_mask: np.ndarray,
    road_z: np.ndarray,
    grid: dict,
    centerline: list[list[float]],
    radius_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    search_r = max(2, int(np.ceil(2.5 / grid["cell_m"])))
    candidates = _centerline_mask(grid, centerline, radius_m) & ~road_mask & ~grass_mask
    patched = road_mask.copy()
    patched_z = road_z.copy()
    added = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    height_window = max(1, int(np.ceil(1.4 / grid["cell_m"])))

    for j, i in np.argwhere(candidates):
        road_dirs = 0
        grass_dirs = 0
        edge_dirs = 0
        for di, dj in directions:
            seen_road = False
            seen_grass = False
            hit_edge = False
            for step in range(1, search_r + 1):
                ii = i + di * step
                jj = j + dj * step
                if ii < 0 or jj < 0 or ii >= grid["width"] or jj >= grid["height"]:
                    hit_edge = True
                    break
                if road_mask[jj, ii]:
                    seen_road = True
                    break
                if grass_mask[jj, ii]:
                    seen_grass = True
                    break
            road_dirs += int(seen_road)
            grass_dirs += int(seen_grass)
            edge_dirs += int(hit_edge)
        if road_dirs < 7 or grass_dirs > 0 or edge_dirs > 0:
            continue
        j0, j1 = max(0, j - height_window), min(grid["height"], j + height_window + 1)
        i0, i1 = max(0, i - height_window), min(grid["width"], i + height_window + 1)
        nearby_z = road_z[j0:j1, i0:i1][road_mask[j0:j1, i0:i1]]
        if nearby_z.size == 0:
            continue
        patched[j, i] = True
        patched_z[j, i] = float(np.median(nearby_z))
        added += 1

    patched_grass = grass_mask & ~patched
    return patched, patched_grass, patched_z, added


def export_track_json(
    project_dir: Path,
    cloud_name: str = "cloud_with_cones.npz",
    cones_name: str = "cones.json",
    out_name: str = "track.json",
    cell_m: float = 0.35,
    grass_margin_m: float = 12.0,
    road_buffer_m: float = 1.2,
    max_cells_per_class: int | None = None,
) -> Path:
    project_dir = Path(project_dir)
    cloud = np.load(project_dir / cloud_name)
    xyz, cls = cloud["xyz"], cloud["cls"]
    road_xyz = xyz[cls == ROAD]
    grass_xyz = xyz[cls == GRASS]
    all_ground = xyz[(cls == ROAD) | (cls == GRASS)]
    ground_z = float(np.percentile(all_ground[:, 2], 5)) if len(all_ground) else 0.0

    grid = _make_grid(all_ground if len(all_ground) else xyz, cell_m, grass_margin_m)
    road_count, road_zsum = _accumulate(grid, road_xyz)
    grass_count, grass_zsum = _accumulate(grid, grass_xyz)

    road_mask = binary_closing(road_count >= 2, structure=np.ones((3, 3), dtype=bool))

    road_z = np.full((grid["height"], grid["width"]), ground_z, dtype=np.float64)
    np.divide(road_zsum, road_count, out=road_z, where=road_count > 0)
    grass_z = np.full((grid["height"], grid["width"]), ground_z - 0.03, dtype=np.float64)
    np.divide(grass_zsum, grass_count, out=grass_z, where=grass_count > 0)

    centerline_hint = _camera_centerline(project_dir)
    road_buffer = binary_dilation(road_mask, iterations=max(1, int(round(road_buffer_m / cell_m))))
    grass_extent = binary_dilation(road_mask, iterations=max(1, int(round(grass_margin_m / cell_m))))
    grass_mask = grass_extent & ~road_buffer
    road_mask, grass_mask, road_z, patched_road_holes = _patch_centerline_road_holes(
        road_mask,
        grass_mask,
        road_z,
        grid,
        centerline_hint,
    )

    cones_summary = json.loads((project_dir / cones_name).read_text()) if (project_dir / cones_name).exists() else {"cones": []}
    cones = [{
        "id": int(c["id"]),
        "xyz": [round(float(v), 3) for v in c["xyz"]],
        "n_observations": int(c["n_observations"]),
        "mean_conf": float(c["mean_conf"]),
    } for c in cones_summary.get("cones", [])]

    out = {
        "schema": "autocross-track-v1",
        "units": "meters",
        "coordinate_frame": {
            "source": "OpenSfM GPS-anchored ENU-like world",
            "horizontal_axes": ["x", "y"],
            "vertical_axis": "z",
        },
        "grid": grid,
        "road_cells": _cells(road_mask, road_z, max_cells_per_class),
        "grass_cells": _cells(grass_mask, grass_z, max_cells_per_class),
        "cones": cones,
        "centerline_hint": centerline_hint,
        "counts": {
            "road_cells": int(road_mask.sum()),
            "grass_cells": int(grass_mask.sum()),
            "cones": len(cones),
            "road_points": int((cls == ROAD).sum()),
            "grass_points": int((cls == GRASS).sum()),
            "cone_points": int((cls == CONE).sum()),
            "patched_road_holes": patched_road_holes,
        },
    }

    out_path = project_dir / out_name
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"  track JSON: {out_path}")
    print(
        f"  road_cells={out['counts']['road_cells']:,} grass_cells={out['counts']['grass_cells']:,} "
        f"cones={len(cones):,} patched_road_holes={patched_road_holes:,}"
    )
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export measured track geometry for React/WebGL")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--cloud", default="cloud_with_cones.npz")
    p.add_argument("--cones", default="cones.json")
    p.add_argument("--out", default="track.json")
    p.add_argument("--cell", type=float, default=0.35)
    p.add_argument("--grass-margin", type=float, default=12.0)
    p.add_argument("--road-buffer", type=float, default=1.2)
    p.add_argument("--max-cells-per-class", type=int, default=None)
    args = p.parse_args()
    export_track_json(
        args.project,
        cloud_name=args.cloud,
        cones_name=args.cones,
        out_name=args.out,
        cell_m=args.cell,
        grass_margin_m=args.grass_margin,
        road_buffer_m=args.road_buffer,
        max_cells_per_class=args.max_cells_per_class,
    )
