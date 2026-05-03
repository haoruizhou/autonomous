"""Diagnostic images to inspect upstream pipeline + exported track JSON.

Saves PNGs into <project>/diag/.

  uv run python -m pipeline_v2.diag --project results/sfm_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROAD, GRASS, CONE = 1, 2, 3


def _grid(xyz: np.ndarray, cell_m: float, margin: float) -> tuple[float, float, int, int]:
    x0 = float(xyz[:, 0].min() - margin)
    y0 = float(xyz[:, 1].min() - margin)
    W = int(np.ceil((xyz[:, 0].max() + margin - x0) / cell_m)) + 1
    H = int(np.ceil((xyz[:, 1].max() + margin - y0) / cell_m)) + 1
    return x0, y0, W, H


def _rasterize_z(xyz: np.ndarray, cell_m: float, x0: float, y0: float, W: int, H: int, agg: str = "median") -> np.ndarray:
    ix = np.floor((xyz[:, 0] - x0) / cell_m).astype(np.int32)
    iy = np.floor((xyz[:, 1] - y0) / cell_m).astype(np.int32)
    valid = (ix >= 0) & (iy >= 0) & (ix < W) & (iy < H)
    ix, iy, z = ix[valid], iy[valid], xyz[valid, 2]
    img = np.full((H, W), np.nan, dtype=np.float32)
    if agg == "median":
        order = np.lexsort((z, iy, ix))
        ix_s, iy_s, z_s = ix[order], iy[order], z[order]
        keys = ix_s.astype(np.int64) * (H + 1) + iy_s.astype(np.int64)
        starts = np.r_[0, np.where(np.diff(keys) != 0)[0] + 1, len(keys)]
        for s, e in zip(starts[:-1], starts[1:]):
            img[iy_s[s], ix_s[s]] = float(np.median(z_s[s:e]))
    return img


def _height_image(xyz: np.ndarray, cell_m: float, label: str, out: Path) -> None:
    if len(xyz) == 0:
        return
    x0, y0, W, H = _grid(xyz, cell_m, 4.0)
    img = _rasterize_z(xyz, cell_m, x0, y0, W, H)
    fig, ax = plt.subplots(figsize=(11, 11))
    finite = np.isfinite(img)
    if finite.any():
        zs = img[finite]
        vmin, vmax = float(np.percentile(zs, 2)), float(np.percentile(zs, 98))
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(img, origin="lower", cmap="terrain", vmin=vmin, vmax=vmax,
                   extent=(x0, x0 + W * cell_m, y0, y0 + H * cell_m))
    ax.set_aspect("equal")
    ax.set_title(f"{label}: median z per {cell_m:g}m cell  (vmin={vmin:.2f} vmax={vmax:.2f}, range={vmax - vmin:.2f}m)")
    plt.colorbar(im, ax=ax, label="z (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


def _z_hist(xyz_by_class: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, xyz in xyz_by_class.items():
        if len(xyz) == 0:
            continue
        ax.hist(xyz[:, 2], bins=120, alpha=0.45, label=f"{label} (n={len(xyz):,})")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("points")
    ax.set_title("z distribution by class — large spread => noisy reconstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


def _topdown_classes(xyz: np.ndarray, cls: np.ndarray, cones: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    rs = xyz[cls == ROAD]
    gs = xyz[cls == GRASS]
    if len(gs):
        ax.scatter(gs[::20, 0], gs[::20, 1], s=0.3, c="#4c9f55", label="grass", alpha=0.4)
    if len(rs):
        ax.scatter(rs[::20, 0], rs[::20, 1], s=0.3, c="#5ba9ff", label="road", alpha=0.5)
    if cones:
        cx = np.array([c["xyz"][0] for c in cones])
        cy = np.array([c["xyz"][1] for c in cones])
        ax.scatter(cx, cy, s=18, c="#ff8c00", edgecolors="black", linewidths=0.4, label="cones")
    ax.set_aspect("equal")
    ax.set_title("Top-down classes (subsampled)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def _track_json_diag(track_path: Path, out: Path) -> None:
    data = json.loads(track_path.read_text())
    grid = data["grid"]
    W, H = grid["width"], grid["height"]
    road_z = np.full((H, W), np.nan, dtype=np.float32)
    grass_z = np.full((H, W), np.nan, dtype=np.float32)
    for i, j, z in data["road_cells"]:
        road_z[j, i] = z
    for i, j, z in data["grass_cells"]:
        grass_z[j, i] = z

    centerline = np.array(data.get("centerline_hint", []), dtype=np.float32)
    cones = data.get("cones", [])

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    extent = (grid["x0"], grid["x0"] + W * grid["cell_m"], grid["y0"], grid["y0"] + H * grid["cell_m"])

    finite_r = np.isfinite(road_z)
    if finite_r.any():
        zs = road_z[finite_r]
        vmin, vmax = float(np.percentile(zs, 2)), float(np.percentile(zs, 98))
        im0 = axes[0].imshow(road_z, origin="lower", cmap="terrain", vmin=vmin, vmax=vmax, extent=extent)
        axes[0].set_title(f"track.json road_cells z  range={vmax - vmin:.2f}m")
        plt.colorbar(im0, ax=axes[0], label="z (m)")
    if len(centerline):
        for ax in axes:
            ax.plot(centerline[:, 0], centerline[:, 1], color="magenta", linewidth=1.0, alpha=0.7, label="centerline")
    if cones:
        cx = np.array([c["xyz"][0] for c in cones])
        cy = np.array([c["xyz"][1] for c in cones])
        for ax in axes:
            ax.scatter(cx, cy, s=12, c="#ff8c00", edgecolors="black", linewidths=0.3, label="cones")

    classes = np.full((H, W), 0, dtype=np.uint8)
    classes[finite_r] = 2
    classes[np.isfinite(grass_z) & ~finite_r] = 1
    cmap = matplotlib.colors.ListedColormap(["#222222", "#3e7b48", "#4097ff"])
    axes[1].imshow(classes, origin="lower", cmap=cmap, extent=extent, vmin=0, vmax=2)
    axes[1].set_title("track.json classes  (blue=road, green=grass)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def _centerline_height_profile(track_path: Path, project_dir: Path, out: Path) -> None:
    data = json.loads(track_path.read_text())
    grid = data["grid"]
    cell = grid["cell_m"]
    W, H = grid["width"], grid["height"]
    road_z = np.full((H, W), np.nan, dtype=np.float32)
    for i, j, z in data["road_cells"]:
        road_z[j, i] = z
    cones = data.get("cones", [])

    cloud = np.load(project_dir / "cloud_with_cones.npz")
    xyz, cls = cloud["xyz"], cloud["cls"]
    road_pts = xyz[cls == ROAD]

    centerline = np.array(data.get("centerline_hint", []), dtype=np.float32)
    if not len(centerline):
        return
    diffs = np.diff(centerline[:, :2], axis=0)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(diffs, axis=1))]

    grid_z_along = []
    for x, y, _ in centerline:
        i = int(np.floor((x - grid["x0"]) / cell))
        j = int(np.floor((y - grid["y0"]) / cell))
        i = np.clip(i, 0, W - 1)
        j = np.clip(j, 0, H - 1)
        z = road_z[j, i]
        if not np.isfinite(z):
            best = np.nan
            for r in (1, 2, 3, 4, 5):
                jj0, jj1 = max(0, j - r), min(H, j + r + 1)
                ii0, ii1 = max(0, i - r), min(W, i + r + 1)
                window = road_z[jj0:jj1, ii0:ii1]
                if np.isfinite(window).any():
                    best = float(np.nanmedian(window))
                    break
            z = best
        grid_z_along.append(z)
    grid_z_along = np.array(grid_z_along, dtype=np.float32)

    cloud_z_along = []
    for x, y, _ in centerline:
        m = (np.abs(road_pts[:, 0] - x) < 0.6) & (np.abs(road_pts[:, 1] - y) < 0.6)
        cloud_z_along.append(float(np.median(road_pts[m, 2])) if m.any() else np.nan)
    cloud_z_along = np.array(cloud_z_along, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(seg, centerline[:, 2], label="centerline GPS z", color="black", alpha=0.6)
    ax.plot(seg, cloud_z_along, label="cloud road z (median ±0.6m)", color="#4097ff", alpha=0.8)
    ax.plot(seg, grid_z_along, label="track.json road_z at centerline", color="#ff7700", alpha=0.8)
    if cones:
        cx = np.array([c["xyz"][0] for c in cones])
        cy = np.array([c["xyz"][1] for c in cones])
        cz = np.array([c["xyz"][2] for c in cones])
        nearest = []
        for x, y, z in zip(cx, cy, cz):
            d2 = (centerline[:, 0] - x) ** 2 + (centerline[:, 1] - y) ** 2
            k = int(np.argmin(d2))
            if d2[k] < 4.0 ** 2:
                nearest.append((seg[k], z))
        if nearest:
            ns, nz = zip(*nearest)
            ax.scatter(ns, nz, s=14, c="#ff8c00", edgecolors="black", linewidths=0.3, label="cones (nearest centerline)")
    ax.set_xlabel("distance along centerline (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Height profile along walked centerline — bumps here == ride bumps in sim")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main(project_dir: Path) -> None:
    diag = project_dir / "diag"
    diag.mkdir(parents=True, exist_ok=True)
    cloud = np.load(project_dir / "cloud_with_cones.npz")
    xyz, cls = cloud["xyz"], cloud["cls"]
    cones_path = project_dir / "cones.json"
    cones = json.loads(cones_path.read_text()).get("cones", []) if cones_path.exists() else []

    _z_hist({"road": xyz[cls == ROAD], "grass": xyz[cls == GRASS], "cone": xyz[cls == CONE]}, diag / "z_hist.png")
    _topdown_classes(xyz, cls, cones, diag / "topdown_classes.png")
    _height_image(xyz[cls == ROAD], 0.4, "road cloud", diag / "road_height_cloud.png")
    _height_image(xyz[cls == GRASS], 0.4, "grass cloud", diag / "grass_height_cloud.png")

    track_path = project_dir / "track.json"
    if track_path.exists():
        _track_json_diag(track_path, diag / "track_json_overview.png")
        _centerline_height_profile(track_path, project_dir, diag / "height_profile.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    args = p.parse_args()
    main(args.project)
