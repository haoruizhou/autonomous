"""Quick preview of an OpenSfM sparse reconstruction.

Outputs:
    <project>/preview_topdown.png   — matplotlib top-down: shots + points + GPS
    <project>/sparse.ply             — point cloud for MeshLab
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_recs(project_dir: Path) -> list[dict]:
    return json.loads((project_dir / "reconstruction.json").read_text())


def _shot_xyz(shot: dict) -> np.ndarray:
    R = np.array(shot["rotation"])
    t = np.array(shot["translation"])
    theta = np.linalg.norm(R)
    if theta < 1e-9:
        Rm = np.eye(3)
    else:
        k = R / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return -Rm.T @ t


def render_topdown(project_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    recs = _load_recs(project_dir)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for rec in recs:
        rec["_shots_sorted"] = sorted(
            rec["shots"].values(), key=lambda s: s.get("capture_time", 0.0)
        )
    all_cams = np.concatenate([
        np.array([_shot_xyz(s) for s in rec["_shots_sorted"]])
        for rec in recs
    ])
    cx, cy = all_cams[:, 0], all_cams[:, 1]
    pad = max(20.0, 0.2 * max(np.ptp(cx), np.ptp(cy)))
    xlim = (cx.min() - pad, cx.max() + pad)
    ylim = (cy.min() - pad, cy.max() + pad)

    for i, rec in enumerate(recs):
        c = colors[i % len(colors)]
        pts = np.array([p["coordinates"] for p in rec["points"].values()])
        if len(pts):
            m = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
                 (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
            ax.scatter(pts[m, 0], pts[m, 1], s=0.4, c=c, alpha=0.3,
                       label=f"rec{i} pts ({m.sum()}/{len(pts)})")
        shots_sorted = rec["_shots_sorted"]
        cams = np.array([_shot_xyz(s) for s in shots_sorted])
        ax.plot(cams[:, 0], cams[:, 1], "-", c=c, lw=1.5,
                label=f"rec{i} cams ({len(cams)})")
        gps = np.array([s["gps_position"] for s in shots_sorted])
        ax.plot(gps[:, 0], gps[:, 1], "--", c=c, lw=0.8, alpha=0.6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Sparse reconstruction (top-down) — {project_dir.name}\n"
                 "solid = SfM cam path, dashed = GPS prior")
    ax.grid(alpha=0.3)
    out = project_dir / "preview_topdown.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def export_ply(project_dir: Path) -> Path:
    recs = _load_recs(project_dir)
    rows = []
    for rec in recs:
        for p in rec["points"].values():
            x, y, z = p["coordinates"]
            r, g, b = p.get("color", [200, 200, 200])
            rows.append((x, y, z, int(r), int(g), int(b)))
    out = project_dir / "sparse.ply"
    with out.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in rows:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    args = p.parse_args()
    print("topdown:", render_topdown(args.project))
    print("ply:    ", export_ply(args.project))
