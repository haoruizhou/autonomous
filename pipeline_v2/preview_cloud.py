"""Top-down classified preview of cloud.npz."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def render(project_dir: Path, cell_m: float = 0.25, out_name: str = "cloud_topdown.png") -> Path:
    import matplotlib.pyplot as plt
    project_dir = Path(project_dir)
    d = np.load(project_dir / "cloud.npz")
    xyz, cls = d["xyz"], d["cls"]

    # Crop to a tight box around the camera path (drops far grandstand returns).
    rec_path = project_dir / "undistorted" / "reconstruction.json"
    if rec_path.exists():
        import json
        recs = json.loads(rec_path.read_text())
        cams = []
        for r in recs:
            for s in r["shots"].values():
                cams.append(s["gps_position"][:2])
        cams = np.array(cams)
        xlim = (cams[:, 0].min() - 20, cams[:, 0].max() + 20)
        ylim = (cams[:, 1].min() - 20, cams[:, 1].max() + 20)
        m = ((xyz[:, 0] >= xlim[0]) & (xyz[:, 0] <= xlim[1]) &
             (xyz[:, 1] >= ylim[0]) & (xyz[:, 1] <= ylim[1]))
        xyz = xyz[m]; cls = cls[m]

    # Bin by majority class per cell.
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    ix = ((xyz[:, 0] - x0) / cell_m).astype(np.int32)
    iy = ((xyz[:, 1] - y0) / cell_m).astype(np.int32)
    W = ix.max() + 1; H = iy.max() + 1
    grid = np.zeros((H, W), dtype=np.uint8)
    counts = np.zeros((H, W, 5), dtype=np.int32)
    for c in (1, 2, 3):  # road, grass, cone
        mask = cls == c
        np.add.at(counts[..., c], (iy[mask], ix[mask]), 1)
    grid = counts.argmax(axis=2).astype(np.uint8)
    grid[counts.sum(axis=2) == 0] = 0

    palette = np.array([
        [255, 255, 255],   # 0 empty/other
        [160, 160, 160],   # 1 road
        [60, 200, 80],     # 2 grass
        [255, 140, 0],     # 3 cone
    ], dtype=np.uint8)
    img = palette[grid]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, origin="lower",
              extent=[x0, x0 + W * cell_m, y0, y0 + H * cell_m])
    ax.set_aspect("equal")
    ax.set_title(f"Labeled cloud top-down — {project_dir.name} ({cell_m} m/cell)")
    out = project_dir / out_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--cell", type=float, default=0.25)
    args = p.parse_args()
    render(args.project, cell_m=args.cell)
