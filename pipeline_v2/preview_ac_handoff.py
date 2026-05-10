from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_obj_topdown(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            _tag, x, _y, z = line.split()[:4]
            vertices.append((float(x), float(z)))
        elif line.startswith("f "):
            idx = []
            for part in line.split()[1:]:
                idx.append(int(part.split("/")[0]) - 1)
            if len(idx) == 3:
                faces.append(idx)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def load_csv_points(path: Path) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def load_cones_csv(path: Path) -> np.ndarray:
    if not Path(path).exists():
        return np.zeros((0, 2), dtype=np.float64)
    return load_csv_points(path)


def render_preview(handoff_dir: Path, out_path: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    handoff_dir = Path(handoff_dir)
    if out_path is None:
        out_path = handoff_dir / "preview.png"
    out_path = Path(out_path)

    vertices, faces = load_obj_topdown(handoff_dir / "track_asphalt.obj")
    guide = load_csv_points(handoff_dir / "guide_path.csv") if (handoff_dir / "guide_path.csv").exists() else np.zeros((0, 2))
    cones = load_cones_csv(handoff_dir / "cones.csv")

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#111111")
    ax.set_facecolor("#161616")

    if len(vertices) and len(faces):
        tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
        ax.triplot(tri, color="#303030", linewidth=0.12, alpha=0.45)
        ax.tripcolor(tri, facecolors=np.full(len(faces), 0.35), cmap="gray", vmin=0.0, vmax=1.0, alpha=0.9)

    if len(guide):
        ax.plot(guide[:, 0], guide[:, 1], color="#42d9ff", linewidth=1.8, label="guide path")
        ax.scatter([guide[0, 0]], [guide[0, 1]], s=80, color="#00ff44", edgecolors="black", linewidths=0.6, label="start")
        ax.scatter([guide[-1, 0]], [guide[-1, 1]], s=80, color="#ff2020", edgecolors="black", linewidths=0.6, label="end")

    if len(cones):
        ax.scatter(cones[:, 0], cones[:, 1], s=36, color="#ff8c00", edgecolors="black", linewidths=0.35, label="cones")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#555555", alpha=0.35, linewidth=0.7)
    ax.set_xlabel("X meters", color="#dddddd")
    ax.set_ylabel("Y meters", color="#dddddd")
    ax.tick_params(colors="#dddddd")
    ax.set_title(
        f"Flat AC handoff preview — {len(vertices):,} verts, {len(faces):,} faces, {len(cones):,} cones",
        color="#eeeeee",
    )
    if len(guide) or len(cones):
        leg = ax.legend(facecolor="#222222", edgecolor="#888888")
        for text in leg.get_texts():
            text.set_color("#eeeeee")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  preview: {out_path}")
    return out_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Render top-down preview of flat AC handoff package")
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    render_preview(args.handoff, args.out)


if __name__ == "__main__":
    main()
