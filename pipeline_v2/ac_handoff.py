from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation


def build_flat_asphalt_mask(
    road_cells: list[list[float]],
    grid: dict,
    *,
    close_cells: int = 2,
    apron_cells: int = 4,
) -> np.ndarray:
    mask = np.zeros((int(grid["height"]), int(grid["width"])), dtype=bool)
    for i, j, _z in road_cells:
        ii = int(i)
        jj = int(j)
        if 0 <= ii < mask.shape[1] and 0 <= jj < mask.shape[0]:
            mask[jj, ii] = True
    if close_cells > 0:
        mask = binary_closing(mask, structure=np.ones((close_cells * 2 + 1, close_cells * 2 + 1), dtype=bool))
    if apron_cells > 0:
        mask = binary_dilation(mask, iterations=int(apron_cells))
    return mask


def cell_vertices_flat(i: int, j: int, grid: dict, *, z: float = 0.0) -> list[tuple[float, float, float]]:
    cell = float(grid["cell_m"])
    x0 = float(grid["x0"])
    y0 = float(grid["y0"])
    x = x0 + i * cell
    y = y0 + j * cell
    return [
        (x, z, y),
        (x + cell, z, y),
        (x + cell, z, y + cell),
        (x, z, y + cell),
    ]


def write_asphalt_obj(mask: np.ndarray, grid: dict, out_obj: Path, *, mtl_name: str = "track_asphalt.mtl") -> None:
    lines = ["# Flat Assetto Corsa handoff asphalt mesh", f"mtllib {mtl_name}", "g asphalt", "usemtl asphalt"]
    vertex_count = 0
    for j, i in np.argwhere(mask):
        verts = cell_vertices_flat(int(i), int(j), grid)
        for x, y, z in verts:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        a, b, c, d = vertex_count + 1, vertex_count + 2, vertex_count + 3, vertex_count + 4
        lines.append(f"f {a} {b} {c}")
        lines.append(f"f {a} {c} {d}")
        vertex_count += 4
    out_obj.write_text("\n".join(lines) + "\n")


def write_mtl(out_mtl: Path) -> None:
    out_mtl.write_text(
        "newmtl asphalt\n"
        "Ka 0.08 0.08 0.08\n"
        "Kd 0.28 0.28 0.27\n"
        "Ks 0.02 0.02 0.02\n"
    )


def _flat_cone(cone: dict) -> dict:
    out = dict(cone)
    xyz = list(out.get("xyz", [0.0, 0.0, 0.0]))
    out["xyz"] = [float(xyz[0]), float(xyz[1]), 0.0]
    return out


def write_cones_csv(cones: list[dict], out_csv: Path) -> None:
    lines = ["id,x,y,z,n_observations,mean_conf"]
    for cone in cones:
        flat = _flat_cone(cone)
        x, y, z = flat["xyz"]
        lines.append(
            f"{int(flat.get('id', 0))},{x:.3f},{y:.3f},{z:.3f},"
            f"{int(flat.get('n_observations', 0))},{float(flat.get('mean_conf', 0.0)):.3f}"
        )
    out_csv.write_text("\n".join(lines) + "\n")


def write_guide_path(centerline: list[list[float]], out_csv: Path) -> None:
    lines = ["x,y,z"]
    for x, y, _z in centerline:
        lines.append(f"{float(x):.3f},{float(y):.3f},0.000")
    out_csv.write_text("\n".join(lines) + "\n")


def write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        "# Assetto Corsa handoff package\n\n"
        "This package is a flat, track-shaped engineering export for Blender/ksEditor conversion.\n\n"
        "Files:\n"
        "- `track_asphalt.obj` / `track_asphalt.mtl`: flat asphalt footprint, OBJ Y-up, meters.\n"
        "- `cones.csv`: flattened cone placements in source XY meters.\n"
        "- `cones.json`: same cone records with Z forced to 0.\n"
        "- `guide_path.csv`: flattened camera/GPS guide path for visual reference.\n\n"
        "Suggested flow: import OBJ into Blender, verify scale/orientation, add/replace cone models, "
        "export FBX, then convert to KN5 with ksEditor for Assetto Corsa.\n"
    )


def export_ac_handoff(
    project_dir: Path,
    *,
    out_dir: Path | None = None,
    apron_cells: int = 4,
    close_cells: int = 2,
) -> Path:
    project_dir = Path(project_dir)
    if out_dir is None:
        out_dir = project_dir / "ac_handoff"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    track = json.loads((project_dir / "track.json").read_text())
    grid = track["grid"]
    mask = build_flat_asphalt_mask(
        track.get("road_cells", []),
        grid,
        close_cells=close_cells,
        apron_cells=apron_cells,
    )

    write_asphalt_obj(mask, grid, out_dir / "track_asphalt.obj")
    write_mtl(out_dir / "track_asphalt.mtl")
    cones = [_flat_cone(c) for c in track.get("cones", [])]
    write_cones_csv(cones, out_dir / "cones.csv")
    (out_dir / "cones.json").write_text(json.dumps({"cones": cones}, indent=2))
    write_guide_path(track.get("centerline_hint", []), out_dir / "guide_path.csv")
    write_readme(out_dir)

    print(f"  AC handoff: {out_dir}")
    print(f"  asphalt cells: {int(mask.sum()):,}")
    print(f"  cones: {len(cones):,}")
    return out_dir


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Export flat Assetto Corsa handoff assets from track.json")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--apron-cells", type=int, default=4)
    p.add_argument("--close-cells", type=int, default=2)
    args = p.parse_args()
    export_ac_handoff(
        args.project,
        out_dir=args.out_dir,
        apron_cells=args.apron_cells,
        close_cells=args.close_cells,
    )


if __name__ == "__main__":
    main()
