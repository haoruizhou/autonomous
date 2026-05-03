"""Stage F — export CARLA-inspectable OBJ/MTL assets from the v2 cloud."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation

ROAD, GRASS, CONE = 1, 2, 3


class ObjWriter:
    def __init__(self, mtl_name: str):
        self.lines = ["# FSAE autocross track — pipeline_v2", f"mtllib {mtl_name}", ""]
        self.nv = 0

    def group(self, name: str, material: str) -> None:
        self.lines.extend(["", f"g {name}", f"usemtl {material}"])

    def vertex(self, x: float, y: float, z: float) -> int:
        self.nv += 1
        self.lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        return self.nv

    def face(self, *idx: int) -> None:
        self.lines.append("f " + " ".join(str(i) for i in idx))

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def _height_percentile(xyz: np.ndarray, default: float = 0.0) -> float:
    if len(xyz) == 0:
        return default
    return float(np.percentile(xyz[:, 2], 5))


def _make_grid(xyz: np.ndarray, cell_m: float, margin_m: float = 0.0) -> dict:
    x0 = float(xyz[:, 0].min() - margin_m)
    y0 = float(xyz[:, 1].min() - margin_m)
    W = int(np.ceil((xyz[:, 0].max() + margin_m - x0) / cell_m)) + 1
    H = int(np.ceil((xyz[:, 1].max() + margin_m - y0) / cell_m)) + 1
    return {"x0": x0, "y0": y0, "W": W, "H": H, "cell": cell_m}


def _accumulate(grid: dict, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor((xyz[:, 0] - grid["x0"]) / grid["cell"]).astype(np.int32)
    iy = np.floor((xyz[:, 1] - grid["y0"]) / grid["cell"]).astype(np.int32)
    valid = (ix >= 0) & (iy >= 0) & (ix < grid["W"]) & (iy < grid["H"])
    ix, iy = ix[valid], iy[valid]
    z = xyz[valid, 2]
    count = np.zeros((grid["H"], grid["W"]), dtype=np.int32)
    zsum = np.zeros((grid["H"], grid["W"]), dtype=np.float64)
    np.add.at(count, (iy, ix), 1)
    np.add.at(zsum, (iy, ix), z)
    return count, zsum


def _emit_grid(
    writer: ObjWriter,
    grid: dict,
    occupied: np.ndarray,
    zgrid: np.ndarray,
    material: str,
    group: str,
    z_offset: float = 0.0,
) -> tuple[int, int]:
    vid: dict[tuple[int, int], int] = {}
    writer.group(group, material)
    for j, i in np.argwhere(occupied):
        vid[(int(j), int(i))] = writer.vertex(
            grid["x0"] + (i + 0.5) * grid["cell"],
            float(zgrid[j, i] + z_offset),
            grid["y0"] + (j + 0.5) * grid["cell"],
        )

    faces = 0
    H, W = occupied.shape
    for j in range(H - 1):
        for i in range(W - 1):
            a = vid.get((j, i))
            b = vid.get((j, i + 1))
            c = vid.get((j + 1, i))
            d = vid.get((j + 1, i + 1))
            if a and b and c:
                writer.face(a, b, c)
                faces += 1
            if b and d and c:
                writer.face(b, d, c)
                faces += 1
    return len(vid), faces


def _nearest_surface_z(point_xy: np.ndarray, surface_xy: np.ndarray, surface_z: np.ndarray, fallback: float) -> float:
    if len(surface_xy) == 0:
        return fallback
    d2 = np.sum((surface_xy - point_xy) ** 2, axis=1)
    return float(surface_z[int(np.argmin(d2))])


def _cone_instances(
    writer: ObjWriter,
    cones: list[dict],
    surface_xy: np.ndarray,
    surface_z: np.ndarray,
    fallback_z: float,
    radius_m: float,
    height_m: float,
) -> tuple[int, int]:
    writer.group("cones", "cone")
    faces = 0
    for cone in cones:
        x, y, _ = [float(v) for v in cone["xyz"]]
        z0 = _nearest_surface_z(np.array([x, y]), surface_xy, surface_z, fallback_z) + 0.04
        v1 = writer.vertex(x - radius_m, z0, y - radius_m)
        v2 = writer.vertex(x + radius_m, z0, y - radius_m)
        v3 = writer.vertex(x + radius_m, z0, y + radius_m)
        v4 = writer.vertex(x - radius_m, z0, y + radius_m)
        tip = writer.vertex(x, z0 + height_m, y)
        writer.face(v1, v2, v3)
        writer.face(v1, v3, v4)
        writer.face(v1, v2, tip)
        writer.face(v2, v3, tip)
        writer.face(v3, v4, tip)
        writer.face(v4, v1, tip)
        faces += 6
    return len(cones) * 5, faces


def export_obj(
    project_dir: Path,
    cloud_name: str = "cloud_with_cones.npz",
    cones_name: str = "cones.json",
    obj_name: str = "track.obj",
    mtl_name: str = "track.mtl",
    road_cell_m: float = 0.35,
    grass_margin_m: float = 20.0,
    cone_radius_m: float = 0.28,
    cone_height_m: float = 0.70,
) -> Path:
    project_dir = Path(project_dir)
    cloud = np.load(project_dir / cloud_name)
    xyz, cls = cloud["xyz"], cloud["cls"]
    cones_path = project_dir / cones_name
    cones = json.loads(cones_path.read_text()).get("cones", []) if cones_path.exists() else []

    road_xyz = xyz[cls == ROAD]
    grass_xyz = xyz[cls == GRASS]
    all_ground = xyz[(cls == ROAD) | (cls == GRASS)]
    ground_z = _height_percentile(all_ground)

    grid = _make_grid(all_ground if len(all_ground) else xyz, road_cell_m, margin_m=grass_margin_m)
    road_count, road_zsum = _accumulate(grid, road_xyz)
    grass_count, grass_zsum = _accumulate(grid, grass_xyz)
    road_occ = binary_closing(road_count >= 2, structure=np.ones((3, 3), dtype=bool))
    grass_candidate = binary_dilation(road_occ, iterations=max(1, int(round(grass_margin_m / road_cell_m))))
    grass_occ = grass_candidate & ~binary_dilation(road_occ, iterations=max(1, int(round(1.2 / road_cell_m))))

    road_z = np.full((grid["H"], grid["W"]), ground_z, dtype=np.float64)
    np.divide(road_zsum, road_count, out=road_z, where=road_count > 0)
    grass_z = np.full((grid["H"], grid["W"]), ground_z - 0.03, dtype=np.float64)
    np.divide(grass_zsum, grass_count, out=grass_z, where=grass_count > 0)

    road_cells = np.argwhere(road_occ)
    road_xy = np.column_stack([
        grid["x0"] + (road_cells[:, 1] + 0.5) * grid["cell"],
        grid["y0"] + (road_cells[:, 0] + 0.5) * grid["cell"],
    ]) if len(road_cells) else np.zeros((0, 2), dtype=np.float64)
    road_surface_z = road_z[road_cells[:, 0], road_cells[:, 1]] if len(road_cells) else np.zeros(0, dtype=np.float64)

    writer = ObjWriter(mtl_name)
    grass_v, grass_f = _emit_grid(writer, grid, grass_occ, grass_z, "grass", "grass", z_offset=-0.02)
    road_v, road_f = _emit_grid(writer, grid, road_occ, road_z, "road", "road", z_offset=0.02)
    cone_v, cone_f = _cone_instances(writer, cones, road_xy, road_surface_z, ground_z, cone_radius_m, cone_height_m)

    obj_path = project_dir / obj_name
    mtl_path = project_dir / mtl_name
    writer.write(obj_path)
    mtl_path.write_text(
        "newmtl road\n"
        "Ka 0.08 0.08 0.08\n"
        "Kd 0.32 0.32 0.32\n"
        "Ks 0.02 0.02 0.02\n\n"
        "newmtl grass\n"
        "Ka 0.03 0.12 0.03\n"
        "Kd 0.12 0.55 0.16\n"
        "Ks 0.00 0.00 0.00\n\n"
        "newmtl cone\n"
        "Ka 0.25 0.10 0.00\n"
        "Kd 1.00 0.42 0.00\n"
        "Ks 0.05 0.03 0.00\n"
    )
    print(f"  OBJ: {obj_path}")
    print(f"  MTL: {mtl_path}")
    print(f"  grass: {grass_v:,} verts, {grass_f:,} faces")
    print(f"  road:  {road_v:,} verts, {road_f:,} faces")
    print(f"  cones: {len(cones):,} instances, {cone_v:,} verts, {cone_f:,} faces")
    return obj_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export v2 labeled cloud to OBJ/MTL")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--cloud", default="cloud_with_cones.npz")
    p.add_argument("--cones", default="cones.json")
    p.add_argument("--road-cell", type=float, default=0.35)
    p.add_argument("--grass-margin", type=float, default=20.0)
    args = p.parse_args()
    export_obj(
        args.project,
        cloud_name=args.cloud,
        cones_name=args.cones,
        road_cell_m=args.road_cell,
        grass_margin_m=args.grass_margin,
    )
