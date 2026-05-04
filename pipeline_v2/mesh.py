"""Stage mesh — dense labeled point cloud → vertex-colored GLB for the frontend.

Reads ``cloud.npz`` (same artifact as ``cloud.assemble_project``): ``xyz``, ``rgb``, ``cls``.
Keeps road + grass only; cones stay as instanced geometry from ``track.json``.

After writing ``track_mesh.glb`` under ``project_dir``, optionally copies it to
``<frontend_dir>/public/data/`` so ``--stage mesh`` is self-contained.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

ROAD, GRASS = 1, 2


def build_track_mesh_glb(
    project_dir: Path,
    *,
    cloud_name: str = "cloud.npz",
    out_name: str = "track_mesh.glb",
    frontend_dir: Path | None = None,
    poisson_depth: int = 9,
    voxel_m: float = 0.05,
    target_triangles: int = 200_000,
    density_quantile: float = 0.02,
    bbox_scale: float = 1.02,
) -> Path:
    try:
        import open3d as o3d
        import trimesh
    except ImportError as e:
        raise RuntimeError(
            "mesh stage requires open3d and trimesh. Open3D provides wheels for "
            "CPython 3.10–3.12 only. Install with:  uv sync --extra mesh  "
            "(use Python <=3.12 for that environment)."
        ) from e

    project_dir = Path(project_dir)
    cloud_path = project_dir / cloud_name
    if not cloud_path.exists():
        raise FileNotFoundError(f"Missing {cloud_path}; run the cloud stage first.")

    data = np.load(cloud_path)
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    rgb = np.asarray(data["rgb"], dtype=np.float64)
    cls = np.asarray(data["cls"], dtype=np.int32)

    mask = np.isin(cls, (ROAD, GRASS))
    xyz = xyz[mask]
    rgb = rgb[mask]
    if len(xyz) < 500:
        raise RuntimeError(f"Too few road/grass points after filter: {len(xyz)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb / 255.0, 0.0, 1.0))

    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_m))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) < 500:
        raise RuntimeError(f"Too few points after cleanup: {len(pcd.points)}")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 3, max_nn=30),
    )
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth),
    )

    dens = np.asarray(densities, dtype=np.float64).reshape(-1)
    if len(dens) == len(mesh.vertices) and len(dens) > 0:
        thresh = float(np.quantile(dens, density_quantile))
        remove = dens < thresh
        mesh.remove_vertices_by_mask(remove)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(xyz))
    bbox = bbox.scale(bbox_scale, bbox.get_center())
    mesh = mesh.crop(bbox)

    n_tri = len(mesh.triangles)
    if n_tri > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles))

    mesh.compute_vertex_normals()

    # Nearest-neighbour colours from the voxelised point cloud (Poisson has no RGB).
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    tree = KDTree(pts)
    v_np = np.asarray(mesh.vertices, dtype=np.float64)
    _, nn = tree.query(v_np, k=1)
    nn = np.asarray(nn, dtype=np.int64).reshape(-1)
    vcols = cols[nn]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcols, 0.0, 1.0))

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    vc = (np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.column_stack([vc, np.full((len(vc),), 255, dtype=np.uint8)])

    tm = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgba, process=False)
    glb_bytes = trimesh.exchange.gltf.export_glb(tm)

    out_path = project_dir / out_name
    out_path.write_bytes(glb_bytes)
    print(f"  GLB: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if frontend_dir is not None:
        fd = Path(frontend_dir)
        dest_dir = fd / "public" / "data"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / out_name
        shutil.copy2(out_path, dest)
        print(f"  synced {out_path} → {dest}")

    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build vertex-colored track_mesh.glb from cloud.npz")
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--cloud", default="cloud.npz")
    ap.add_argument("--out", default="track_mesh.glb")
    ap.add_argument("--frontend-dir", type=Path, default=None)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--voxel-m", type=float, default=0.05)
    ap.add_argument("--target-triangles", type=int, default=200_000)
    args = ap.parse_args()
    build_track_mesh_glb(
        args.project,
        cloud_name=args.cloud,
        out_name=args.out,
        frontend_dir=args.frontend_dir,
        poisson_depth=args.poisson_depth,
        voxel_m=args.voxel_m,
        target_triangles=args.target_triangles,
    )
