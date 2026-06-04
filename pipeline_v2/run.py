"""End-to-end orchestrator for pipeline_v2.

Stages (in order):
    A extract   — videos + GPX → OpenSfM project
    B sfm       — OpenSfM (Docker): features, match, reconstruct, undistort, dense
    C semantic  — Mask2Former labels for each undistorted component
    D cloud     — assemble labeled point cloud (cloud.npz)
    D2 dense_depthmap — project PLY into per-frame depthmaps (for GPU/OpenMVS path)
    E cones     — YOLO-World cone detection + cluster + stamp into cloud_with_cones.npz
    F export    — track.json (browser) + track.obj/.mtl (CARLA-ish)
    F2 mesh     — cloud.npz → vertex-colored track_mesh.glb (+ copy to frontend); needs ``uv sync --extra mesh`` (Py 3.10–3.12)
    G diag      — diagnostic PNGs (z hist, height profile, top-down classes)
    H sync      — copy track.{json,obj,mtl,glb} into frontend/public/data/

Designed so a single `--stage all` run can be left running overnight.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pipeline_v2 import cloud, diag, export, export_json, extract, sfm

_ALL_STAGES = ("extract", "masks", "sfm", "semantic", "cloud", "dense_depthmap", "cones", "export", "mesh", "diag", "sync")


def _components(project_dir: Path) -> list[tuple[str, str]]:
    """Return [(undistorted_subdir, labels_subdir), ...] for whichever recs exist."""
    pairs: list[tuple[str, str]] = []
    for undist, lbls in (
        ("undistorted", "labels"),
        ("undistorted_rec1", "labels_rec1"),
        ("undistorted_rec2", "labels_rec2"),
        ("undistorted_rec3", "labels_rec3"),
    ):
        if (project_dir / undist / "reconstruction.json").exists() or (project_dir / undist / "images").is_dir():
            pairs.append((undist, lbls))
    return pairs


def _sync_frontend(project_dir: Path, frontend_dir: Path) -> None:
    target = frontend_dir / "public" / "data"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("track.json", "track.obj", "track.mtl", "track_mesh.glb"):
        src = project_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
            print(f"  synced {src} → {target / name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Photogrammetry-first track reconstruction (end-to-end)")
    p.add_argument("--video", type=Path, action="append", default=[],
                   help="Input video (repeatable; use once per clip). Required for extract.")
    p.add_argument("--gpx", type=Path, default=None,
                   help="GPX path. Required for extract.")
    p.add_argument("--out", type=Path, required=True, help="Project / output directory")
    p.add_argument("--fps", type=float, default=6.0, help="Frame sampling rate (default 6 fps)")
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="Start offset in seconds into each video (default 0)")
    p.add_argument("--duration-sec", type=float, default=None,
                   help="Clip duration in seconds (default: full video)")
    p.add_argument("--video-start", default=None,
                   help="Explicit video frame-0 UTC start (ISO-8601, e.g. 2026-04-29T12:56:48Z). "
                        "Overrides filename/metadata. Applies to a single --video only.")
    p.add_argument("--min-coverage", type=float, default=0.6,
                   help="Warn if GPS covers less than this fraction of the video (default 0.6)")
    p.add_argument("--reject-coverage", type=float, default=0.1,
                   help="Reject if GPS covers less than this fraction of the video (default 0.1)")
    p.add_argument("--stage", choices=(*_ALL_STAGES, "all"), default="all")
    p.add_argument("--skip", action="append", default=[],
                   help=f"Stages to skip when --stage all; choices: {_ALL_STAGES}")
    p.add_argument("--no-dense", action="store_true",
                   help="In sfm stage: skip undistort + depthmaps")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU-accelerated OpenMVS for dense reconstruction (requires opensfm:ubuntu24_cuda)")
    p.add_argument("--docker-image", default="opensfm:ubuntu24")
    p.add_argument("--docker-platform", default=None)

    p.add_argument("--cloud-stride", type=int, default=4)
    p.add_argument("--cloud-max-depth", type=float, default=40.0)

    p.add_argument("--cone-model", default="yolov8s-world.pt")
    p.add_argument("--cone-conf", type=float, default=0.12)
    p.add_argument("--cone-cluster-radius", type=float, default=0.75)
    p.add_argument("--cone-min-observations", type=int, default=2)
    p.add_argument("--cone-max-depth", type=float, default=45.0)

    p.add_argument("--export-cell", type=float, default=0.35)
    p.add_argument("--export-grass-margin", type=float, default=12.0)
    p.add_argument("--export-road-buffer", type=float, default=1.2)

    p.add_argument("--mesh-poisson-depth", type=int, default=9)
    p.add_argument("--mesh-voxel-m", type=float, default=0.05)
    p.add_argument("--mesh-target-triangles", type=int, default=200_000)

    p.add_argument("--frontend-dir", type=Path, default=Path("frontend"))
    args = p.parse_args()

    project_dir: Path = args.out
    selected = set(_ALL_STAGES) if args.stage == "all" else {args.stage}
    selected -= set(args.skip)

    def run(stage: str) -> bool:
        return stage in selected

    if run("extract") and (not args.video or args.gpx is None):
        p.error("--video (repeat at least once) and --gpx are required when running the extract stage")

    if run("extract"):
        print("\n[A] Extract frames + GPS → OpenSfM project")
        extract.write_project(args.video, args.gpx, project_dir, sample_fps=args.fps,
                              start_sec=args.start_sec, duration_sec=args.duration_sec,
                              video_start_override=args.video_start,
                              min_coverage=args.min_coverage,
                              reject_coverage=args.reject_coverage)

    if run("masks"):
        from pipeline_v2 import semantic  # noqa: PLC0415  lazy — loads torch/transformers
        print("\n[A2] Generate OpenSfM feature masks (suppress people/vehicles)")
        if not (project_dir / "images").is_dir():
            print("  [skip] images/ not found — run extract first")
        else:
            semantic.generate_opensfm_masks(project_dir)

    if run("sfm"):
        print("\n[B] OpenSfM (Docker)")
        if args.gpu:
            gpu_image = args.docker_image if args.docker_image != "opensfm:ubuntu24" else "opensfm:ubuntu24_cuda"
            sfm.run_opensfm_gpu(project_dir, image=gpu_image)
        else:
            stages = list(sfm._DEFAULT_STAGES)
            if args.no_dense:
                stages = [s for s in stages if s not in ("undistort", "compute_depthmaps")]
            sfm.run_opensfm(project_dir, image=args.docker_image, stages=stages, platform=args.docker_platform)
        for k, v in sfm.collect_outputs(project_dir).items():
            print(f"  {k}: {v} (exists={v.exists()})")

    components = _components(project_dir)

    if run("semantic"):
        from pipeline_v2 import semantic  # noqa: PLC0415  lazy — loads torch/transformers
        print("\n[C] Semantic labels (Mask2Former-Cityscapes)")
        if not components:
            print("  [skip] no undistorted/* found — run sfm first")
        for undist, lbls in components:
            print(f"  → {undist} / {lbls}")
            semantic.label_project(
                project_dir,
                images_subdir=f"{undist}/images",
                out_subdir=lbls,
            )

    if run("cloud"):
        print("\n[D] Assemble labeled point cloud")
        cloud.assemble_project(
            project_dir,
            components=tuple(components) if components else (("undistorted", "labels"), ("undistorted_rec1", "labels_rec1")),
            pixel_stride=args.cloud_stride,
            max_depth_m=args.cloud_max_depth,
        )

    if run("dense_depthmap"):
        from pipeline_v2 import dense_depthmap
        print("\n[D2] Generate dense depthmaps from OpenMVS PLY")
        ply_path = project_dir / "undistorted" / "openmvs" / "scene_dense.ply"
        if not ply_path.exists():
            print(f"  [skip] {ply_path} not found — is OpenMVS dense reconstruction done?")
        else:
            dense_depthmap.generate_depthmaps(
                project_dir=project_dir,
                ply_path=ply_path,
                model_name=args.cone_model,
                conf=args.cone_conf,
                max_depth_m=args.cone_max_depth,
            )

    # Auto-detect: prefer .dense.npz (OpenMVS GPU path) if available,
    # otherwise fall back to .clean.npz (OpenSfM CPU patch-match depthmaps).
    # This works on both GPU and CPU-only (Mac) systems.
    def _ply_mode() -> bool:
        for undist, _ in components:
            dm_dir = project_dir / undist / "depthmaps"
            if dm_dir.is_dir():
                sample = next(dm_dir.glob("*.dense.npz"), None)
                if sample is not None:
                    return True
        return False

    ply_mode = _ply_mode()
    ply_mode_label = "dense (OpenMVS GPU)" if ply_mode else "clean (OpenSfM CPU)"
    if run("cones"):
        from pipeline_v2 import cones  # noqa: PLC0415  lazy — loads torch/ultralytics
        print(f"\n[E] Cone detection + clustering + stamp  [ply_mode={ply_mode_label}]")
        cones.detect_project(
            project_dir,
            components=tuple(u for u, _ in components) if components else ("undistorted", "undistorted_rec1"),
            model_name=args.cone_model,
            conf=args.cone_conf,
            cluster_radius_m=args.cone_cluster_radius,
            min_observations=args.cone_min_observations,
            max_depth_m=args.cone_max_depth,
            ply_mode=ply_mode,
        )
        cones.stamp_cloud(project_dir)

    if run("export"):
        print("\n[F] Export track.json + track.obj/.mtl")
        export_json.export_track_json(
            project_dir,
            cell_m=args.export_cell,
            grass_margin_m=args.export_grass_margin,
            road_buffer_m=args.export_road_buffer,
        )
        try:
            export.export_obj(project_dir)
        except Exception as e:
            print(f"  [warn] export.export_obj failed: {e}")

    if run("mesh"):
        from pipeline_v2.mesh import build_track_mesh_glb  # noqa: PLC0415  lazy — needs open3d
        print("\n[F2] Vertex-colored mesh (Poisson) → track_mesh.glb")
        try:
            build_track_mesh_glb(
                project_dir,
                frontend_dir=args.frontend_dir,
                poisson_depth=args.mesh_poisson_depth,
                voxel_m=args.mesh_voxel_m,
                target_triangles=args.mesh_target_triangles,
            )
        except Exception as e:
            print(f"  [warn] build_track_mesh_glb failed: {e}")

    if run("diag"):
        print("\n[G] Diagnostics")
        try:
            diag.main(project_dir)
        except Exception as e:
            print(f"  [warn] diag failed: {e}")

    if run("sync"):
        print("\n[H] Sync frontend assets")
        _sync_frontend(project_dir, args.frontend_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
