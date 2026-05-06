"""Stage B — run OpenSfM on a project dir via Docker.

Expects a project layout produced by `extract.write_project()`:
    <project>/images/*.jpg
    <project>/exif_overrides.json
    <project>/config.yaml

Runs the OpenSfM pipeline and emits:
    <project>/exif/*.json                # parsed EXIF (GPS injected)
    <project>/features/*.npz             # SIFT features
    <project>/matches/*.pkl.gz           # matches
    <project>/tracks.csv                 # track graph
    <project>/reconstruction.json        # camera poses + sparse cloud (GPS-anchored)
    <project>/undistorted/...
    <project>/undistorted/depthmaps/...
    <project>/undistorted/depthmaps/merged.ply   # dense cloud (final SfM output)

Implementation: shell out to a single Docker run that executes all OpenSfM
commands in sequence inside the container. The project dir is bind-mounted at
/project; OpenSfM is invoked from /source/OpenSfM/bin/opensfm where it lives.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

# Stages run end-to-end inside the container. `compute_depthmaps` is the
# expensive one; can be skipped via --no-dense.
_DEFAULT_STAGES: Sequence[str] = (
    "extract_metadata",
    "detect_features",
    "match_features",
    "create_tracks",
    "reconstruct",
    "undistort",
    "compute_depthmaps",
)

# GPU path: replace compute_depthmaps with export_openmvs; OpenMVS handles depthmaps.
_GPU_STAGES: Sequence[str] = (
    "extract_metadata",
    "detect_features",
    "match_features",
    "create_tracks",
    "reconstruct",
    "undistort",
    "export_openmvs",
)


def _have_docker() -> bool:
    return shutil.which("docker") is not None


def _purge_corrupt_intermediate_files(project_dir: Path) -> None:
    """Delete zero-byte or corrupt .npz/.exif files left by interrupted runs.

    OpenSfM feature files are numpy zip archives. A partial write leaves a file
    that passes os.path.exists() but raises BadZipFile when joblib workers open
    it, aborting the whole stage. Removing them lets detect_features/
    extract_metadata recompute cleanly on restart.
    """
    import zipfile
    removed = 0
    for subdir in ("features", "exif"):
        d = project_dir / subdir
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix == ".npz":
                if f.stat().st_size == 0:
                    f.unlink()
                    removed += 1
                else:
                    try:
                        with zipfile.ZipFile(f):
                            pass
                    except zipfile.BadZipFile:
                        f.unlink()
                        removed += 1
            elif f.suffix == ".exif" and f.stat().st_size == 0:
                f.unlink()
                removed += 1
    if removed:
        print(f"  Purged {removed} corrupt intermediate file(s) before starting Docker")


def run_opensfm(
    project_dir: Path,
    image: str = "opensfm:ubuntu24",
    stages: Sequence[str] = _DEFAULT_STAGES,
    extra_env: dict | None = None,
    platform: str | None = None,
) -> None:
    """Run the OpenSfM pipeline inside Docker against a prepared project dir."""
    project_dir = Path(project_dir).resolve()
    if not (project_dir / "images").is_dir():
        raise FileNotFoundError(f"{project_dir}/images is missing — run extract.py first")
    if not (project_dir / "config.yaml").exists():
        raise FileNotFoundError(f"{project_dir}/config.yaml is missing")
    if not _have_docker():
        raise RuntimeError("docker binary not found on PATH")
    _purge_corrupt_intermediate_files(project_dir)

    cmds = " && ".join(f"bin/opensfm {s} /project" for s in stages)
    docker_cmd = ["docker", "run", "--rm",
                  "-v", f"{project_dir}:/project",
                  "-w", "/source/OpenSfM"]
    if platform:
        docker_cmd.extend(["--platform", platform])
    for k, v in (extra_env or {}).items():
        docker_cmd.extend(["-e", f"{k}={v}"])
    docker_cmd.extend([image, "bash", "-lc", cmds])

    print(f"  Running OpenSfM in Docker on {project_dir} …")
    print(f"  Stages: {' → '.join(stages)}")
    subprocess.run(docker_cmd, check=True)
    subprocess.run(["docker", "run", "--rm", "-v", f"{project_dir}:/project",
                    image, "chmod", "-R", "a+rX", "/project"], check=False)
    print("  OpenSfM done.")


def run_opensfm_gpu(
    project_dir: Path,
    image: str = "opensfm:ubuntu24_cuda",
    stages: Sequence[str] = _GPU_STAGES,
    cuda_device: int = 0,
    openmvs_resolution: int = 1,
) -> None:
    """Run OpenSfM pipeline with GPU-accelerated depthmaps via OpenMVS.

    OpenSfM runs the usual stages up to export_openmvs, then OpenMVS
    DensifyPointCloud uses CUDA to produce the dense point cloud at
    <project>/undistorted/openmvs/scene_dense.ply.

    Args:
        openmvs_resolution: 0=full, 1=half (default), 2=quarter. Half is
            sufficient for track-width measurements and runs ~4× faster.
    """
    project_dir = Path(project_dir).resolve()
    if not (project_dir / "images").is_dir():
        raise FileNotFoundError(f"{project_dir}/images is missing — run extract.py first")
    if not _have_docker():
        raise RuntimeError("docker binary not found on PATH")
    _purge_corrupt_intermediate_files(project_dir)

    sfm_cmds = " && ".join(f"bin/opensfm {s} /project" for s in stages)
    mvs_scene = "/project/undistorted/openmvs/scene.mvs"
    mvs_cmd = (
        f"DensifyPointCloud {mvs_scene}"
        f" --cuda-device {cuda_device}"
        f" --number-views-fuse 2"
        f" --resolution-level {openmvs_resolution}"
    )
    full_cmd = f"{sfm_cmds} && {mvs_cmd}"

    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{project_dir}:/project",
        "-w", "/source/OpenSfM",
        image, "bash", "-lc", full_cmd,
    ]
    print(f"  Running OpenSfM+OpenMVS (GPU) on {project_dir} …")
    print(f"  Stages: {' → '.join(stages)} → DensifyPointCloud (CUDA)")
    subprocess.run(docker_cmd, check=True)
    subprocess.run(["docker", "run", "--rm", "-v", f"{project_dir}:/project",
                    image, "chmod", "-R", "a+rX", "/project"], check=False)
    print("  OpenSfM+OpenMVS GPU done.")


def collect_outputs(project_dir: Path) -> dict:
    """Return paths to the artifacts the rest of the pipeline cares about."""
    project_dir = Path(project_dir)
    return {
        "reconstruction": project_dir / "reconstruction.json",
        "tracks": project_dir / "tracks.csv",
        "undistorted_dir": project_dir / "undistorted",
        "depthmaps_dir": project_dir / "undistorted" / "depthmaps",
        "merged_ply": project_dir / "undistorted" / "depthmaps" / "merged.ply",
        "openmvs_ply": project_dir / "undistorted" / "openmvs" / "scene_dense.ply",
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run OpenSfM on a prepared project directory")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--image", default="opensfm:ubuntu24",
                   help="Docker image tag")
    p.add_argument("--no-dense", action="store_true",
                   help="Skip undistort + compute_depthmaps (sparse only)")
    p.add_argument("--platform", default=None,
                   help="Docker --platform value, e.g. linux/amd64 if Rosetta is needed")
    args = p.parse_args()
    stages = list(_DEFAULT_STAGES)
    if args.no_dense:
        stages = [s for s in stages if s not in ("undistort", "compute_depthmaps")]
    run_opensfm(args.project, image=args.image, stages=stages, platform=args.platform)
    for k, v in collect_outputs(args.project).items():
        print(f"  {k}: {v} (exists={v.exists()})")
