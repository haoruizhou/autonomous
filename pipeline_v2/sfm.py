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


def _have_docker() -> bool:
    return shutil.which("docker") is not None


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
    print("  OpenSfM done.")


def collect_outputs(project_dir: Path) -> dict:
    """Return paths to the artifacts the rest of the pipeline cares about."""
    project_dir = Path(project_dir)
    return {
        "reconstruction": project_dir / "reconstruction.json",
        "tracks": project_dir / "tracks.csv",
        "undistorted_dir": project_dir / "undistorted",
        "depthmaps_dir": project_dir / "undistorted" / "depthmaps",
        "merged_ply": project_dir / "undistorted" / "depthmaps" / "merged.ply",
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
