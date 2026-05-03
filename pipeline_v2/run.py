"""End-to-end orchestrator for pipeline_v2.

Currently scaffolds:
    Stage A — extract: video + GPX → OpenSfM project layout
    Stage B — sfm:     OpenSfM via Docker → reconstruction.json + dense cloud

Subsequent stages (semantic projection, cone re-anchoring, OBJ + xodr export)
will be wired in as they land.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_v2 import extract, sfm


def main() -> None:
    p = argparse.ArgumentParser(description="Photogrammetry-first track reconstruction")
    p.add_argument("--video", type=Path, action="append", required=True,
                   help="Input video (repeatable; use once per clip)")
    p.add_argument("--gpx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Project / output directory")
    p.add_argument("--fps", type=float, default=4.0,
                   help="Frame sampling rate (default 4 fps)")
    p.add_argument("--stage", choices=["extract", "sfm", "all"], default="all")
    p.add_argument("--no-dense", action="store_true",
                   help="In sfm stage: skip undistort + depthmaps")
    p.add_argument("--docker-image", default="opensfm:ubuntu24")
    p.add_argument("--docker-platform", default=None,
                   help="Docker --platform, e.g. linux/amd64 for Rosetta")
    args = p.parse_args()

    project_dir = args.out

    if args.stage in ("extract", "all"):
        print("\n[A] Extract frames + GPS → OpenSfM project")
        extract.write_project(args.video, args.gpx, project_dir,
                              sample_fps=args.fps)

    if args.stage in ("sfm", "all"):
        print("\n[B] OpenSfM (Docker)")
        stages = list(sfm._DEFAULT_STAGES)
        if args.no_dense:
            stages = [s for s in stages if s not in ("undistort", "compute_depthmaps")]
        sfm.run_opensfm(project_dir,
                        image=args.docker_image,
                        stages=stages,
                        platform=args.docker_platform)
        for k, v in sfm.collect_outputs(project_dir).items():
            print(f"  {k}: {v} (exists={v.exists()})")


if __name__ == "__main__":
    main()
