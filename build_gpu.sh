#!/usr/bin/env bash
# Build the GPU-accelerated OpenSfM+OpenMVS image.
# Run this after the CPU reconstruction finishes so the build doesn't compete for RAM.
#
# Usage:  bash build_gpu.sh [--no-cache]
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/2] Building opensfm:ubuntu24_cuda (OpenSfM + OpenMVS/CUDA) …"
echo "      This clones OpenMVS from GitHub and compiles with nvcc — expect 15-30 min."

docker build \
    -f Dockerfile.gpu \
    -t opensfm:ubuntu24_cuda \
    "$@" \
    .

echo "[2/2] Verifying GPU access …"
docker run --rm --gpus all opensfm:ubuntu24_cuda \
    bash -c "nvidia-smi | head -4 && DensifyPointCloud --version 2>&1 | head -2"

echo ""
echo "Image ready: opensfm:ubuntu24_cuda"
echo ""
echo "To run the full GPU pipeline on a new dataset:"
echo "  uv run python -m pipeline_v2.run \\"
echo "    --video <clip1.mp4> --video <clip2.mp4> \\"
echo "    --gpx <track.gpx> \\"
echo "    --out <results_dir> \\"
echo "    --fps 6 --gpu"
echo ""
echo "To run only depthmaps on an existing reconstructed project (skip sfm):"
echo "  docker run --rm --gpus all \\"
echo "    -v <project_dir>:/project \\"
echo "    -w /source/OpenSfM \\"
echo "    opensfm:ubuntu24_cuda bash -lc \\"
echo "    'bin/opensfm export_openmvs /project && DensifyPointCloud /project/undistorted/openmvs/scene.mvs --cuda-device 0 --resolution-level 1'"
