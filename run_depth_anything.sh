#!/usr/bin/env bash
# run_depth_anything.sh
# Runs Depth Anything 3 on NH1.mp4 using a UV-managed environment.
# Usage: bash run_depth_anything.sh [optional_video_path]
set -euo pipefail

VIDEO="${1:-/Users/hz/GitHub/autonomous/NHautocross/NH1.mp4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="${SCRIPT_DIR}/Depth-Anything-3"
DA3_VENV="${SCRIPT_DIR}/.venv-da3"
OUTPUT_DIR="${SCRIPT_DIR}/depth_output/$(date +%Y%m%d_%H%M%S)"

echo "=== Depth Anything 3 Runner ==="
echo "Video      : ${VIDEO}"
echo "DA3 dir    : ${DA3_DIR}"
echo "Output dir : ${OUTPUT_DIR}"
echo

# ── 1. Clone DA3 if needed ────────────────────────────────────────────────────
if [ ! -d "${DA3_DIR}" ]; then
  echo "[1/5] Cloning Depth-Anything-3..."
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "${DA3_DIR}"
else
  echo "[1/5] Depth-Anything-3 already cloned at ${DA3_DIR}"
fi

# ── 2. Create a dedicated UV venv for DA3 (needs numpy<2) ────────────────────
if [ ! -d "${DA3_VENV}" ]; then
  echo "[2/5] Creating UV venv at ${DA3_VENV}..."
  uv venv "${DA3_VENV}" --python 3.11
else
  echo "[2/5] UV venv already exists at ${DA3_VENV}"
fi

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/5] Installing DA3 dependencies (this may take a while on first run)..."
source "${DA3_VENV}/bin/activate"

# Install PyTorch for Apple Silicon (MPS) - use pip index for the right build
uv pip install --python "${DA3_VENV}" torch torchvision --quiet

# Install xformers (may fail on M-series; that's OK)
uv pip install --python "${DA3_VENV}" xformers --quiet || \
  echo "  ↳ xformers not available for this platform, continuing without it"

# Install DA3 core deps (note: requires numpy<2)
uv pip install --python "${DA3_VENV}" \
  "numpy<2" opencv-python pillow einops huggingface_hub imageio \
  trimesh omegaconf typer safetensors e3nn moviepy plyfile \
  pillow-heif requests --quiet

# Install DA3 package itself in editable mode
uv pip install --python "${DA3_VENV}" -e "${DA3_DIR}" --quiet

# ── 4. Create output directory ────────────────────────────────────────────────
echo "[4/5] Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ── 5. Run DA3 on the video ───────────────────────────────────────────────────
echo "[5/5] Running Depth Anything 3 on: ${VIDEO}"
echo "      Model: DA3MONO-LARGE (best for M2 — no xformers needed)"
echo "      Output: ${OUTPUT_DIR}"
echo

# Use DA3 CLI. DA3MONO-LARGE is the most M2-friendly monocular depth model.
# For metric depth, switch to DA3METRIC-LARGE.
"${DA3_VENV}/bin/python" -m depth_anything_3.cli video \
  "${VIDEO}" \
  --model-dir "depth-anything/DA3MONO-LARGE" \
  --fps 5 \
  --export-dir "${OUTPUT_DIR}" \
  --export-format "depth_vis" \
  2>&1 || {
    # Fallback: try the da3 entrypoint if the module approach doesn't work
    echo "Trying da3 CLI entrypoint..."
    "${DA3_VENV}/bin/da3" video \
      "${VIDEO}" \
      --model-dir "depth-anything/DA3MONO-LARGE" \
      --fps 5 \
      --export-dir "${OUTPUT_DIR}" \
      --export-format "depth_vis"
  }

echo
echo "✅ Done! Results saved to: ${OUTPUT_DIR}"
