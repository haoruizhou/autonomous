# Photogrammetry Pipeline — Mac Test Session (2026-05-08/09)

## What we did

Validated the pipeline on a MacBook Pro using the merged NH1+NH2 video
(`NH_20260429125648Z.mp4`) and a stitched GPX (`NHautocross/nh_stitched_1fps.gpx`).

### Key results

| Run | fps | Frames | Components | Points | Notes |
|-----|-----|--------|------------|--------|-------|
| 1fps smoke | 1 | 595 | 8 | 11k sparse | Masking crippled features |
| **2fps merge test** | **2** | **1189** | **1** | **1.27M** | Merge works ✓ |

The merged video stitches cleanly — all 1189 frames land in a single SfM
component with no seam at the NH1→NH2 boundary.

---

## Pipeline changes this session

### 1. Person/vehicle feature masking (`semantic.py`, `extract.py`, `run.py`)

Added `generate_opensfm_masks()` to `semantic.py`: runs Mask2Former on raw
`images/` before SfM and writes `masks/<name>.jpg.png` (white = masked).
OpenSfM suppresses SIFT keypoints in masked regions via `use_masks: yes`.

**Naming fix:** OpenSfM expects `masks/000000.jpg.png` (image filename + `.png`),
not `masks/000000.png`. Fixed in `generate_opensfm_masks()`.

Added `masks` as stage A2 in `run.py` (between extract and sfm). Opt out with
`--skip masks`.

**Note:** Masking turned out to hurt more than help on this dataset. The road
surface is low-texture asphalt — after masking 87–97% of SIFT features (which
sit on people/cars), the surviving features are nearly all featureless pavement.
`use_masks` is **not** in the default config. Use `--skip masks` (or omit the
stage) for normal runs. RANSAC handles moving objects naturally.

### 2. OpenSfM config tuning (`extract.py`)

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `sift_peak_threshold` | 0.066 | 0.04 | More features on low-contrast pavement |
| `feature_min_frames` | 2000 | 4000 | Try harder per frame |
| `matching_gps_distance` | 60m | 100m | Restored from original working config |
| `use_masks` | yes | removed | Masking hurts on low-texture surfaces |

### 3. Sparse cloud fallback fix (`cloud.py`)

OpenSfM strips observations from both `reconstruction.json` and
`undistorted/reconstruction.json` output files. The sparse fallback in
`cloud.py` previously relied on observations to build the shot→point index
and produced zero points.

Fix: when observations are empty, project all 3D points into every camera
directly (brute-force, vectorized numpy). Additionally, read original
`reconstruction.json` for points+observations when available, and use
undistorted poses from `undistorted/reconstruction.json`.

---

## What didn't work

- **1fps + masking**: Produced 8 fragmented components. Masking removed
  features from people/cars which had the highest SIFT density, leaving nearly
  nothing on flat pavement.
- **`--no-dense` + cloud**: Cloud assembly fails without undistorted images.
  Workaround: run `bin/opensfm undistort` in Docker manually, then continue.

---

## Decision: video strategy

The merged video approach works. For future sessions:

- **Preferred**: single continuous recording covering the full course.
- **Fallback**: NH2 only (the longer, main-course video). NH1 is a warm-up
  and not worth the stitching complexity if the merge ever breaks.

---

## Next: full 6fps run on Linux

Config ready. Run on the Linux box (24-core, RTX 3080):

```bash
cd /home/hz/autonomous
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH_20260429125648Z.mp4 \
  --gpx NHautocross/nh_stitched_1fps.gpx \
  --out results/sfm_merged_6fps \
  --fps 6 \
  --skip masks \
  --gpu
```

Expected: ~3,570 frames, single component, dense OpenMVS cloud (~5-18M pts).
