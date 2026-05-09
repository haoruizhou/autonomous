# Pipeline v2 — Photogrammetry Track Reconstruction

This document describes the current `pipeline_v2` workflow for reconstructing an autocross track from phone/dashcam video plus GPS.

## Current recommendation

Use one continuous video when possible. The merged NH1+NH2 video test succeeded at 2 fps:

- Input video: `NH_20260429125648Z.mp4`
- Input GPX: `NHautocross/nh_stitched_1fps.gpx`
- Result: 1189 / 1189 frames in one OpenSfM reconstruction component
- Sparse cloud preview: ~1.0M road/grass points

For future recordings, record the whole course in one shot. If stitching ever fails, use NH2 only; it is the longer/main-course clip and not worth extra merge engineering.

## Stage table

| Stage | Script | Input | Output | Notes |
|---|---|---|---|---|
| A extract | `pipeline_v2/extract.py` | video(s), GPX | `images/*.jpg`, `exif_overrides.json`, `frame_index.json`, `config.yaml` | Samples frames and injects per-frame GPS priors for OpenSfM. |
| A2 masks | `pipeline_v2/semantic.py` | raw `images/*.jpg` | `masks/<image>.jpg.png` | Optional OpenSfM feature masks. Currently skip by default; masking hurt low-texture pavement matching. |
| B SfM | `pipeline_v2/sfm.py` | OpenSfM project dir | `features/`, `matches/`, `tracks.csv`, `reconstruction.json` | Dockerized OpenSfM: metadata, SIFT, matching, tracks, reconstruct. |
| B undistort | `pipeline_v2/sfm.py` | `reconstruction.json` | `undistorted/images/`, `undistorted/reconstruction.json` | Needed before semantic/cloud. |
| B depthmaps | `pipeline_v2/sfm.py` | `undistorted/images/` | `undistorted/depthmaps/*.clean.npz`, `merged.ply` | CPU dense path via OpenSfM `compute_depthmaps`. This is required for reliable elevation. |
| B GPU dense | `pipeline_v2/sfm.py` | `undistorted/openmvs/scene.mvs` | `undistorted/openmvs/scene_dense.ply` | GPU path via OpenMVS/CUDA. Preferred on Linux RTX 3080. |
| C semantic | `pipeline_v2/semantic.py` | `undistorted/images/*.jpg` | `labels/*.npy`, `labels/*_vis.png` | Mask2Former-Cityscapes maps pixels to road/grass/removed. |
| D cloud | `pipeline_v2/cloud.py` | reconstruction, labels, depthmaps/PLY | `cloud.npz` | Produces labeled XYZ/RGB/class cloud. Sparse fallback is preview-only for elevation. |
| E cones | `pipeline_v2/cones.py` | images, reconstruction, depthmaps | `cones.json`, `cloud_with_cones.npz` | Cone detection/clustering/stamping. Skipped in local 2 fps preview. |
| F export | `pipeline_v2/export_json.py`, `pipeline_v2/export.py` | `cloud_with_cones.npz` | `track.json`, `track.obj`, `track.mtl` | Browser simulator and OBJ outputs. |
| F2 mesh | `pipeline_v2/mesh.py` | `cloud.npz` | `track_mesh.glb` | Optional Poisson mesh. May fail on noisy sparse clouds; OBJ/JSON are the primary outputs. |
| G diag | `pipeline_v2/diag.py` | `cloud.npz`, `track.json` | `diag/*.png` | QA plots: top-down classes, z histograms, height profile, track overview. |
| H sync | `pipeline_v2/run.py` | track assets | `frontend/public/data/*` | Copies latest assets for the frontend. |

## Important lessons from the Mac 2 fps run

### Merged video works

The 2 fps merged-video reconstruction produced one connected component:

```text
Reconstruction 0: 1189 images, 1274623 points
```

This validates the merged video + stitched GPX idea.

### Sparse fallback is good for shape, not elevation

The 2 fps local run skipped depthmaps (`--no-dense`). `cloud.py` therefore used sparse SfM points. That produced a correct-looking top-down road shape, but the elevation profile was poor:

- GPX elevation range: ~8.7 m
- Exported sparse road cloud p1→p99 Z range: ~47.7 m

Root cause: sparse SfM tie-points are not road-surface samples. Without depthmaps, points from background/objects can project onto road-labeled pixels and get classified as road. Use depthmaps/OpenMVS for any real elevation/mesh work.

### Feature masks are optional and usually disabled

A Mask2Former-based masks stage exists, but it should be skipped for this dataset. People/cars occupy a small image area but contain most SIFT features. Masking them removed too many usable features on otherwise low-texture pavement. RANSAC is a better filter for moving-object matches here.

## Recommended commands

### Local Mac smoke test: 2 fps, no dense

Use this only to validate connectivity and top-down shape.

```bash
uv run python -m pipeline_v2.run \
  --video NH_20260429125648Z.mp4 \
  --gpx NHautocross/nh_stitched_1fps.gpx \
  --out results/test_2fps_mac \
  --fps 2 \
  --no-dense \
  --skip masks \
  --skip dense_depthmap --skip cones --skip mesh --skip sync
```

If `--no-dense` skipped undistort, run:

```bash
docker run --rm \
  -v /Users/hz/GitHub/autonomous/results/test_2fps_mac:/project \
  -w /source/OpenSfM \
  opensfm:ubuntu24 bash -lc 'bin/opensfm undistort /project'
```

Then continue semantic/cloud/export/diag:

```bash
uv run python -m pipeline_v2.run \
  --video NH_20260429125648Z.mp4 \
  --gpx NHautocross/nh_stitched_1fps.gpx \
  --out results/test_2fps_mac \
  --fps 2 \
  --skip extract --skip masks --skip sfm \
  --skip dense_depthmap --skip cones --skip mesh --skip sync
```

### Full Linux run: 6 fps with GPU dense

Use this for the actual reconstruction result.

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

## Current OpenSfM tuning

`extract.py` writes the default OpenSfM config. Important values:

```yaml
feature_type: SIFT
feature_min_frames: 4000
feature_process_size: 2048
sift_peak_threshold: 0.04
matching_gps_distance: 100
matching_gps_neighbors: 24
matching_time_neighbors: 20
matching_use_filters: yes
lowes_ratio: 0.85
bundle_use_gps: yes
bundle_compensate_gps_bias: yes
depthmap_method: PATCH_MATCH_SAMPLE
depthmap_resolution: 640
depthmap_num_neighbors: 6
```

These settings restore the wide GPS matching window from the first usable product while increasing feature density for low-contrast pavement.
