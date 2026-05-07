# 4fps Full Run — Progress & Issues

## Goal
Full NH1+NH2 reconstruction at 4fps with GPS, GPU dense reconstruction, and complete pipeline 
output (track.json, track_mesh.glb) ready to view in the frontend.

## Dataset
- NH1.mp4: 755 frames @ 4fps, 100% GPS
- NH2.mp4: 1622 frames @ 4fps, 100% GPS
- Total: **2377 frames**, all GPS-tagged

## Current Status (May 7, 2026)

### Reconstruction
- **Component 0**: 1,622 shots (mostly NH2). Undistorted, densified (1.3GB PLY), and labeled.
- **Component 1**: 697 shots (mostly NH1). Undistorted, densified (680MB PLY), and labeled.
- **Component 2**: 58 shots. Registered but not yet undistorted/densified.
- **Total Registered**: 2,377 frames.

### Identified Issues
1. **PLY Parsing Bug (CRITICAL)**: `cloud.py`'s `_read_ply_xyz_rgb` does not handle PLY "list" properties. OpenMVS `scene_dense.ply` includes `view_indices` and `view_weights` as lists. The parser currently reads these as if they were vertex coordinates/colors, leading to:
    - Garbage XYZ values (extreme outliers, e.g., `1e33`).
    - Extremely low labeling rate (~0.33%) because garbage points don't project onto cameras.
    - *Opportunity*: These `view_indices` actually contain the exact camera IDs that saw each point. We can use them for perfect labeling!
2. **Component Misalignment**: Verified ~100m vertical (Z) shift between Component 0 (NH2) and Component 1 (NH1). bundle_compensate_gps_bias likely applied different offsets to each component.
3. **MVS Outliers**: Densification produces some "floater" points (triangulation artifacts) which require Z-filtering in `export_json.py`.

### Tasks
- [ ] Fix `cloud.py` PLY parser to handle/skip list properties.
- [ ] Update `cloud.py` to use `view_indices` from PLY for labeling (replaces k-NN search).
- [ ] Implement rigid alignment (Z-bias correction) between components.
- [ ] Process Component 2.
- [ ] Re-generate final cloud and mesh.

## Command History

### 1. SfM Run
```bash
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --out results/sfm_full_v2 --fps 4
```

### 2. Manual Component 1 Recovery
```bash
# Undistort
docker run --rm -v /home/hz/autonomous/results/sfm_full_v2:/project opensfm:ubuntu24_omp bash -lc 'bin/opensfm undistort /project --reconstruction-index 1 --output undistorted_rec1'

# Densify
docker run --rm --gpus all -v /home/hz/autonomous/results/sfm_full_v2/undistorted_rec1:/project opensfm:ubuntu24_cuda bash -lc 'DensifyPointCloud /project/openmvs/scene.mvs --cuda-device 0'
```

### 3. Labeling (Interrupted/Sparse)
```bash
.venv/bin/python -m pipeline_v2.run --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 --skip extract --skip sfm
```
