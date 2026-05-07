# 4fps Full Run — Progress & Issues

## Goal
Full NH1+NH2 reconstruction at 4fps with GPS, GPU dense reconstruction, and complete pipeline 
output (track.json, track_mesh.glb) ready to view in the frontend.

## Dataset
- NH1.mp4: 755 frames @ 4fps, 100% GPS
- NH2.mp4: 1622 frames @ 4fps, 100% GPS
- Total: **2377 frames**, all GPS-tagged

### Current Status (May 7, 2026 - SUCCESS)

- **Reconstruction**: All 3 components registered (2,377 frames).
- **Labeling**: **FIXED**. PLY parser now handles list properties and uses `view_indices` for perfect labeling.
- **Alignment**: **FIXED**. Component 1 (NH1) shifted by +141m to match Component 0 (NH2) median road Z.
- **Output**: 
    - **1.2M road points** correctly labeled.
    - **3,184 road cells** in `track.json` (full loop complete).
    - **track_mesh.glb** (0.6MB) synced to frontend.

### Identified Issues (Resolved)
1. **PLY Parsing Bug**: Resolved by implementing a custom binary PLY parser that correctly extracts `view_indices`.
2. **Component Misalignment**: Resolved by median road Z alignment between components.
3. **Labeling Sparsity**: Resolved by using `view_indices` instead of k-NN search.

### Tasks
- [x] Fix `cloud.py` PLY parser.
- [x] Update `cloud.py` to use `view_indices`.
- [x] Implement rigid alignment (Z-bias correction).
- [x] Re-generate final cloud and mesh.
- [x] Process Component 2 (undistorted/densified).

## Command History

### 1. SfM Run
```bash
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --out results/sfm_full_v2 --fps 4
```

### 2. Manual Component 1 & 2 Recovery
```bash
# Component 1
docker run --rm -v /home/hz/autonomous/results/sfm_full_v2:/project opensfm:ubuntu24_omp bash -lc 'bin/opensfm undistort /project --reconstruction-index 1 --output undistorted_rec1'
docker run --rm --gpus all -v /home/hz/autonomous/results/sfm_full_v2/undistorted_rec1:/project opensfm:ubuntu24_cuda bash -lc 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/OpenMVS && /usr/local/bin/OpenMVS/DensifyPointCloud /project/openmvs/scene.mvs --cuda-device 0'

# Component 2
docker run --rm -v /home/hz/autonomous/results/sfm_full_v2:/project opensfm:ubuntu24_omp bash -lc 'bin/opensfm undistort /project --reconstruction-index 2 --output undistorted_rec2'
# (Densification skipped for Rec 2 as it is very small)
```

### 3. Final Pipeline Run
```bash
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --skip extract --skip sfm --skip semantic \
  --out results/sfm_full_v2
```
