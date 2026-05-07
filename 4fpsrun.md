# 4fps Full Run — Progress & Issues

## Goal
Full NH1+NH2 reconstruction at 4fps with GPS, GPU dense reconstruction, and complete pipeline
output (track.json, track_mesh.glb) ready to view in the frontend.

## Dataset
- NH1.mp4: 755 frames @ 4fps, 100% GPS
- NH2.mp4: 1622 frames @ 4fps, 100% GPS
- Total: **2377 frames**, all GPS-tagged
- GPX: 2026-04-29 12:31:35 → 13:07:47 UTC (covers both videos)

## Commands Run

### 1. Clean previous 6fps artifacts
```bash
docker run --rm -v results/sfm_full_v2:/project opensfm:ubuntu24_omp \
  bash -c 'rm -rf /project/* /project/.[^.]* 2>/dev/null'
```

### 2. Extract frames at 4fps
```bash
.venv/bin/python -m pipeline_v2.extract \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --project results/sfm_full_v2 --fps 4
```
*Note: required `sudo apt-get install -y ffmpeg` first — imageio_ffmpeg bundles ffmpeg but not
ffprobe; `video_creation_time()` silently returned None without system ffprobe installed.*

### 3. OpenSfM reconstruction (CPU, ~18 hrs)
```bash
docker run -d --name opensfm_full \
  -v results/sfm_full_v2:/project \
  -v reconstruction_ckpt.py:/source/OpenSfM/opensfm/reconstruction.py:ro \
  -w /source/OpenSfM opensfm:ubuntu24_omp bash -lc \
  'bin/opensfm extract_metadata /project && \
   bin/opensfm detect_features /project && \
   bin/opensfm match_features /project && \
   bin/opensfm create_tracks /project && \
   bin/opensfm reconstruct /project && \
   bin/opensfm undistort /project && \
   bin/opensfm export_openmvs /project'
```
*Config: processes=16 (reduced from 24 to avoid WSL OOM during large global BAs)*

**Global BA timing (grew with dataset size):**
- 1038 shots: 37 min
- 1292 shots: 72 min
- 1292 shots: 73 min (retriangulate + another BA)
- 1593 shots: 154 min
- 1593 shots: 143 min
- 1622 shots: 154 min

**Result:** 3 reconstruction components — component 0: 1622 shots, component 1: 697 shots,
component 2: 58 shots. `undistort` + `export_openmvs` only ran on component 0.

### 4. GPU densify — component 0
```bash
docker run -d --name densify_full --gpus all \
  -v results/sfm_full_v2:/project \
  -e LD_LIBRARY_PATH=/usr/local/lib/OpenMVS \
  opensfm:ubuntu24_cuda bash -lc \
  '/usr/local/bin/OpenMVS/DensifyPointCloud /project/undistorted/openmvs/scene.mvs \
     --cuda-device 0 --resolution-level 1'
```
*Output: `undistorted/openmvs/scene_dense.ply` (1.3 GB, ~19M points)*
*Depth maps: 1622 frames processed in 3m 23s at 1024px resolution*

### 5. First pipeline run (failed — only component 0)
```bash
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --out results/sfm_full_v2 --fps 4 \
  --skip extract --skip sfm \
  --cloud-stride 1 --mesh-poisson-depth 11
```
*Result: 99 road cells, 0 cones — only component 0 was processed*

### 6. Undistort component 1
```bash
# OpenSfM undistort only does component 0 by default; use --reconstruction-index
docker run --rm -v results/sfm_full_v2:/project -w /source/OpenSfM opensfm:ubuntu24_omp bash -lc \
  'bin/opensfm undistort --reconstruction-index 1 --output undistorted_rec1 /project'
```
*Output: `undistorted_rec1/` with 697 undistorted frames + reconstruction.json*

### 7. Export OpenMVS for component 1 (via temp project)
```bash
# export_openmvs has no --reconstruction-index flag; create temp project
docker run --rm -v results/sfm_full_v2:/project -w /source/OpenSfM opensfm:ubuntu24_omp bash -lc '
  mkdir -p /project/comp1
  cp /project/config.yaml /project/comp1/
  cp /project/camera_models.json /project/comp1/
  ln -s /project/undistorted_rec1/images /project/comp1/images
  cp /project/undistorted_rec1/reconstruction.json /project/comp1/reconstruction.json
  cp /project/undistorted_rec1/tracks.csv /project/comp1/tracks.csv
  bin/opensfm undistort /project/comp1 && bin/opensfm export_openmvs /project/comp1'
```

### 8. GPU densify — component 1
```bash
docker run --rm --gpus all \
  -v results/sfm_full_v2:/project \
  -e LD_LIBRARY_PATH=/usr/local/lib/OpenMVS \
  opensfm:ubuntu24_cuda bash -lc \
  '/usr/local/bin/OpenMVS/DensifyPointCloud /project/comp1/undistorted/openmvs/scene.mvs \
     --cuda-device 0 --resolution-level 1'
```
*Output: `comp1/undistorted/openmvs/scene_dense.ply` (680 MB)*

### 9. Wire comp1 dense PLY into standard pipeline structure
```bash
docker run --rm -v results/sfm_full_v2:/project opensfm:ubuntu24_cuda bash -c \
  'mkdir -p /project/undistorted_rec1/openmvs && \
   cp /project/comp1/undistorted/openmvs/scene_dense.ply /project/undistorted_rec1/openmvs/ && \
   chmod -R a+rX /project/undistorted_rec1/'
```

### 10. Re-run semantic labeling for component 1
```bash
# Semantic labels already exist for component 0 (labels/)
# labels_rec1/ needs to be created
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --out results/sfm_full_v2 --fps 4 \
  --skip extract --skip sfm --skip semantic \  # TODO: semantic already done for comp0; need comp1 labels
  --cloud-stride 1 --mesh-poisson-depth 11
```
**⚠ CURRENTLY RUNNING** (`b46lxroaf`)

## Issues Encountered & Fixes

| Issue | Fix |
|-------|-----|
| WSL OOM during global BA at 24 processes | Reduced `processes: 16` in config.yaml |
| Container crashed WSL with `--restart unless-stopped` | Removed auto-restart policy |
| `ffprobe` missing → GPS coverage = 0 | `sudo apt-get install -y ffmpeg` (system ffprobe) |
| Corrupted .npz features from interrupted run | `sfm.py` now calls `_purge_corrupt_intermediate_files()` before Docker |
| BadZipFile crash in feature detection | Same fix as above |
| Docker outputs owned by root → PermissionError | `sfm.py` runs `chmod a+rX /project` after every container |
| NaN/Inf in OpenMVS PLY → KDTree crash | Filter added in `_read_ply_xyz_rgb()` |
| `export_openmvs` has no `--reconstruction-index` flag | Create temp project per component (workaround) |
| Only component 0 undistorted by default | Use `--reconstruction-index 1 --output undistorted_rec1` |
| DensifyPointCloud VRAM OOM at `--resolution-level 0` | Default changed to level 1 in `run_opensfm_gpu()` |

## Current Status
- ✅ Component 0: 1622 shots, 1.3 GB dense PLY
- ✅ Component 1: 697 shots, 680 MB dense PLY  
- ✅ Component 2: 58 shots (skipped — too small)
- ✅ Semantic labels: component 0 done (labels/)
- ⚠ Semantic labels: component 1 **missing** (labels_rec1/ not yet created)
- 🔄 Cloud + mesh: **currently running** with both dense PLYs (skip semantic)

## Still To Do
1. **Check cloud output** — expect 10k+ road cells (vs 99 from single component)
2. **Semantic labels for component 1** — re-run semantic stage for `undistorted_rec1`
3. **Inspect mesh in frontend** at http://localhost:5174
4. **Check cone detection** — 0 cones so far; may need cone visibility tuning
5. **Component 3 (58 shots)** — likely covers a gap; handle if needed
