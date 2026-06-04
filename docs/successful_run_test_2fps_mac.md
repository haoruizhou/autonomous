# Successful Run — `results/test_2fps_mac`

This is the first complete end-to-end pipeline run produced **on the Mac** (no GPU /
no Docker, CPU depthmaps). It is the known-good baseline: every stage from frame
extraction through the Assetto Corsa handoff completed, and the SfM solved into a
single fully-connected reconstruction.

Produced on branch `mac-fixes`. SfM finished 2026-05-09; cloud / cones / export /
mesh / AC handoff iterated through 2026-05-10.

## What "successful" means here

- **SfM converged to ONE component** containing all 1,189 shots — no fragmentation.
  This is the hard part for a phone-only autocross reconstruction and the main reason
  this run is worth keeping as a reference.
- Every downstream stage produced output (cloud → cones → export → mesh → AC handoff).

## Inputs

- Frames extracted at **2 fps** (hence `test_2fps_mac`) → 1,189 frames.
- GPS aligned per-frame from the autocross GPX.

## Reconstruction

| Metric | Value |
|--------|-------|
| Components | 1 (fully connected) |
| Shots (camera poses) | 1,189 |
| Sparse points | 1,274,623 |

## Stage outputs

| Stage | File(s) | Notes |
|-------|---------|-------|
| Extract | `frame_index.json`, `extract_summary.json`, `exif_overrides.json` | 1,189 frames + GPS |
| SfM | `reconstruction.json` (407 MB), `tracks.csv` (434 MB) | 1 component, 1,189 shots |
| Depthmaps | `compute_depthmaps.log` | CPU PATCH_MATCH path |
| Semantic | `labels/` (2,379 files) | road / grass / cone classes |
| Cloud | `cloud.npz` (78 MB) | labeled point cloud |
| Cones | `cone_detections.json`, `cones.json`, `cloud_with_cones.npz` | |
| Terrain | `cloud_terrain_with_cones.npz` | terrain-flattened variant |
| Export | `track.obj` (14 MB), `track.mtl`, `track.json` | road + grass mesh |
| Mesh | `track_mesh.glb` (2.6 MB) | Poisson, vertex-colored |
| AC handoff | `ac_handoff/` | `track_asphalt.obj`, `cones.csv`, `guide_path.csv`, `preview.png` |
| Diagnostics | `diag/`, `topdown.png` | z-hist, top-down classes, height profiles |

### Export numbers (from `export_step.log`)

- Road z-filter: median 120.4, tol ±18.3 m → 5,736,410 road points kept
- Grid: `road_cells=51,321`, `grass_cells=102,664`, `patched_road_holes=336`
- Grass mesh: 154,463 verts / 299,441 faces
- Road mesh: 63,849 verts / 120,181 faces

## Known caveat

`track.obj` exported with **`cones=0`** — cone detection ran and produced
`cones.json` (cones present), but cones were not baked into the OBJ mesh. The cones
do live in the AC handoff (`ac_handoff/cones.csv`, `ac_handoff/cones.json`), so the
Assetto Corsa path is unaffected. If cones are wanted in the photogrammetry OBJ
itself, that's the place to look.

## Frontend

The frontend serves this run's artifacts directly:

- `frontend/public/data/{track.obj, track.mtl, track.json, track_mesh.glb}`
  are copies of the `test_2fps_mac` outputs (synced 2026-05-10).
- Run `cd frontend && npm run dev` to view the track in the browser.

## Reproduce

This run was CPU-only on Mac. The equivalent invocation (adjust paths/fps):

```bash
.venv/bin/python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --out results/test_2fps_mac --fps 2
```
