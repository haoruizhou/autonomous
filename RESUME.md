# Resume on Windows PC — `results/sfm_full_v2`

Last updated: 2026-05-03.

## Current state on Mac

The 6 fps OpenSfM rerun is still running on the Mac in Docker, but it appears to be stuck/quiet during `reconstruct` after many hours. The container is alive and CPU-active, but no `reconstruction.json` checkpoint has been written yet.

Current live process details when this file was written:

- Docker container: `bold_wright` (`c996e3a7bc0f`)
- Command inside container:
  ```bash
  bin/opensfm create_tracks /project && bin/opensfm reconstruct /project && bin/opensfm undistort /project && bin/opensfm compute_depthmaps /project
  ```
- Log: `logs/run_v2_resume.log`
- Output/project dir: `results/sfm_full_v2`
- Last visible reconstruction log line:
  ```text
  2026-05-03 20:02:04,588 INFO: Adding NH2_012620.jpg to the reconstruction
  ```
- Last visible shot count:
  ```text
  Reconstruction now has 1262 shots
  ```
- Docker RAM has been high but not OOM at last checks, around `23.8 GiB / 27.36 GiB`.

## Important checkpoint status

The expensive track-merge stage completed successfully:

- `results/sfm_full_v2/tracks.csv` exists and is about `1.8G`.
- Whole `results/sfm_full_v2` directory is about `20G`.

No reconstruction checkpoint exists yet:

- `results/sfm_full_v2/reconstruction.json` does **not** exist as of this note.

Meaning: moving this to the PC preserves extraction, features, matches, and `tracks.csv`, but the PC will restart from the beginning of `reconstruct`, not from shot 1262.

## Why moving to PC may help

The Mac Docker VM has about 27.36 GiB RAM available and the run has repeatedly approached 90–95% memory during `reconstruct`.

A Windows desktop with 64 GB RAM should be a better fit for this 6 fps SfM project. The GPU is not the main factor for OpenSfM `create_tracks`/`reconstruct`; RAM is.

## What to transfer

Transfer this entire directory from Mac to Windows:

```text
results/sfm_full_v2/
```

This includes OpenSfM project data:

- `images/`
- `features/`
- `matches/`
- `tracks.csv`
- `exif_overrides.json`
- `frame_index.json`
- `config.yaml`
- other OpenSfM outputs already produced before `reconstruct`

Safest is to copy the whole folder, not cherry-pick files.

## Suggested LAN transfer methods

### Option A — rsync over SSH, best if available

On Windows, enable OpenSSH Server and find the Windows IP.

On Mac:

```bash
rsync -avh --progress results/sfm_full_v2/ WINDOWS_USER@WINDOWS_IP:/c/Users/WINDOWS_USER/Desktop/sfm_full_v2/
```

`rsync` can resume if interrupted.

### Option B — scp, simpler but weaker resume

```bash
scp -r results/sfm_full_v2 WINDOWS_USER@WINDOWS_IP:/c/Users/WINDOWS_USER/Desktop/
```

### Option C — LocalSend GUI

Install LocalSend on both machines and send the `results/sfm_full_v2` folder over LAN. This is easier but less robust for resume than `rsync`.

## Resume command on Windows PC

Use Docker with the same OpenSfM image (`opensfm:ubuntu24`) if available on the PC.

From the PC, mount the transferred project directory and run from `reconstruct` onward:

```bash
docker run --rm \
  -v C:/Users/WINDOWS_USER/Desktop/sfm_full_v2:/project \
  -w /source/OpenSfM \
  opensfm:ubuntu24 \
  bash -lc "bin/opensfm reconstruct /project && bin/opensfm undistort /project && bin/opensfm compute_depthmaps /project"
```

If running from PowerShell, this single-line form may be easier:

```powershell
docker run --rm -v C:/Users/WINDOWS_USER/Desktop/sfm_full_v2:/project -w /source/OpenSfM opensfm:ubuntu24 bash -lc "bin/opensfm reconstruct /project && bin/opensfm undistort /project && bin/opensfm compute_depthmaps /project"
```

If the image does not exist on the PC, rebuild or load the same `opensfm:ubuntu24` image used on the Mac before running the command.

## After OpenSfM finishes on PC

Copy/sync the completed `sfm_full_v2` directory back into this repo layout if needed, then run the remaining Python stages from the repo root:

```bash
uv run python -m pipeline_v2.run \
  --video NHautocross/NH1.mp4 \
  --video NHautocross/NH2.mp4 \
  --gpx NHautocross/autocross.gpx \
  --out results/sfm_full_v2 \
  --fps 6 \
  --skip extract \
  --skip sfm
```

That runs:

```text
semantic → cloud → cones → export → diag → frontend sync
```

Expected final files include:

- `results/sfm_full_v2/cloud.npz`
- `results/sfm_full_v2/cloud_with_cones.npz`
- `results/sfm_full_v2/cones.json`
- `results/sfm_full_v2/track.json`
- `results/sfm_full_v2/track.obj`
- `results/sfm_full_v2/track.mtl`
- `results/sfm_full_v2/diag/*.png`
- synced frontend assets at `frontend/public/data/track.{json,obj,mtl}`

## What to check after PC `reconstruct`

The main question is whether the two arcs / two-video split is fixed.

After `reconstruct`, inspect:

1. Whether `reconstruction.json` exists.
2. Whether it contains one reconstruction or multiple components.
3. Whether both `NH1_...` and `NH2_...` shots are in the same component.
4. Whether diagnostics later show the old ~4 m vertical offset is gone.

Useful quick check once `reconstruction.json` exists:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/sfm_full_v2/reconstruction.json')
recs = json.loads(p.read_text())
print('components:', len(recs))
for i, rec in enumerate(recs):
    shots = rec.get('shots', {})
    nh1 = sum(1 for s in shots if s.startswith('NH1_'))
    nh2 = sum(1 for s in shots if s.startswith('NH2_'))
    print(i, 'shots', len(shots), 'NH1', nh1, 'NH2', nh2)
PY
```

Good sign:

- `components: 1`
- same component has both `NH1 > 0` and `NH2 > 0`

Bad/old-pattern sign:

- separate components, one mostly `NH1`, one mostly `NH2`

## Notes

- Do not delete the Mac `results/sfm_full_v2` until the PC run has produced `reconstruction.json` and later artifacts.
- If the Mac run eventually finishes, it may still be useful to compare with the PC run.
- If restarting from scratch becomes necessary, consider a lower-memory config: 4 fps, or 6 fps with fewer features (`feature_process_size: 1600`, `feature_min_frames: 1200–1600`, fewer GPS neighbors).
