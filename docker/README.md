# CPU Docker image

Single self-contained image that runs the whole pipeline on CPU.

## Build
```bash
docker build -t autonomous:cpu -f Dockerfile.cpu .
```

## Run (one command)
```bash
docker run --rm \
  -v "$PWD":/in:ro -v /tmp/out:/out \
  autonomous:cpu \
  --video /in/NH_20260429125648Z.mp4 \
  --gpx /in/NHautocross/autocross.gpx \
  --out /out --fps 2 \
  --cone-model /opt/models/yolov8s-world.pt
```
Outputs (track.obj, track.json, track_mesh.glb, ac_handoff/, diag/) appear in /tmp/out.

## Notes
- OpenSfM runs natively inside the image (no docker-in-docker); `--sfm-mode auto`
  detects this via `AUTONOMOUS_IN_CONTAINER`.
- Models (YOLO + Mask2Former) are baked in; runs work offline (`--network none`).
- Dense depthmaps are slow on CPU; add `--no-dense` for a faster run.
- The GPU path (`Dockerfile.gpu`, OpenMVS/CUDA) is separate and unchanged.
