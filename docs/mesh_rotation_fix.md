# Mesh rotation fix

The v2 2fps mesh/frontend mismatch came from inconsistent coordinate transforms between the GLB mesh path and the browser overlay path.

Source data uses an OpenSfM/GPS-aligned ENU-like frame:

- `x`, `y`: horizontal ground plane
- `z`: vertical/up

Three.js uses `y` as vertical/up, so source coordinates need to enter the scene as:

```text
source (x, y, z) -> three (x, z, -y)
```

## Root cause

`pipeline_v2/mesh.py` exported `track_mesh.glb` in a different basis than the frontend grid/centerline overlay. One attempted GLB transform mapped source height into a horizontal frontend axis, which made the photogrammetry mesh appear rotated relative to `track.json`.

After correcting the GLB export, the frontend still had a mirrored overlay because `sourceToThree()` mapped source `y` to positive Three.js `z`, while loaded GLB/OBJ meshes are rendered with `scale.z = -1`.

## Fix

- `pipeline_v2/mesh.py`
  - `source_to_glb_vertices()` exports GLB vertices as `(x, z, y)` before the frontend Z flip.
- `frontend/src/lib/coordinates.ts`
  - `sourceToThree()` maps source points to `(x - origin.x, z - origin.z, -(y - origin.y))`.
  - `threeToSource()` is the inverse: `(x + origin.x, origin.y - z, y + origin.z)`.

This makes direct overlays, centerline, cones, picking, vehicle logic, and loaded mesh assets share the same scene convention.

## Verification

Commands run after the fix:

```bash
uv run pytest tests/test_pipeline_v2_geo.py -q
npm --prefix frontend run typecheck
```

The regenerated 2fps GLB was synced to:

```text
frontend/public/data/track_mesh.glb
```

Frontend-space orientation check:

```text
mesh frontend xz span/angle       [226.168 201.869] -142.58
centerline frontend xz span/angle [215.116 163.323] -143.04
```
