import type { LoadedTrack, SourcePoint, TrackJson } from '../types/track';

export async function loadTrack(): Promise<LoadedTrack> {
  const res = await fetch('/data/track.json');
  if (!res.ok) throw new Error(`Failed to load track.json: ${res.status}`);
  const raw: TrackJson = await res.json();

  const roadSet = new Set<string>();
  const grassSet = new Set<string>();
  const roadHeight = new Map<string, number>();
  const grassHeight = new Map<string, number>();

  const key = (i: number, j: number) => `${i},${j}`;

  for (const [i, j, z] of raw.road_cells) {
    roadSet.add(key(i, j));
    roadHeight.set(key(i, j), z);
  }
  for (const [i, j, z] of raw.grass_cells) {
    grassSet.add(key(i, j));
    grassHeight.set(key(i, j), z);
  }

  // Origin: centroid of road cells (or first cell) in source coordinates.
  const { x0, y0, cell_m } = raw.grid;
  let ox = 0, oy = 0, oz = 0;
  const n = raw.road_cells.length || 1;
  for (const [i, j, z] of raw.road_cells) {
    ox += x0 + (i + 0.5) * cell_m;
    oy += y0 + (j + 0.5) * cell_m;
    oz += z;
  }
  const origin: SourcePoint = { x: ox / n, y: oy / n, z: oz / n };

  return {
    raw,
    renderRoadCells: raw.road_cells,
    visualRoadFillCells: raw.road_cells,
    renderGrassCells: raw.grass_cells,
    origin,
    roadSet,
    grassSet,
    roadHeight,
    grassHeight,
  };
}
