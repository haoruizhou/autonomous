import * as THREE from 'three';
import type { SourcePoint, TrackCell, TrackGrid } from '../types/track';
import { sourceToThree } from './coordinates';

export function buildCellSurfaceGeometry(
  cells: TrackCell[],
  grid: TrackGrid,
  origin: SourcePoint,
  yOffset: number,
): THREE.BufferGeometry {
  const { x0, y0, cell_m } = grid;
  const positions: number[] = [];
  const normals: number[] = [];

  for (const [i, j, z] of cells) {
    const sx = x0 + i * cell_m;
    const sy = y0 + j * cell_m;
    const sz = z + yOffset;

    const corners = [
      sourceToThree({ x: sx,           y: sy,           z: sz }, origin),
      sourceToThree({ x: sx + cell_m,  y: sy,           z: sz }, origin),
      sourceToThree({ x: sx + cell_m,  y: sy + cell_m,  z: sz }, origin),
      sourceToThree({ x: sx,           y: sy + cell_m,  z: sz }, origin),
    ];

    for (const idx of [0, 1, 2, 0, 2, 3]) {
      const [tx, ty, tz] = corners[idx];
      positions.push(tx, ty, tz);
      normals.push(0, 1, 0);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geo;
}

export function buildCenterlineGeometry(
  centerline: [number, number, number][],
  origin: SourcePoint,
): THREE.BufferGeometry {
  const points = centerline.map(([x, y, z]) => {
    const [tx, ty, tz] = sourceToThree({ x, y, z }, origin);
    return new THREE.Vector3(tx, ty + 0.05, tz);
  });
  return new THREE.BufferGeometry().setFromPoints(points);
}
