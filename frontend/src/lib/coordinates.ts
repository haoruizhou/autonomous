import * as THREE from 'three';
import type { SourcePoint } from '../types/track';

// Source frame: x,y horizontal; z vertical (up).
// Three.js frame: x,z horizontal; y vertical (up).
// GridMesh/PhotogrammetryMesh apply scale.z = -1 on loaded geometry,
// so source.y maps to -three.z in the scene.

export function sourceToThree(
  p: { x: number; y: number; z: number },
  origin: SourcePoint,
): [number, number, number] {
  return [p.x - origin.x, p.z - origin.z, -(p.y - origin.y)];
}

export function threeToSource(
  p: THREE.Vector3,
  origin: SourcePoint,
): { x: number; y: number; z: number } {
  return { x: p.x + origin.x, y: origin.y - p.z, z: p.y + origin.z };
}
