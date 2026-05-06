import * as THREE from 'three';
import type { LoadedTrack, TrackCone } from '../types/track';
import { sourceToThree, threeToSource } from './coordinates';

export type CameraMode = 'follow' | 'orbit';
export type SurfaceType = 'road' | 'grass' | 'outside';

export type InputState = {
  throttle: boolean;
  brake: boolean;
  left: boolean;
  right: boolean;
  hardBrake: boolean;
};

export type VehicleState = {
  position: THREE.Vector3;
  heading: number;
  speed: number;
};

const CONE_HIT_RADIUS = 0.9;

function cellKey(i: number, j: number) { return `${i},${j}`; }

function getSurface(pos: THREE.Vector3, track: LoadedTrack): SurfaceType {
  const src = threeToSource(pos, track.origin);
  const { x0, y0, cell_m } = track.raw.grid;
  const i = Math.floor((src.x - x0) / cell_m);
  const j = Math.floor((src.y - y0) / cell_m);
  const k = cellKey(i, j);
  if (track.roadSet.has(k)) return 'road';
  if (track.grassSet.has(k)) return 'grass';
  return 'outside';
}

export function groundHeight(src: { x: number; y: number; z: number }, track: LoadedTrack): number {
  const { x0, y0, cell_m } = track.raw.grid;
  const i = Math.floor((src.x - x0) / cell_m);
  const j = Math.floor((src.y - y0) / cell_m);
  const k = cellKey(i, j);
  return track.roadHeight.get(k) ?? track.grassHeight.get(k) ?? src.z;
}

function getGroundY(pos: THREE.Vector3, track: LoadedTrack): number {
  const src = threeToSource(pos, track.origin);
  const { x0, y0, cell_m } = track.raw.grid;
  const i = Math.floor((src.x - x0) / cell_m);
  const j = Math.floor((src.y - y0) / cell_m);
  const k = cellKey(i, j);
  const z = track.roadHeight.get(k) ?? track.grassHeight.get(k) ?? src.z;
  const [, ty] = sourceToThree({ x: src.x, y: src.y, z }, track.origin);
  return ty;
}

export function createInitialVehicle(track: LoadedTrack): VehicleState {
  const cells = track.raw.road_cells;
  if (cells.length === 0) {
    return { position: new THREE.Vector3(0, 0.3, 0), heading: 0, speed: 0 };
  }
  const [i, j, z] = cells[Math.floor(cells.length / 4)];
  const { x0, y0, cell_m } = track.raw.grid;
  const [tx, ty, tz] = sourceToThree(
    { x: x0 + (i + 0.5) * cell_m, y: y0 + (j + 0.5) * cell_m, z },
    track.origin,
  );
  return { position: new THREE.Vector3(tx, ty + 0.3, tz), heading: 0, speed: 0 };
}

export function updateVehicle(
  state: VehicleState,
  input: InputState,
  track: LoadedTrack,
  dt: number,
): { state: VehicleState; surface: SurfaceType } {
  const surface = getSurface(state.position, track);
  const maxSpeed = surface === 'road' ? 22 : 12;
  const drag = surface !== 'road' ? 8 : 4;

  let { speed, heading } = state;

  if (input.hardBrake) {
    speed = Math.max(0, speed - 30 * dt);
  } else if (input.throttle && !input.brake) {
    speed = Math.min(maxSpeed, speed + 14 * dt);
  } else if (input.brake && !input.throttle) {
    speed = Math.max(0, speed - 18 * dt);
  } else {
    speed = Math.max(0, speed - drag * dt);
  }

  if (speed > 0.5) {
    heading += ((input.left ? 1 : 0) - (input.right ? 1 : 0)) * 2.0 * dt;
  }

  const newPos = state.position.clone().addScaledVector(
    new THREE.Vector3(Math.sin(heading), 0, Math.cos(heading)),
    speed * dt,
  );
  newPos.y = getGroundY(newPos, track) + 0.3;

  return { state: { position: newPos, heading, speed }, surface };
}

export function countConeHits(
  cones: TrackCone[],
  source: { x: number; y: number; z: number },
  hitIds: Set<number>,
): number {
  let added = 0;
  for (const cone of cones) {
    if (hitIds.has(cone.id)) continue;
    if (Math.hypot(cone.xyz[0] - source.x, cone.xyz[1] - source.y) < CONE_HIT_RADIUS) {
      hitIds.add(cone.id);
      added++;
    }
  }
  return added;
}
