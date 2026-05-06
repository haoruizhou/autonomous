import * as THREE from 'three';

// ConeGeometry points up (+Y). Lying flat = tilt 90° around Z, then spin by yaw around Y.
export function coneLyingAlongYaw(yaw: number): THREE.Quaternion {
  const tilt = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
  const spin = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
  return spin.multiply(tilt);
}
