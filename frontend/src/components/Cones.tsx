import { useLayoutEffect, useRef } from 'react';
import * as THREE from 'three';
import type { LoadedTrack } from '../types/track';
import { sourceToThree } from '../lib/coordinates';

type Props = {
  track: LoadedTrack;
};

export default function Cones({ track }: Props) {
  const ref = useRef<THREE.InstancedMesh>(null);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    for (let index = 0; index < track.raw.cones.length; index += 1) {
      const cone = track.raw.cones[index];
      const [x, y, z] = sourceToThree({ x: cone.xyz[0], y: cone.xyz[1], z: cone.xyz[2] + 0.22 }, track.origin);
      dummy.position.set(x, y, z);
      dummy.rotation.set(0, 0, Math.PI);
      const scale = 0.8 + Math.min(0.5, cone.n_observations / 50);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      mesh.setColorAt(index, new THREE.Color('#ff8c00'));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [track]);

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, track.raw.cones.length]} castShadow receiveShadow>
      <coneGeometry args={[0.18, 0.44, 12]} />
      <meshStandardMaterial color="#ff8c00" emissive="#221000" vertexColors roughness={0.55} />
    </instancedMesh>
  );
}
