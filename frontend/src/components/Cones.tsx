import { useLayoutEffect, useRef } from 'react';
import * as THREE from 'three';
import type { LoadedTrack } from '../types/track';
import { sourceToThree } from '../lib/coordinates';
import { groundHeight } from '../lib/vehicle';

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
      const zSource = groundHeight(track, { x: cone.xyz[0], y: cone.xyz[1], z: cone.xyz[2] });
      const [x, y, z] = sourceToThree({ x: cone.xyz[0], y: cone.xyz[1], z: zSource + 0.44 }, track.origin);
      dummy.position.set(x, y, z);
      dummy.rotation.set(0, 0, 0);
      const scale = 1.6 + Math.min(1.0, cone.n_observations / 25);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [track]);

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, track.raw.cones.length]} castShadow receiveShadow>
      <coneGeometry args={[0.22, 0.88, 16]} />
      <meshStandardMaterial color="#ff8c00" emissive="#331400" roughness={0.55} />
    </instancedMesh>
  );
}
