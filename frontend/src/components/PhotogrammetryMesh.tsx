import { useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import { MTLLoader, OBJLoader } from 'three-stdlib';
import * as THREE from 'three';
import type { LoadedTrack } from '../types/track';
import { sourceToThree } from '../lib/coordinates';

type Props = {
  track: LoadedTrack;
};

export default function PhotogrammetryMesh({ track }: Props) {
  const materials = useLoader(MTLLoader, '/data/track.mtl');
  materials.preload();
  const obj = useLoader(OBJLoader, '/data/track.obj', (loader) => {
    materials.preload();
    loader.setMaterials(materials);
  });

  const scene = useMemo(() => {
    const offset = sourceToThree({ x: 0, y: 0, z: 0 }, track.origin);
    const cloned = obj.clone(true);
    cloned.position.set(offset[0], offset[1], offset[2]);
    cloned.scale.z = -1;
    cloned.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.receiveShadow = true;
        child.castShadow = child.name.toLowerCase().includes('cone');
        if (child.material instanceof THREE.MeshPhongMaterial || child.material instanceof THREE.MeshStandardMaterial) {
          child.material.side = THREE.DoubleSide;
        }
      }
    });
    return cloned;
  }, [obj, track]);

  return <primitive object={scene} />;
}
