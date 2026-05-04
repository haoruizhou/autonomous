import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { LoadedTrack } from '../types/track';
import { sourceToThree } from '../lib/coordinates';

type Props = {
  track: LoadedTrack;
  /** Public URL to the vertex-colored GLB (default: synced by pipeline ``mesh`` stage). */
  url?: string;
};

function applyShadowAndSide(root: THREE.Object3D): void {
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.receiveShadow = true;
    child.castShadow = false;
    const hasVertexColors = !!child.geometry?.attributes?.color;
    const mats = Array.isArray(child.material) ? child.material : [child.material];
    for (const m of mats) {
      m.side = THREE.DoubleSide;
      if (hasVertexColors && 'vertexColors' in m) {
        (m as THREE.MeshStandardMaterial).vertexColors = true;
      }
      if (m instanceof THREE.MeshStandardMaterial || m instanceof THREE.MeshPhysicalMaterial) {
        m.roughness = Math.max(m.roughness ?? 0.9, 0.88);
        m.metalness = m.metalness ?? 0;
      }
    }
  });
}

/** Dense fused surface from ``track_mesh.glb`` (Poisson + per-vertex RGB). */
export default function PhotogrammetryMesh({ track, url = '/data/track_mesh.glb' }: Props) {
  const [root, setRoot] = useState<THREE.Group | null>(null);

  const offset = useMemo(
    () => sourceToThree({ x: 0, y: 0, z: 0 }, track.origin),
    [track.origin.x, track.origin.y, track.origin.z],
  );

  useEffect(() => {
    let cancelled = false;
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        if (cancelled) return;
        const group = gltf.scene.clone(true);
        group.position.set(offset[0], offset[1], offset[2]);
        group.scale.z = -1;
        applyShadowAndSide(group);
        setRoot(group);
      },
      undefined,
      () => {
        if (!cancelled) setRoot(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [url, offset]);

  if (!root) return null;
  return <primitive object={root} />;
}
