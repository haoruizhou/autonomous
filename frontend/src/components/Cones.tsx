import { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { LoadedTrack, TrackCone } from '../types/track';
import type { EditorTool } from '../types/editor';
import { sourceToThree } from '../lib/coordinates';
import { groundHeight } from '../lib/vehicle';

type Props = {
  track: LoadedTrack;
  cones: TrackCone[];
  editMode?: boolean;
  activeTool?: EditorTool;
  onRemove?: (id: number) => void;
  onToggleDown?: (id: number) => void;
};

export default function Cones({ track, cones, editMode = false, activeTool, onRemove, onToggleDown }: Props) {
  const coneGeo = useMemo(() => new THREE.ConeGeometry(0.22, 0.88, 16), []);

  const boundaryMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#ff8c00', emissive: '#331400', roughness: 0.55 }),
    [],
  );

  // Direction cones animate between orange and white in sync — like road-diverge >>> signs.
  const dirMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#ff8c00', emissive: '#ff4400', emissiveIntensity: 0.5, roughness: 0.45 }),
    [],
  );
  const _colorOrange = useMemo(() => new THREE.Color('#ff8c00'), []);
  const _colorWhite  = useMemo(() => new THREE.Color('#ffffff'), []);

  useFrame(({ clock }) => {
    // ~2.5 full cycles/sec gives a noticeable but not jarring flicker
    const t = (Math.sin(clock.elapsedTime * Math.PI * 5) + 1) / 2;
    dirMat.color.lerpColors(_colorOrange, _colorWhite, t * 0.72);
    dirMat.emissiveIntensity = THREE.MathUtils.lerp(0.1, 2.4, t);
  });

  return (
    <>
      {cones.map((cone) => {
        const isDirection  = (cone.cone_type ?? 'boundary') === 'direction';
        const isKnockedOver = cone.knocked_over ?? false;

        const zSource = groundHeight(track, { x: cone.xyz[0], y: cone.xyz[1], z: cone.xyz[2] });
        // Knocked-over: cone lies on its side, center sits at the base radius above the ground.
        const yOffset = isKnockedOver ? 0.22 : 0.44;
        const [x, y, z] = sourceToThree(
          { x: cone.xyz[0], y: cone.xyz[1], z: zSource + yOffset },
          track.origin,
        );
        const scale = 1.6 + Math.min(1.0, cone.n_observations / 25);
        // Give each knocked cone a different yaw so they look naturally scattered.
        const yaw = isKnockedOver ? (cone.id % 8) * (Math.PI / 4) : 0;

        const handleClick = editMode
          ? (e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation();
              if (activeTool === 'remove')      onRemove?.(cone.id);
              else if (activeTool === 'toggle-down') onToggleDown?.(cone.id);
            }
          : undefined;

        return (
          <mesh
            key={cone.id}
            position={[x, y, z]}
            rotation={[0, yaw, isKnockedOver ? Math.PI / 2 : 0]}
            scale={scale}
            // geometry + material set via Three.js property setters — R3F won't dispose
            // shared instances when individual cones are removed.
            geometry={coneGeo}
            material={isDirection ? dirMat : boundaryMat}
            castShadow
            receiveShadow
            onClick={handleClick}
          />
        );
      })}
    </>
  );
}
