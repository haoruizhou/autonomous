import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { type LoadedTrack, type TrackCone, isDirectionCone } from '../types/track';
import type { EditorTool } from '../types/editor';
import { coneLyingAlongYaw } from '../lib/flatConeOrientation';
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

/** Same orange as boundary cones (`MeshStandardMaterial.color`). */
const CONE_ORANGE = '#ff8c00';
/** Same emissive tint as boundary cones. */
const CONE_EMIT = '#331400';

/** One cycle along cone V: orange — white — orange — white (equal widths). */
function makeStripeTexture(): THREE.CanvasTexture {
  const H = 256;
  const regions: Array<{ t0: number; t1: number; color: string }> = [
    { t0: 0.0, t1: 0.25, color: CONE_ORANGE },
    { t0: 0.25, t1: 0.5, color: '#ffffff' },
    { t0: 0.5, t1: 0.75, color: CONE_ORANGE },
    { t0: 0.75, t1: 1.0, color: '#ffffff' },
  ];

  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = H;
  const ctx = canvas.getContext('2d')!;
  for (const r of regions) {
    ctx.fillStyle = r.color;
    const y0 = Math.floor(r.t0 * H);
    const y1 = Math.ceil(r.t1 * H);
    ctx.fillRect(0, y0, 2, y1 - y0 + 1);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1, 1);
  tex.center.set(0, 0);
  return tex;
}

export default function Cones({ track, cones, editMode = false, activeTool, onRemove, onToggleDown }: Props) {
  const coneGeo = useMemo(() => new THREE.ConeGeometry(0.22, 0.88, 16), []);

  const boundaryMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: CONE_ORANGE,
        emissive: CONE_EMIT,
        emissiveIntensity: 1,
        roughness: 0.55,
        metalness: 0,
      }),
    [],
  );

  const dirMat = useMemo(() => {
    const map = makeStripeTexture();
    // No emissiveMap: same per-pixel mix as boundary (albedo from map + uniform emissive tint).
    return new THREE.MeshStandardMaterial({
      map,
      color: new THREE.Color(0xffffff),
      emissive: new THREE.Color(CONE_EMIT),
      emissiveIntensity: 1,
      roughness: 0.55,
      metalness: 0,
    });
  }, []);

  const stripeScroll = useRef(0);
  useFrame((_, dt) => {
    const map = dirMat.map;
    if (!map) return;
    stripeScroll.current -= dt * 0.52;
    map.offset.y = stripeScroll.current;
  });

  return (
    <>
      {cones.map((cone) => {
        const isDirection = isDirectionCone(cone);
        const isBoundaryKnocked = !isDirection && (cone.knocked_over ?? false);
        const isLying = isDirection || isBoundaryKnocked;

        const zSource = groundHeight({ x: cone.xyz[0], y: cone.xyz[1], z: cone.xyz[2] }, track);
        const yOffset = isLying ? 0.22 : 0.44;
        const [x, y, z] = sourceToThree(
          { x: cone.xyz[0], y: cone.xyz[1], z: zSource + yOffset },
          track.origin,
        );
        const scale = 1.6 + Math.min(1.0, cone.n_observations / 25);

        let yaw = 0;
        if (isDirection) {
          const y = cone.yaw;
          yaw = typeof y === 'number' && Number.isFinite(y) ? y : 0;
        } else if (isBoundaryKnocked) {
          const y = cone.yaw;
          yaw =
            typeof y === 'number' && Number.isFinite(y)
              ? y
              : (((cone.id * 2654435761) >>> 0) / 2 ** 32) * Math.PI * 2;
        }

        const handleClick = editMode
          ? (e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation();
              if (activeTool === 'remove') onRemove?.(cone.id);
              else if (activeTool === 'toggle-down' && !isDirection) onToggleDown?.(cone.id);
            }
          : undefined;

        if (isLying) {
          return (
            <mesh
              key={cone.id}
              position={[x, y, z]}
              quaternion={coneLyingAlongYaw(yaw)}
              scale={scale}
              geometry={coneGeo}
              material={isDirection ? dirMat : boundaryMat}
              castShadow
              receiveShadow
              onClick={handleClick}
            />
          );
        }

        return (
          <mesh
            key={cone.id}
            position={[x, y, z]}
            scale={scale}
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
