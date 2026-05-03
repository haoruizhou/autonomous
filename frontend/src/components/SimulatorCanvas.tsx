import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { LoadedTrack } from '../types/track';
import { buildCellSurfaceGeometry, buildCenterlineGeometry } from '../lib/trackMesh';
import { sourceToThree, threeToSource } from '../lib/coordinates';
import {
  type CameraMode,
  type InputState,
  type SurfaceType,
  countConeHits,
  createInitialVehicle,
  updateVehicle,
} from '../lib/vehicle';
import Cones from './Cones';
import PhotogrammetryMesh from './PhotogrammetryMesh';

type Props = {
  track: LoadedTrack;
  showGrass: boolean;
  showCenterline: boolean;
  cameraMode: CameraMode;
  onTelemetry: (telemetry: { speed: number; surface: SurfaceType; coneHits: number }) => void;
};

function useKeyboard(onReset: () => void, onCycleCamera: () => void) {
  const input = useRef<InputState>({ throttle: false, brake: false, left: false, right: false, hardBrake: false });
  useEffect(() => {
    const set = (event: KeyboardEvent, value: boolean) => {
      switch (event.code) {
        case 'KeyW':
        case 'ArrowUp':
          input.current.throttle = value;
          break;
        case 'KeyS':
        case 'ArrowDown':
          input.current.brake = value;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          input.current.left = value;
          break;
        case 'KeyD':
        case 'ArrowRight':
          input.current.right = value;
          break;
        case 'Space':
          input.current.hardBrake = value;
          break;
        case 'KeyR':
          if (value) onReset();
          break;
        case 'KeyC':
          if (value) onCycleCamera();
          break;
      }
    };
    const down = (event: KeyboardEvent) => set(event, true);
    const up = (event: KeyboardEvent) => set(event, false);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [onCycleCamera, onReset]);
  return input;
}

function Vehicle({ track, cameraMode, onTelemetry }: Pick<Props, 'track' | 'cameraMode' | 'onTelemetry'>) {
  const mesh = useRef<THREE.Mesh>(null);
  const hitIds = useRef(new Set<number>());
  const [vehicle, setVehicle] = useState(() => createInitialVehicle(track));
  const { camera } = useThree();
  const input = useKeyboard(() => {
    hitIds.current.clear();
    setVehicle(createInitialVehicle(track));
  }, () => window.dispatchEvent(new CustomEvent('cycle-camera')));

  useFrame((_, dt) => {
    const result = updateVehicle(vehicle, input.current, track, Math.min(dt, 0.05));
    setVehicle(result.state);
    const source = threeToSource(result.state.position, track.origin);
    const addedHits = countConeHits(track, source, hitIds.current);
    if (addedHits > 0 || Math.random() < 0.2) {
      onTelemetry({ speed: result.state.speed, surface: result.surface, coneHits: hitIds.current.size });
    }
    if (mesh.current) {
      mesh.current.position.copy(result.state.position);
      mesh.current.rotation.y = result.state.heading;
    }
    if (cameraMode === 'follow') {
      const behind = new THREE.Vector3(
        result.state.position.x - Math.sin(result.state.heading) * 8,
        result.state.position.y + 5,
        result.state.position.z - Math.cos(result.state.heading) * 8,
      );
      camera.position.lerp(behind, 0.12);
      camera.lookAt(result.state.position.x, result.state.position.y + 0.5, result.state.position.z);
    }
  });

  return (
    <mesh ref={mesh} castShadow>
      <boxGeometry args={[1.2, 0.45, 2.2]} />
      <meshStandardMaterial color="#4da3ff" roughness={0.4} />
    </mesh>
  );
}

function TrackScene({ track, showGrass, showCenterline, cameraMode, onTelemetry }: Props) {
  const roadGeometry = useMemo(
    () => buildCellSurfaceGeometry(track.renderRoadCells, track.raw.grid, track.origin, 0.02),
    [track],
  );
  const grassGeometry = useMemo(
    () => buildCellSurfaceGeometry(track.renderGrassCells, track.raw.grid, track.origin, -0.02),
    [track],
  );
  const centerlineGeometry = useMemo(
    () => buildCenterlineGeometry(track.raw.centerline_hint, track.origin),
    [track],
  );

  const start = track.raw.centerline_hint[0];
  const startPosition = start ? sourceToThree({ x: start[0], y: start[1], z: start[2] + 28 }, track.origin) : [0, 40, 40];

  return (
    <>
      <ambientLight intensity={0.65} />
      <directionalLight position={[40, 80, 25]} intensity={1.4} castShadow />
      <PhotogrammetryMesh track={track} />
      <mesh geometry={roadGeometry} receiveShadow visible={showCenterline}>
        <meshStandardMaterial color="#5ba9ff" transparent opacity={0.22} roughness={0.92} side={THREE.DoubleSide} />
      </mesh>
      {showGrass && (
        <mesh geometry={grassGeometry} receiveShadow visible={showCenterline}>
          <meshStandardMaterial color="#4c9f55" transparent opacity={0.18} roughness={0.95} side={THREE.DoubleSide} />
        </mesh>
      )}
      <Cones track={track} />
      {showCenterline && <primitive object={new THREE.Line(centerlineGeometry, new THREE.LineBasicMaterial({ color: '#32c7ff' }))} />}
      <Vehicle track={track} cameraMode={cameraMode} onTelemetry={onTelemetry} />
      {cameraMode === 'orbit' && <OrbitControls target={[0, 0, 0]} />}
      <perspectiveCamera position={startPosition as [number, number, number]} />
    </>
  );
}

export default function SimulatorCanvas(props: Props) {
  return (
    <Canvas shadows camera={{ position: [0, 55, 85], fov: 55, near: 0.1, far: 1200 }}>
      <color attach="background" args={["#171a1f"]} />
      <fog attach="fog" args={["#171a1f", 220, 650]} />
      <TrackScene {...props} />
    </Canvas>
  );
}
