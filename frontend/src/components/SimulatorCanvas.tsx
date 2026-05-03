import { Canvas, useFrame, useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { LoadedTrack, TrackCone } from '../types/track';
import type { EditorTool } from '../types/editor';
import { buildCellSurfaceGeometry, buildCenterlineGeometry } from '../lib/trackMesh';
import { sourceToThree, threeToSource } from '../lib/coordinates';
import {
  type CameraMode,
  type InputState,
  type SurfaceType,
  type VehicleState,
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
  // Editor
  editMode: boolean;
  editorTool: EditorTool;
  editedCones: TrackCone[];
  onAddCone: (xyz: [number, number, number], type: 'boundary' | 'direction') => void;
  onRemoveCone: (id: number) => void;
  onToggleDown: (id: number) => void;
};

function useKeyboard(onReset: () => void, onCycleCamera: () => void) {
  const input = useRef<InputState>({ throttle: false, brake: false, left: false, right: false, hardBrake: false });
  useEffect(() => {
    const set = (event: KeyboardEvent, value: boolean) => {
      switch (event.code) {
        case 'KeyW': case 'ArrowUp':    input.current.throttle  = value; break;
        case 'KeyS': case 'ArrowDown':  input.current.brake     = value; break;
        case 'KeyA': case 'ArrowLeft':  input.current.left      = value; break;
        case 'KeyD': case 'ArrowRight': input.current.right     = value; break;
        case 'Space': input.current.hardBrake = value; break;
        case 'KeyR': if (value) onReset(); break;
        case 'KeyC': if (value) onCycleCamera(); break;
      }
    };
    const down = (e: KeyboardEvent) => set(e, true);
    const up   = (e: KeyboardEvent) => set(e, false);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [onCycleCamera, onReset]);
  return input;
}

type VehicleProps = {
  track: LoadedTrack;
  enabled: boolean;
  vehiclePosRef: React.MutableRefObject<THREE.Vector3>;
  vehicleHeadingRef: React.MutableRefObject<number>;
  onTelemetry: Props['onTelemetry'];
};

function Vehicle({ track, enabled, vehiclePosRef, vehicleHeadingRef, onTelemetry }: VehicleProps) {
  const mesh = useRef<THREE.Mesh>(null);
  const hitIds = useRef(new Set<number>());
  const vehicleRef = useRef<VehicleState>(createInitialVehicle(track));
  const telemetryTimer = useRef(0);

  const handleReset = useCallback(() => {
    hitIds.current.clear();
    vehicleRef.current = createInitialVehicle(track);
  }, [track]);

  const handleCycleCamera = useCallback(
    () => window.dispatchEvent(new CustomEvent('cycle-camera')),
    [],
  );

  const input = useKeyboard(handleReset, handleCycleCamera);

  useFrame((_, dt) => {
    if (!enabled) return;

    const result = updateVehicle(vehicleRef.current, input.current, track, Math.min(dt, 0.05));
    vehicleRef.current = result.state;

    vehiclePosRef.current.copy(result.state.position);
    vehicleHeadingRef.current = result.state.heading;

    const source = threeToSource(result.state.position, track.origin);
    const addedHits = countConeHits(track, source, hitIds.current);

    telemetryTimer.current += dt;
    if (addedHits > 0 || telemetryTimer.current >= 0.083) {
      telemetryTimer.current = 0;
      onTelemetry({ speed: result.state.speed, surface: result.surface, coneHits: hitIds.current.size });
    }

    if (mesh.current) {
      mesh.current.position.copy(result.state.position);
      mesh.current.rotation.y = result.state.heading;
    }
  });

  return (
    <mesh ref={mesh} castShadow>
      <boxGeometry args={[1.2, 0.45, 2.2]} />
      <meshStandardMaterial color="#4da3ff" roughness={0.4} />
    </mesh>
  );
}

type CameraControllerProps = {
  vehiclePosRef: React.MutableRefObject<THREE.Vector3>;
  vehicleHeadingRef: React.MutableRefObject<number>;
  cameraMode: CameraMode;
};

function CameraController({ vehiclePosRef, vehicleHeadingRef, cameraMode }: CameraControllerProps) {
  const { camera, controls } = useThree();
  const prevMode = useRef<CameraMode>(cameraMode);
  const behindVec = useRef(new THREE.Vector3());

  useFrame(() => {
    const pos = vehiclePosRef.current;

    if (cameraMode === 'follow') {
      const heading = vehicleHeadingRef.current;
      behindVec.current.set(
        pos.x - Math.sin(heading) * 8,
        pos.y + 5,
        pos.z - Math.cos(heading) * 8,
      );
      camera.position.lerp(behindVec.current, 0.12);
      camera.lookAt(pos.x, pos.y + 0.5, pos.z);
    } else if (cameraMode === 'orbit' && prevMode.current === 'follow' && controls) {
      const oc = controls as unknown as { target: THREE.Vector3; update: () => void };
      oc.target.copy(pos);
      oc.update();
    }

    prevMode.current = cameraMode;
  });

  return null;
}

function TrackScene({
  track,
  showGrass,
  showCenterline,
  cameraMode,
  onTelemetry,
  editMode,
  editorTool,
  editedCones,
  onAddCone,
  onRemoveCone,
  onToggleDown,
}: Props) {
  const vehiclePosRef    = useRef<THREE.Vector3>(new THREE.Vector3());
  const vehicleHeadingRef = useRef<number>(0);

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
  const centerlineLine = useMemo(
    () => new THREE.Line(centerlineGeometry, new THREE.LineBasicMaterial({ color: '#32c7ff' })),
    [centerlineGeometry],
  );

  const handleGroundClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      if (!editMode) return;
      if (editorTool !== 'add-boundary' && editorTool !== 'add-direction') return;
      e.stopPropagation();
      const src = threeToSource(e.point, track.origin);
      onAddCone([src.x, src.y, src.z], editorTool === 'add-direction' ? 'direction' : 'boundary');
    },
    [editMode, editorTool, onAddCone, track],
  );

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

      {/* Invisible ground plane for cone placement — clicks fall through from cones via stopPropagation */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} visible={false} onClick={handleGroundClick}>
        <planeGeometry args={[600, 600]} />
        <meshBasicMaterial />
      </mesh>

      <Cones
        track={track}
        cones={editedCones}
        editMode={editMode}
        activeTool={editorTool}
        onRemove={onRemoveCone}
        onToggleDown={onToggleDown}
      />
      {showCenterline && <primitive object={centerlineLine} />}
      <Vehicle
        track={track}
        enabled={!editMode}
        vehiclePosRef={vehiclePosRef}
        vehicleHeadingRef={vehicleHeadingRef}
        onTelemetry={onTelemetry}
      />
      <CameraController
        vehiclePosRef={vehiclePosRef}
        vehicleHeadingRef={vehicleHeadingRef}
        cameraMode={cameraMode}
      />
      {cameraMode === 'orbit' && <OrbitControls makeDefault />}
    </>
  );
}

export default function SimulatorCanvas(props: Props) {
  return (
    <Canvas shadows camera={{ position: [0, 55, 85], fov: 55, near: 0.1, far: 1200 }}>
      <color attach="background" args={['#171a1f']} />
      <fog attach="fog" args={['#171a1f', 220, 650]} />
      <TrackScene {...props} />
    </Canvas>
  );
}
