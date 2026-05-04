import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { LoadedTrack, TrackCone } from '../types/track';
import type { EditorTool } from '../types/editor';
import { buildCellSurfaceGeometry, buildCenterlineGeometry } from '../lib/trackMesh';
import { threeToSource } from '../lib/coordinates';
import { coneLyingAlongYaw } from '../lib/flatConeOrientation';
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
import GridMesh from './GridMesh';
import PhotogrammetryMesh from './PhotogrammetryMesh';

type Props = {
  track: LoadedTrack;
  showGrass: boolean;
  showCenterline: boolean;
  showWorldMesh: boolean;
  cameraMode: CameraMode;
  onTelemetry: (telemetry: { speed: number; surface: SurfaceType; coneHits: number }) => void;
  editMode: boolean;
  editorTool: EditorTool;
  editedCones: TrackCone[];
  onAddCone: (xyz: [number, number, number], type: 'boundary' | 'direction', yaw?: number) => void;
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
  cones: TrackCone[];
  enabled: boolean;
  vehiclePosRef: React.MutableRefObject<THREE.Vector3>;
  vehicleHeadingRef: React.MutableRefObject<number>;
  onTelemetry: Props['onTelemetry'];
};

function Vehicle({ track, cones, enabled, vehiclePosRef, vehicleHeadingRef, onTelemetry }: VehicleProps) {
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
    const addedHits = countConeHits(cones, source, hitIds.current);

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

// Two-click direction placement: first pointerdown = anchor, second = tip direction (XZ).
type PendingDirection = { pos: THREE.Vector3; angle: number };

function TrackScene({
  track,
  showGrass,
  showCenterline,
  showWorldMesh,
  cameraMode,
  onTelemetry,
  editMode,
  editorTool,
  editedCones,
  onAddCone,
  onRemoveCone,
  onToggleDown,
}: Props) {
  const vehiclePosRef = useRef<THREE.Vector3>(new THREE.Vector3());
  const vehicleHeadingRef = useRef<number>(0);

  const pendingDirRef = useRef<PendingDirection | null>(null);
  const [pendingDir, setPendingDir] = useState<PendingDirection | null>(null);

  const { gl, camera } = useThree();
  const cameraRef = useRef(camera);
  cameraRef.current = camera;

  const editorRef = useRef({ editMode, editorTool });
  const onAddConeRef = useRef(onAddCone);
  const trackRef = useRef(track);
  useEffect(() => {
    editorRef.current = { editMode, editorTool };
  }, [editMode, editorTool]);
  useEffect(() => {
    onAddConeRef.current = onAddCone;
  }, [onAddCone]);
  useEffect(() => {
    trackRef.current = track;
  }, [track]);

  useEffect(() => {
    const isAdd = editMode && (editorTool === 'add-boundary' || editorTool === 'add-direction');
    if (!isAdd && pendingDirRef.current) {
      pendingDirRef.current = null;
      setPendingDir(null);
    }
  }, [editMode, editorTool]);

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

  const previewGeo = useMemo(() => new THREE.ConeGeometry(0.22, 0.88, 16), []);
  const previewMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#ff8c00',
        emissive: '#ff8c00',
        emissiveIntensity: 0.45,
        transparent: true,
        opacity: 0.5,
        roughness: 0.45,
        depthWrite: false,
      }),
    [],
  );

  useEffect(() => {
    const canvas = gl.domElement;
    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();

    const groundPoint = (clientX: number, clientY: number): THREE.Vector3 | null => {
      const rect = canvas.getBoundingClientRect();
      ndc.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(ndc, cameraRef.current);
      const out = new THREE.Vector3();
      return raycaster.ray.intersectPlane(groundPlane, out) ? out : null;
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const { editMode: em, editorTool: et } = editorRef.current;
      if (!em) return;
      if (et !== 'add-boundary' && et !== 'add-direction') return;
      const point = groundPoint(e.clientX, e.clientY);
      if (!point) return;
      e.preventDefault();
      e.stopPropagation();

      if (et === 'add-boundary') {
        const src = threeToSource(point, trackRef.current.origin);
        onAddConeRef.current([src.x, src.y, src.z], 'boundary');
        return;
      }

      if (pendingDirRef.current === null) {
        const next: PendingDirection = { pos: point.clone(), angle: 0 };
        pendingDirRef.current = next;
        setPendingDir({ pos: next.pos.clone(), angle: 0 });
      } else {
        const start = pendingDirRef.current.pos;
        const dx = point.x - start.x;
        const dz = point.z - start.z;
        const angle =
          Math.hypot(dx, dz) < 1e-4 ? pendingDirRef.current.angle : Math.atan2(dx, dz);
        const src = threeToSource(start, trackRef.current.origin);
        onAddConeRef.current([src.x, src.y, src.z], 'direction', angle);
        pendingDirRef.current = null;
        setPendingDir(null);
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!pendingDirRef.current) return;
      if (editorRef.current.editorTool !== 'add-direction') return;
      const p = groundPoint(e.clientX, e.clientY);
      if (!p) return;
      const start = pendingDirRef.current.pos;
      const dx = p.x - start.x;
      const dz = p.z - start.z;
      if (Math.hypot(dx, dz) < 1e-3) return;
      const angle = Math.atan2(dx, dz);
      const next: PendingDirection = { pos: start, angle };
      pendingDirRef.current = next;
      setPendingDir({ pos: start.clone(), angle });
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && pendingDirRef.current) {
        pendingDirRef.current = null;
        setPendingDir(null);
      }
    };

    canvas.addEventListener('pointerdown', onPointerDown, { capture: true });
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('keydown', onKey);
    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown, { capture: true });
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('keydown', onKey);
    };
  }, [gl]);

  const aimLine = useMemo(() => {
    if (!pendingDir) return null;
    const len = 4;
    const a = pendingDir.pos;
    const b = new THREE.Vector3(
      a.x + Math.sin(pendingDir.angle) * len,
      a.y + 0.05,
      a.z + Math.cos(pendingDir.angle) * len,
    );
    const geom = new THREE.BufferGeometry().setFromPoints([a, b]);
    const mat = new THREE.LineBasicMaterial({ color: '#ffd400', transparent: true, opacity: 0.92 });
    return new THREE.Line(geom, mat);
  }, [pendingDir]);

  return (
    <>
      <ambientLight intensity={0.65} />
      <directionalLight position={[40, 80, 25]} intensity={1.4} castShadow />
      {showWorldMesh && <PhotogrammetryMesh track={track} />}
      <GridMesh track={track} />
      <mesh geometry={roadGeometry} receiveShadow visible={showCenterline}>
        <meshStandardMaterial color="#5ba9ff" transparent opacity={0.22} roughness={0.92} side={THREE.DoubleSide} />
      </mesh>
      {showGrass && (
        <mesh geometry={grassGeometry} receiveShadow visible={showCenterline}>
          <meshStandardMaterial color="#4c9f55" transparent opacity={0.18} roughness={0.95} side={THREE.DoubleSide} />
        </mesh>
      )}

      {pendingDir && editorTool === 'add-direction' && (
        <>
          <mesh
            position={[pendingDir.pos.x, pendingDir.pos.y + 0.22 * 1.8, pendingDir.pos.z]}
            quaternion={coneLyingAlongYaw(pendingDir.angle)}
            scale={1.8}
            geometry={previewGeo}
            material={previewMat}
          />
          {aimLine && <primitive object={aimLine} />}
        </>
      )}

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
        cones={editedCones}
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
      {cameraMode === 'orbit' && (
        <OrbitControls
          makeDefault
          enabled={!(editMode && (editorTool === 'add-boundary' || editorTool === 'add-direction'))}
        />
      )}
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
