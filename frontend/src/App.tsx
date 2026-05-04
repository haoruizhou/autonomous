import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { type LoadedTrack, type TrackCone, isDirectionCone } from './types/track';
import type { EditorTool } from './types/editor';
import { loadTrack } from './lib/trackLoader';
import type { CameraMode, SurfaceType } from './lib/vehicle';
import SimulatorCanvas from './components/SimulatorCanvas';
import Hud from './components/Hud';
import EditorToolbar from './components/EditorToolbar';

export default function App() {
  const [track, setTrack] = useState<LoadedTrack | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Simulator state
  const [showGrass, setShowGrass] = useState(true);
  const [showCenterline, setShowCenterline] = useState(true);
  const [showWorldMesh, setShowWorldMesh] = useState(true);
  const [cameraMode, setCameraMode] = useState<CameraMode>('follow');
  const [telemetry, setTelemetry] = useState<{ speed: number; surface: SurfaceType; coneHits: number }>({
    speed: 0, surface: 'outside', coneHits: 0,
  });

  // Editor state
  const [editMode, setEditMode] = useState(false);
  const [editorTool, setEditorTool] = useState<EditorTool>('add-boundary');
  const [editedCones, setEditedCones] = useState<TrackCone[]>([]);
  const nextConeId = useRef(0);

  const cycleCamera = useCallback(() => {
    setCameraMode((mode) => (mode === 'follow' ? 'orbit' : 'follow'));
  }, []);

  useEffect(() => {
    loadTrack().then(setTrack).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err));
    });
  }, []);

  // Seed editor cone list and ID counter once the track is loaded.
  useEffect(() => {
    if (!track) return;
    // Direction cones never use knocked_over — strip legacy flag so knock is boundary-only.
    setEditedCones(
      track.raw.cones.map((c) => {
        if (!isDirectionCone(c)) return c;
        const { knocked_over: _k, ...rest } = c;
        return rest;
      }),
    );
    nextConeId.current = Math.max(0, ...track.raw.cones.map((c) => c.id)) + 1;
  }, [track]);

  useEffect(() => {
    const handler = () => cycleCamera();
    window.addEventListener('cycle-camera', handler);
    return () => window.removeEventListener('cycle-camera', handler);
  }, [cycleCamera]);

  const toggleEdit = useCallback(() => {
    setEditMode((prev) => {
      const next = !prev;
      // Auto-switch to orbit so the user can pan around while editing.
      setCameraMode(next ? 'orbit' : 'follow');
      return next;
    });
  }, []);

  // ── Cone manipulation handlers ─────────────────────────────────────────────

  const addCone = useCallback((xyz: [number, number, number], type: 'boundary' | 'direction', yaw?: number) => {
    const id = nextConeId.current++;
    setEditedCones((prev) => [
      ...prev,
      {
        id, xyz,
        n_observations: 1,
        mean_conf: 1.0,
        cone_type: type,
        // knocked_over is boundary-only; direction cones are always lying (see cone_type + yaw).
        ...(type === 'direction' ? { yaw: yaw ?? 0 } : {}),
      },
    ]);
  }, []);

  const removeCone = useCallback((id: number) => {
    setEditedCones((prev) => prev.filter((c) => c.id !== id));
  }, []);

  const toggleDown = useCallback((id: number) => {
    setEditedCones((prev) => {
      const target = prev.find((c) => c.id === id);
      if (!target || isDirectionCone(target)) return prev;

      return prev.map((c) => {
        if (c.id !== id) return c;

        const wasDown = c.knocked_over ?? false;
        const willBeDown = !wasDown;
        if (!willBeDown) return { ...c, knocked_over: false };

        const yaw = (((c.id * 2654435761) >>> 0) / 2 ** 32) * Math.PI * 2;
        return { ...c, knocked_over: true, yaw };
      });
    });
  }, []);

  // ── Export ─────────────────────────────────────────────────────────────────

  const exportTrack = useCallback(() => {
    if (!track) return;
    const cones = editedCones.map((c) => {
      if (!isDirectionCone(c)) return c;
      const { knocked_over: _k, ...rest } = c;
      return rest;
    });
    const exported = { ...track.raw, cones };
    const blob = new Blob([JSON.stringify(exported, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'track-edited.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [track, editedCones]);

  // ── Editor diff stats ──────────────────────────────────────────────────────

  const originalIds = useMemo(
    () => new Set(track?.raw.cones.map((c) => c.id) ?? []),
    [track],
  );
  const addedCount   = editedCones.filter((c) => !originalIds.has(c.id)).length;
  const removedCount = (track?.raw.cones.length ?? 0) - editedCones.filter((c) => originalIds.has(c.id)).length;
  const knockedCount = editedCones.filter(
    (c) => !isDirectionCone(c) && c.knocked_over,
  ).length;

  // ── Render ─────────────────────────────────────────────────────────────────

  if (error) return <div className="error">{error}</div>;
  if (!track) return <div className="loading">Loading measured track…</div>;

  return (
    <div className={`app${editMode ? ` edit-mode tool-${editorTool}` : ''}`}>
      <div className="hud-stack">
        <Hud
          track={track}
          speed={telemetry.speed}
          surface={telemetry.surface}
          coneHits={telemetry.coneHits}
          showGrass={showGrass}
          showCenterline={showCenterline}
          showWorldMesh={showWorldMesh}
          cameraMode={cameraMode}
          editMode={editMode}
          onToggleGrass={() => setShowGrass((v) => !v)}
          onToggleCenterline={() => setShowCenterline((v) => !v)}
          onToggleWorldMesh={() => setShowWorldMesh((v) => !v)}
          onCycleCamera={cycleCamera}
          onToggleEdit={toggleEdit}
        />
        {editMode && (
          <EditorToolbar
            activeTool={editorTool}
            onToolChange={setEditorTool}
            addedCount={addedCount}
            removedCount={removedCount}
            knockedCount={knockedCount}
            onExport={exportTrack}
          />
        )}
      </div>
      <SimulatorCanvas
        track={track}
        showGrass={showGrass}
        showCenterline={showCenterline}
        showWorldMesh={showWorldMesh}
        cameraMode={cameraMode}
        onTelemetry={setTelemetry}
        editMode={editMode}
        editorTool={editorTool}
        editedCones={editedCones}
        onAddCone={addCone}
        onRemoveCone={removeCone}
        onToggleDown={toggleDown}
      />
    </div>
  );
}
