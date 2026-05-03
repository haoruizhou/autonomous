import { useCallback, useEffect, useState } from 'react';
import type { LoadedTrack } from './types/track';
import { loadTrack } from './lib/trackLoader';
import type { CameraMode, SurfaceType } from './lib/vehicle';
import SimulatorCanvas from './components/SimulatorCanvas';
import Hud from './components/Hud';

export default function App() {
  const [track, setTrack] = useState<LoadedTrack | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGrass, setShowGrass] = useState(true);
  const [showCenterline, setShowCenterline] = useState(true);
  const [cameraMode, setCameraMode] = useState<CameraMode>('follow');
  const [telemetry, setTelemetry] = useState<{ speed: number; surface: SurfaceType; coneHits: number }>({
    speed: 0,
    surface: 'outside',
    coneHits: 0,
  });

  const cycleCamera = useCallback(() => {
    setCameraMode((mode) => (mode === 'follow' ? 'orbit' : 'follow'));
  }, []);

  useEffect(() => {
    loadTrack().then(setTrack).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err));
    });
  }, []);

  useEffect(() => {
    const handler = () => cycleCamera();
    window.addEventListener('cycle-camera', handler);
    return () => window.removeEventListener('cycle-camera', handler);
  }, [cycleCamera]);

  if (error) return <div className="error">{error}</div>;
  if (!track) return <div className="loading">Loading measured track…</div>;

  return (
    <>
      <Hud
        track={track}
        speed={telemetry.speed}
        surface={telemetry.surface}
        coneHits={telemetry.coneHits}
        showGrass={showGrass}
        showCenterline={showCenterline}
        cameraMode={cameraMode}
        onToggleGrass={() => setShowGrass((value) => !value)}
        onToggleCenterline={() => setShowCenterline((value) => !value)}
        onCycleCamera={cycleCamera}
      />
      <SimulatorCanvas
        track={track}
        showGrass={showGrass}
        showCenterline={showCenterline}
        cameraMode={cameraMode}
        onTelemetry={setTelemetry}
      />
    </>
  );
}
