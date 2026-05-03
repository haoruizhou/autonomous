import type { LoadedTrack } from '../types/track';
import type { CameraMode, SurfaceType } from '../lib/vehicle';

type Props = {
  track: LoadedTrack;
  speed: number;
  surface: SurfaceType;
  coneHits: number;
  showGrass: boolean;
  showCenterline: boolean;
  cameraMode: CameraMode;
  onToggleGrass: () => void;
  onToggleCenterline: () => void;
  onCycleCamera: () => void;
};

export default function Hud({
  track,
  speed,
  surface,
  coneHits,
  showGrass,
  showCenterline,
  cameraMode,
  onToggleGrass,
  onToggleCenterline,
  onCycleCamera,
}: Props) {
  return (
    <div className="hud">
      <h1>Autocross Simulator</h1>
      <dl>
        <dt>Speed</dt>
        <dd>{Math.abs(speed * 2.237).toFixed(1)} mph</dd>
        <dt>Surface</dt>
        <dd>{surface}</dd>
        <dt>Cone hits</dt>
        <dd>{coneHits}</dd>
        <dt>Road cells</dt>
        <dd>{track.raw.counts.road_cells.toLocaleString()}</dd>
        <dt>Grass cells</dt>
        <dd>{track.raw.counts.grass_cells.toLocaleString()}</dd>
        <dt>Cones</dt>
        <dd>{track.raw.counts.cones.toLocaleString()}</dd>
        <dt>Camera</dt>
        <dd>{cameraMode}</dd>
      </dl>
      <div className="toggles">
        <button className={showGrass ? 'active' : ''} onClick={onToggleGrass}>Grass</button>
        <button className={showCenterline ? 'active' : ''} onClick={onToggleCenterline}>Debug grid</button>
        <button onClick={onCycleCamera}>Camera</button>
      </div>
      <div className="controls">
        W/A/S/D or arrows drive · Space brake · R reset · C camera
      </div>
    </div>
  );
}
