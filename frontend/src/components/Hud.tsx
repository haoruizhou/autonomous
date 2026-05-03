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
  const mph = Math.abs(speed * 2.237);

  return (
    <div className="hud">
      <div className="hud-label">Autocross Simulator</div>

      <div className="hud-speedometer">
        <span className="hud-speed-value">{mph.toFixed(0)}</span>
        <span className="hud-speed-unit">mph</span>
      </div>

      <div className="hud-badges">
        <span className={`hud-badge hud-surface hud-surface-${surface}`}>{surface}</span>
        <span className={`hud-badge hud-cones ${coneHits > 0 ? 'hud-cones-hit' : ''}`}>
          {coneHits} {coneHits === 1 ? 'hit' : 'hits'}
        </span>
      </div>

      <div className="hud-divider" />

      <dl className="hud-stats">
        <dt>Road cells</dt>
        <dd>{track.raw.counts.road_cells.toLocaleString()}</dd>
        <dt>Cones</dt>
        <dd>{track.raw.counts.cones.toLocaleString()}</dd>
        <dt>Camera</dt>
        <dd>{cameraMode}</dd>
      </dl>

      <div className="hud-divider" />

      <div className="hud-toggles">
        <button className={showGrass ? 'active' : ''} onClick={onToggleGrass}>Grass</button>
        <button className={showCenterline ? 'active' : ''} onClick={onToggleCenterline}>Overlay</button>
        <button onClick={onCycleCamera}>
          {cameraMode === 'follow' ? '⤢ Orbit' : '⤡ Follow'}
        </button>
      </div>

      <div className="hud-controls">
        <div><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> · <kbd>↑</kbd><kbd>←</kbd><kbd>↓</kbd><kbd>→</kbd> drive</div>
        <div><kbd>Space</kbd> brake · <kbd>R</kbd> reset · <kbd>C</kbd> camera</div>
      </div>
    </div>
  );
}
