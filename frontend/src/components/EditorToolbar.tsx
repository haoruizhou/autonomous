import type { EditorTool } from '../types/editor';

type Props = {
  activeTool: EditorTool;
  onToolChange: (tool: EditorTool) => void;
  addedCount: number;
  removedCount: number;
  knockedCount: number;
  onExport: () => void;
};

const TOOLS: { id: EditorTool; label: string; hint: string }[] = [
  { id: 'add-boundary', label: '+ Boundary', hint: 'Click track surface to place a boundary cone' },
  { id: 'add-direction', label: '+ Direction', hint: '1st click anchor, 2nd click direction; Esc cancels' },
  { id: 'remove',        label: '✕ Remove',   hint: 'Click a cone to delete it' },
  { id: 'toggle-down',   label: '⤵ Knock',    hint: 'Boundary cones only: knock over or stand up (direction cones are not knockable)' },
];

export default function EditorToolbar({
  activeTool,
  onToolChange,
  addedCount,
  removedCount,
  knockedCount,
  onExport,
}: Props) {
  const hasChanges = addedCount > 0 || removedCount > 0 || knockedCount > 0;

  return (
    <div className="editor-toolbar">
      <div className="editor-header">
        <span className="editor-title">Cone Editor</span>
        <div className="editor-stats">
          {addedCount  > 0 && <span className="estat-added">+{addedCount}</span>}
          {removedCount > 0 && <span className="estat-removed">−{removedCount}</span>}
          {knockedCount > 0 && <span className="estat-knocked">{knockedCount} down</span>}
          {!hasChanges   && <span className="estat-none">no changes</span>}
        </div>
      </div>

      <div className="editor-tools">
        {TOOLS.map((tool) => (
          <button
            key={tool.id}
            className={activeTool === tool.id ? 'active' : ''}
            title={tool.hint}
            onClick={() => onToolChange(tool.id)}
          >
            {tool.label}
          </button>
        ))}
      </div>

      <div className="editor-legend">
        <span className="legend-boundary">■ Boundary</span>
        <span className="legend-direction">■ Direction</span>
        <span className="legend-down">■ Knocked over</span>
      </div>

      <button className="export-btn" onClick={onExport}>
        ↓ Export track.json
      </button>
    </div>
  );
}
