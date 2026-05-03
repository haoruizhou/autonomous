export type TrackGrid = {
  x0: number;
  y0: number;
  width: number;
  height: number;
  cell_m: number;
};

export type TrackCell = [i: number, j: number, z: number];

export type TrackCone = {
  id: number;
  xyz: [number, number, number];
  n_observations: number;
  mean_conf: number;
};

export type TrackJson = {
  schema: 'autocross-track-v1';
  units: 'meters';
  coordinate_frame: {
    source: string;
    horizontal_axes: ['x', 'y'];
    vertical_axis: 'z';
  };
  grid: TrackGrid;
  road_cells: TrackCell[];
  grass_cells: TrackCell[];
  cones: TrackCone[];
  centerline_hint: [number, number, number][];
  counts: {
    road_cells: number;
    grass_cells: number;
    cones: number;
    road_points: number;
    grass_points: number;
    cone_points: number;
  };
};

export type SourcePoint = {
  x: number;
  y: number;
  z: number;
};

export type LoadedTrack = {
  raw: TrackJson;
  renderRoadCells: TrackCell[];
  visualRoadFillCells: TrackCell[];
  renderGrassCells: TrackCell[];
  origin: SourcePoint;
  roadSet: Set<string>;
  grassSet: Set<string>;
  roadHeight: Map<string, number>;
  grassHeight: Map<string, number>;
};
