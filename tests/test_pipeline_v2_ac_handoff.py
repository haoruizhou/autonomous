import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_v2.ac_handoff import build_flat_asphalt_mask, cell_vertices_flat, export_ac_handoff


def test_build_flat_asphalt_mask_fills_small_pothole_and_adds_apron():
    road_cells = []
    for j in range(5):
        for i in range(5):
            if (i, j) != (2, 2):
                road_cells.append([i, j, 123.0])
    grid = {"width": 7, "height": 7, "cell_m": 1.0, "x0": 0.0, "y0": 0.0}

    mask = build_flat_asphalt_mask(road_cells, grid, close_cells=1, apron_cells=1)

    assert mask[2, 2]
    assert mask[0, 2]
    assert mask[5, 2]
    assert mask[2, 0]
    assert mask[2, 5]


def test_cell_vertices_flat_uses_xy_from_grid_and_zero_z():
    grid = {"cell_m": 2.0, "x0": 10.0, "y0": -4.0}

    verts = cell_vertices_flat(3, 2, grid, z=0.0)

    assert verts == [
        (16.0, 0.0, 0.0),
        (18.0, 0.0, 0.0),
        (18.0, 0.0, 2.0),
        (16.0, 0.0, 2.0),
    ]


def test_export_ac_handoff_writes_flat_obj_mtl_and_cone_csv(tmp_path):
    track = {
        "grid": {"width": 4, "height": 4, "cell_m": 1.0, "x0": 0.0, "y0": 0.0},
        "road_cells": [[1, 1, 10.0], [2, 1, 10.0], [1, 2, 10.0], [2, 2, 10.0]],
        "cones": [{"id": 7, "xyz": [1.5, 2.5, 10.0], "n_observations": 3, "mean_conf": 0.8}],
        "centerline_hint": [[0.0, 0.0, 10.0], [3.0, 3.0, 10.0]],
    }
    project = tmp_path / "project"
    project.mkdir()
    (project / "track.json").write_text(json.dumps(track))

    out = export_ac_handoff(project, apron_cells=0, close_cells=0)

    assert (out / "track_asphalt.obj").exists()
    assert (out / "track_asphalt.mtl").exists()
    assert (out / "cones.csv").read_text().splitlines()[1] == "7,1.500,2.500,0.000,3,0.800"
    assert "v 1.0000 0.0000 1.0000" in (out / "track_asphalt.obj").read_text()
    assert (out / "guide_path.csv").exists()
    assert (out / "README.md").exists()
