import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_v2.preview_ac_handoff import load_cones_csv, load_csv_points, load_obj_topdown, render_preview


def test_load_obj_topdown_reads_vertices_and_faces(tmp_path):
    obj = tmp_path / "track.obj"
    obj.write_text(
        "v 1.0 0.0 2.0\n"
        "v 3.0 0.0 2.0\n"
        "v 3.0 0.0 4.0\n"
        "f 1 2 3\n"
    )

    vertices, faces = load_obj_topdown(obj)

    np.testing.assert_allclose(vertices, [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(faces, [[0, 1, 2]])


def test_load_csv_points_reads_xy_columns(tmp_path):
    csv = tmp_path / "guide_path.csv"
    csv.write_text("x,y,z\n1.0,2.0,0.0\n3.0,4.0,0.0\n")

    points = load_csv_points(csv)

    np.testing.assert_allclose(points, [[1.0, 2.0], [3.0, 4.0]])


def test_load_cones_csv_returns_empty_when_only_header(tmp_path):
    csv = tmp_path / "cones.csv"
    csv.write_text("id,x,y,z,n_observations,mean_conf\n")

    cones = load_cones_csv(csv)

    assert cones.shape == (0, 2)


def test_render_preview_writes_png(tmp_path):
    handoff = tmp_path / "ac_handoff"
    handoff.mkdir()
    (handoff / "track_asphalt.obj").write_text(
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 0.0 1.0\n"
        "f 1 2 3\n"
    )
    (handoff / "guide_path.csv").write_text("x,y,z\n0.0,0.0,0.0\n1.0,1.0,0.0\n")
    (handoff / "cones.csv").write_text("id,x,y,z,n_observations,mean_conf\n1,0.5,0.5,0.0,2,0.9\n")
    out = handoff / "preview.png"

    render_preview(handoff, out)

    assert out.exists()
    assert out.stat().st_size > 0
