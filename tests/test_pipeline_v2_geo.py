import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_v2.geo import apply_similarity, fit_similarity
from pipeline_v2.mesh import source_to_glb_vertices


def test_fit_similarity_recovers_scale_rotation_translation():
    src = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, 2.0, 0.5],
    ])
    theta = np.deg2rad(35.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    dst = 3.25 * (src @ R.T) + np.array([10.0, -4.0, 2.0])

    transform = fit_similarity(src, dst)
    fitted = apply_similarity(src, transform)

    np.testing.assert_allclose(fitted, dst, atol=1e-5)
    assert transform["rms_error_m"] < 1e-6


def test_source_to_glb_vertices_matches_frontend_grid_basis_before_z_flip():
    src = np.array([
        [10.0, 20.0, 2.0],
        [15.0, 35.0, 3.5],
        [-5.0, 12.0, 1.0],
    ])
    origin = np.array([7.0, 18.0, 1.5])

    vertices, flip_winding = source_to_glb_vertices(src)
    after_frontend_flip = vertices * np.array([1.0, 1.0, -1.0])
    after_offset = after_frontend_flip + np.array([-origin[0], -origin[2], origin[1]])
    direct_overlay = np.column_stack([
        src[:, 0] - origin[0],
        src[:, 2] - origin[2],
        -(src[:, 1] - origin[1]),
    ])

    np.testing.assert_allclose(vertices, np.column_stack([src[:, 0], src[:, 2], src[:, 1]]))
    np.testing.assert_allclose(after_offset, direct_overlay)
    assert flip_winding is False
