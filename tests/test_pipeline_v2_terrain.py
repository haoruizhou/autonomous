import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_v2.terrain import clean_gps_path, correct_cloud_z, limit_local_z_jumps, terrain_z_at_xy


def test_clean_gps_path_removes_stationary_jitter_and_smooths_altitude():
    pts = np.array([
        [0.0, 0.0, 100.0],
        [0.05, 0.02, 130.0],
        [1.0, 0.0, 101.0],
        [2.0, 0.0, 102.0],
        [3.0, 0.0, 103.0],
        [4.0, 0.0, 104.0],
    ])

    cleaned = clean_gps_path(pts, min_step_m=0.5, z_window=5)

    assert cleaned.shape[1] == 3
    assert len(cleaned) == 5
    assert cleaned[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert cleaned[:, 2].max() - cleaned[:, 2].min() < 5.0


def test_terrain_z_at_xy_interpolates_nearby_cleaned_gps_samples():
    gps = np.array([
        [0.0, 0.0, 100.0],
        [10.0, 0.0, 110.0],
        [20.0, 0.0, 120.0],
    ])
    xy = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [5.0, 1.0],
        [15.0, -1.0],
    ])

    z = terrain_z_at_xy(xy, gps, k=2)

    np.testing.assert_allclose(z[:2], [100.0, 110.0], atol=1e-6)
    assert 102.0 < z[2] < 108.0
    assert 112.0 < z[3] < 118.0


def test_correct_cloud_z_clamps_depthmap_residual_to_gps_terrain():
    xyz = np.array([
        [0.0, 0.0, 80.0],
        [10.0, 0.0, 130.0],
        [20.0, 0.0, 119.8],
        [10.0, 10.0, 200.0],
    ])
    cls = np.array([1, 2, 1, 3], dtype=np.uint8)
    gps = np.array([
        [0.0, 0.0, 100.0],
        [10.0, 0.0, 110.0],
        [20.0, 0.0, 120.0],
    ])

    corrected = correct_cloud_z(xyz, cls, gps, residual_m=0.5)

    np.testing.assert_allclose(corrected[:3, 2], [99.5, 110.5, 119.8], atol=1e-6)
    assert corrected[3, 2] == 200.0


def test_limit_local_z_jumps_reduces_neighboring_ground_steps():
    xyz = np.array([
        [0.0, 0.0, 100.0],
        [1.0, 0.0, 104.0],
        [2.0, 0.0, 100.2],
        [10.0, 0.0, 130.0],
    ])
    cls = np.array([1, 1, 2, 3], dtype=np.uint8)

    limited = limit_local_z_jumps(xyz, cls, cell_m=1.1, max_jump_m=0.25, iterations=6)

    assert abs(limited[1, 2] - limited[0, 2]) <= 0.25 + 1e-6
    assert abs(limited[1, 2] - limited[2, 2]) <= 0.25 + 1e-6
    assert limited[3, 2] == 130.0
