import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_v2.geo import apply_similarity, fit_similarity


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
