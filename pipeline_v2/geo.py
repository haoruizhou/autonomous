"""Geometry helpers for putting OpenSfM outputs into the GPS-local frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def angle_axis_to_R(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def camera_center(shot: dict) -> np.ndarray:
    """Return a shot center in the raw OpenSfM reconstruction frame."""
    R = angle_axis_to_R(np.array(shot["rotation"], dtype=np.float64))
    t = np.array(shot["translation"], dtype=np.float64)
    return -R.T @ t


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> dict:
    """Fit dst ~= scale * src @ R.T + t using Umeyama/Kabsch alignment."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError("Need at least 3 points to fit a similarity transform")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    var_src = float(np.mean(np.sum(X * X, axis=1)))
    if var_src <= 1e-12:
        raise ValueError("Cannot fit similarity transform from degenerate source points")

    cov = (X.T @ Y) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        S[-1] *= -1
        R = Vt.T @ U.T
    scale = float(np.sum(S) / var_src)
    t = dst_mean - scale * (src_mean @ R.T)
    fitted = apply_similarity(src, {"scale": scale, "R": R, "t": t})
    err = np.linalg.norm(fitted - dst, axis=1)
    return {
        "scale": scale,
        "R": R,
        "t": t,
        "n": int(len(src)),
        "rms_error_m": float(np.sqrt(np.mean(err * err))),
        "median_error_m": float(np.median(err)),
        "max_error_m": float(np.max(err)),
    }


def apply_similarity(xyz: np.ndarray, transform: dict) -> np.ndarray:
    R = np.asarray(transform["R"], dtype=np.float64)
    t = np.asarray(transform["t"], dtype=np.float64)
    scale = float(transform["scale"])
    return (scale * (np.asarray(xyz, dtype=np.float64) @ R.T) + t).astype(np.float32)


def component_similarity_to_gps(undist_dir: Path) -> dict | None:
    """Fit raw camera centers to OpenSfM's GPS-local positions for a component."""
    rec_path = Path(undist_dir) / "reconstruction.json"
    if not rec_path.exists():
        return None
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    for rec in json.loads(rec_path.read_text()):
        for shot in rec.get("shots", {}).values():
            gps = shot.get("gps_position")
            if gps is None:
                continue
            src.append(camera_center(shot))
            dst.append(np.array(gps, dtype=np.float64))
    if len(src) < 3:
        return None
    return fit_similarity(np.stack(src), np.stack(dst))


def transform_for_json(transform: dict) -> dict:
    return {
        "scale": float(transform["scale"]),
        "R": np.asarray(transform["R"], dtype=float).tolist(),
        "t": np.asarray(transform["t"], dtype=float).tolist(),
        "n": int(transform.get("n", 0)),
        "rms_error_m": float(transform.get("rms_error_m", 0.0)),
        "median_error_m": float(transform.get("median_error_m", 0.0)),
        "max_error_m": float(transform.get("max_error_m", 0.0)),
    }
