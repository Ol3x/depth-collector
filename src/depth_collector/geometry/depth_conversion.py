from __future__ import annotations

import numpy as np


def z_depth_to_distance(z_depth: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
    forward = ray_dir[..., 2:3]
    return z_depth / forward


def clip_distance_to_max_dist(distance: np.ndarray, max_dist: float) -> np.ndarray:
    return np.clip(distance, 0.0, max_dist)


def distance_to_points(distance: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
    return distance * ray_dir
