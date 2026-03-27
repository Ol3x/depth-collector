from __future__ import annotations

import numpy as np

from .camera_models import EquirectangularCameraModel, PinholeCameraModel


def normalize_rays(rays: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / norms


def generate_pinhole_rays(camera: PinholeCameraModel) -> np.ndarray:
    xs = np.arange(camera.width, dtype=np.float32)
    ys = np.arange(camera.height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Canonical camera frame is left, down, forward.
    x_left = (camera.cx - grid_x) / camera.fx
    y_down = (grid_y - camera.cy) / camera.fy
    z_forward = np.ones_like(x_left)
    rays = np.stack((x_left, y_down, z_forward), axis=-1)
    return normalize_rays(rays)


def generate_equirectangular_rays(camera: EquirectangularCameraModel) -> np.ndarray:
    xs = np.arange(camera.width, dtype=np.float32)
    ys = np.arange(camera.height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    lon = ((grid_x + 0.5) / camera.width - 0.5) * (2.0 * np.pi)
    lat = (0.5 - (grid_y + 0.5) / camera.height) * np.pi

    forward = np.cos(lat) * np.cos(lon)
    left = -np.cos(lat) * np.sin(lon)
    down = -np.sin(lat)
    rays = np.stack((left, down, forward), axis=-1)
    return normalize_rays(rays)
