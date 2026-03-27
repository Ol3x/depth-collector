"""Reusable geometry utilities."""

from .camera_models import EquirectangularCameraModel, PinholeCameraModel, ProjectionType
from .depth_conversion import clip_distance_to_max_dist, distance_to_points, z_depth_to_distance
from .ray_generation import generate_equirectangular_rays, generate_pinhole_rays, normalize_rays

__all__ = [
    "EquirectangularCameraModel",
    "PinholeCameraModel",
    "ProjectionType",
    "clip_distance_to_max_dist",
    "distance_to_points",
    "generate_equirectangular_rays",
    "generate_pinhole_rays",
    "normalize_rays",
    "z_depth_to_distance",
]
