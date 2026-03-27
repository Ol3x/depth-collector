from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProjectionType(str, Enum):
    PINHOLE = "pinhole"
    EQUIRECTANGULAR = "equirectangular"


@dataclass(frozen=True)
class PinholeCameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    source_convention: str = "canonical"
    projection_type: ProjectionType = ProjectionType.PINHOLE


@dataclass(frozen=True)
class EquirectangularCameraModel:
    width: int
    height: int
    source_convention: str = "canonical"
    projection_type: ProjectionType = ProjectionType.EQUIRECTANGULAR
