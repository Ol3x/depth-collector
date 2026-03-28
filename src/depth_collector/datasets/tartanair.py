from __future__ import annotations

from .tartan import TartanArchiveUnit, TartanPipeline, TartanSourceItem
from depth_collector.geometry import z_depth_to_distance


TartanAirArchiveUnit = TartanArchiveUnit
TartanAirSourceItem = TartanSourceItem


class TartanAirPipeline(TartanPipeline):
    """Concrete TartanAir pipeline built on the shared Tartan family behavior."""


__all__ = [
    "TartanAirArchiveUnit",
    "TartanAirPipeline",
    "TartanAirSourceItem",
    "z_depth_to_distance",
]
