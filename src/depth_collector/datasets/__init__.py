"""Dataset-specific pipeline implementations live here."""

from .tartan import TartanPipeline
from .tartanair import TartanAirPipeline
from .tartanground import TartanGroundPipeline

PIPELINE_TYPES = {
    "tartanair": TartanAirPipeline,
    "tartanground": TartanGroundPipeline,
}

__all__ = [
    "PIPELINE_TYPES",
    "TartanPipeline",
    "TartanAirPipeline",
    "TartanGroundPipeline",
]
