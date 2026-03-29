"""Dataset-specific pipeline implementations live here."""

from .hypersim import HypersimPipeline
from .megadepth import MegaDepthPipeline
from .tartan import TartanPipeline
from .tartanair import TartanAirPipeline
from .tartanground import TartanGroundPipeline
from .wmg_stereo import WMGStereoPipeline
from .wmg_stereo_flying import WMGStereoFlyingPipeline
from .wmg_stereo_indoor import WMGStereoIndoorPipeline
from .wmg_stereo_nature import WMGStereoNaturePipeline

PIPELINE_TYPES = {
    "hypersim": HypersimPipeline,
    "megadepth": MegaDepthPipeline,
    "tartanair": TartanAirPipeline,
    "tartanground": TartanGroundPipeline,
    "wmg_stereo_flying": WMGStereoFlyingPipeline,
    "wmg_stereo_indoor": WMGStereoIndoorPipeline,
    "wmg_stereo_nature": WMGStereoNaturePipeline,
}

__all__ = [
    "PIPELINE_TYPES",
    "HypersimPipeline",
    "MegaDepthPipeline",
    "TartanPipeline",
    "TartanAirPipeline",
    "TartanGroundPipeline",
    "WMGStereoPipeline",
    "WMGStereoFlyingPipeline",
    "WMGStereoIndoorPipeline",
    "WMGStereoNaturePipeline",
]
