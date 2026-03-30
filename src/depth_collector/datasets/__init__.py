"""Dataset-specific pipeline implementations live here."""

from .diode import DIODEPipeline
from .hypersim import HypersimPipeline
from .megadepth import MegaDepthPipeline
from .tartan import TartanPipeline
from .tartanair import TartanAirPipeline
from .tartanground import TartanGroundPipeline
from .topair import TopAirPipeline
from .tof_360 import ToF360Pipeline
from .urbansyn import UrbanSynPipeline
from .virtual_kitti_2 import VirtualKITTI2Pipeline
from .wmg_stereo import WMGStereoPipeline
from .wmg_stereo_flying import WMGStereoFlyingPipeline
from .wmg_stereo_indoor import WMGStereoIndoorPipeline
from .wmg_stereo_nature import WMGStereoNaturePipeline

PIPELINE_TYPES = {
    "hypersim": HypersimPipeline,
    "megadepth": MegaDepthPipeline,
    "diode_subset_train": DIODEPipeline,
    "tartanair": TartanAirPipeline,
    "tartanground": TartanGroundPipeline,
    "topair": TopAirPipeline,
    "tof_360": ToF360Pipeline,
    "urbansyn": UrbanSynPipeline,
    "virtual_kitti_2": VirtualKITTI2Pipeline,
    "wmg_stereo_flying": WMGStereoFlyingPipeline,
    "wmg_stereo_indoor": WMGStereoIndoorPipeline,
    "wmg_stereo_nature": WMGStereoNaturePipeline,
}

__all__ = [
    "PIPELINE_TYPES",
    "DIODEPipeline",
    "HypersimPipeline",
    "MegaDepthPipeline",
    "TartanPipeline",
    "TartanAirPipeline",
    "TartanGroundPipeline",
    "TopAirPipeline",
    "ToF360Pipeline",
    "UrbanSynPipeline",
    "VirtualKITTI2Pipeline",
    "WMGStereoPipeline",
    "WMGStereoFlyingPipeline",
    "WMGStereoIndoorPipeline",
    "WMGStereoNaturePipeline",
]
