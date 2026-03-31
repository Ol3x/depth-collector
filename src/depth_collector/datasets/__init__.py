"""Dataset-specific pipeline implementations live here."""

from __future__ import annotations

from importlib import import_module

_EXPORT_PATHS = {
    "DIODEPipeline": "depth_collector.datasets.diode:DIODEPipeline",
    "HypersimPipeline": "depth_collector.datasets.hypersim:HypersimPipeline",
    "MegaDepthPipeline": "depth_collector.datasets.megadepth:MegaDepthPipeline",
    "NYUDepthV2Pipeline": "depth_collector.datasets.nyu_depth_v2:NYUDepthV2Pipeline",
    "TartanPipeline": "depth_collector.datasets.tartan:TartanPipeline",
    "TartanAirPipeline": "depth_collector.datasets.tartanair:TartanAirPipeline",
    "TartanGroundPipeline": "depth_collector.datasets.tartanground:TartanGroundPipeline",
    "TopAirPipeline": "depth_collector.datasets.topair:TopAirPipeline",
    "ToF360Pipeline": "depth_collector.datasets.tof_360:ToF360Pipeline",
    "UrbanSynPipeline": "depth_collector.datasets.urbansyn:UrbanSynPipeline",
    "VirtualKITTI2Pipeline": "depth_collector.datasets.virtual_kitti_2:VirtualKITTI2Pipeline",
    "UnrealStereo4KPipeline": "depth_collector.datasets.unrealstereo4k:UnrealStereo4KPipeline",
    "WMGStereoPipeline": "depth_collector.datasets.wmg_stereo:WMGStereoPipeline",
    "WMGStereoFlyingPipeline": "depth_collector.datasets.wmg_stereo_flying:WMGStereoFlyingPipeline",
    "WMGStereoIndoorPipeline": "depth_collector.datasets.wmg_stereo_indoor:WMGStereoIndoorPipeline",
    "WMGStereoNaturePipeline": "depth_collector.datasets.wmg_stereo_nature:WMGStereoNaturePipeline",
}

__all__ = [
    "DIODEPipeline",
    "HypersimPipeline",
    "MegaDepthPipeline",
    "NYUDepthV2Pipeline",
    "TartanPipeline",
    "TartanAirPipeline",
    "TartanGroundPipeline",
    "TopAirPipeline",
    "ToF360Pipeline",
    "UrbanSynPipeline",
    "VirtualKITTI2Pipeline",
    "UnrealStereo4KPipeline",
    "WMGStereoPipeline",
    "WMGStereoFlyingPipeline",
    "WMGStereoIndoorPipeline",
    "WMGStereoNaturePipeline",
]


def __getattr__(name: str) -> object:
    target = _EXPORT_PATHS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, _, attr_name = target.partition(":")
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
