# SOTA MDE Training Dataset Notes

Date of review: 2026-03-27

This document tracks which datasets are used by several strong modern monocular depth or monocular geometry models.

Its purpose is not to copy the authors' full training recipes. The goal is to extract a practical signal for `depth-collector`:

- which datasets recur across strong models
- which datasets seem strategically important to support
- which claims are directly documented vs only partially visible in public materials

## Reliability Legend

- `high`: directly stated in official repo, project page, or paper materials
- `medium`: visible in a public mirror of the paper appendix or supplement, but not yet verified from the official PDF text in this pass
- `low`: inference from adjacent official material; useful but not definitive

## Model Summary

| Model | What is public about training data | Confidence |
| --- | --- | --- |
| `Depth Pro` | A broad real+synthetic mixture is visible in a mirrored appendix table, including many datasets relevant to this project | medium |
| `MoGe` | Official sources say it uses a large mixed dataset; a public mirror exposes a detailed training table | medium |
| `MoGe-2` | A public mirror exposes a detailed appendix table; the paper text also says it trains on a large mixed corpus | medium |
| `Depth Anything V2` | Official sources clearly state 595K synthetic labeled images + 62M+ real pseudo-labeled images; metric models explicitly use Hypersim and Virtual KITTI 2 | high for metric stage, low-to-medium for exact foundation pretraining sources |
| `Depth Anything 3` | Official sources state all models are trained exclusively on public academic datasets, but the exact dataset list is not enumerated in the accessible materials reviewed here | low |
| `Pixel-Perfect Depth` | The paper explicitly states that the model is trained on the official Hypersim training split | high |

## Per-Model Notes

## `Depth Pro`

Publicly visible mirrored appendix material lists the following training datasets:

- 3D Ken Burns
- ARKitScenes
- Bedlam
- BlendedMVG
- DIML Indoor
- Dynamic Replica
- EDEN
- HRWSI
- Hypersim
- IRS
- MVS-Synth
- ReDWeb
- SAILVOS3D
- ScanNet
- SmartPortraits
- Synscapes
- TartanAir
- UASOL
- UnrealStereo4K
- UrbanSyn
- Virtual KITTI 2

Observed implications:

- `Hypersim`, `TartanAir`, and `UrbanSyn` clearly matter in this ecosystem
- `ARKitScenes`, `IRS`, and `ScanNet` also appear in a high-end mixed-data recipe
- the recipe blends synthetic, SfM-like, LiDAR, and indoor/outdoor sources rather than relying on one domain

## `MoGe`

Official repo and paper pages say MoGe trains on a large mixed dataset. A public mirror of the paper exposes a training table that includes:

- MegaDepth
- Taskonomy
- Waymo
- GTA-SfM
- Hypersim
- IRS
- KenBurns
- MatrixCity
- MidAir
- MVS-Synth
- Spring
- Structured3D
- Synthia
- TartanAir
- UrbanSyn
- ObjaverseV1

The mirrored snippet also strongly suggests additional real-data entries at the top of the table, but they were not fully visible in this pass.

Observed implications:

- `Hypersim`, `IRS`, `TartanAir`, and `UrbanSyn` recur again
- `Taskonomy` appears as a very large indoor source
- the model values diverse supervision types, not just clean synthetic renderings

## `MoGe-2`

A public mirror of the appendix exposes a detailed training table with:

- A2D2
- Argoverse2
- ARKitScenes
- BlendedMVS
- MegaDepth
- ScanNet++
- Taskonomy
- Waymo
- ApolloSynthetic
- EDEN
- GTA-SfM
- Hypersim
- IRS
- KenBurns
- MatrixCity
- MidAir
- MVS-Synth
- Structured3D
- Synthia
- Synscapes
- TartanAir
- UnrealStereo4K
- UrbanSyn

Observed implications:

- `Hypersim`, `TartanAir`, `UrbanSyn`, `ARKitScenes`, and `IRS` remain strategically important
- `Taskonomy` and `Structured3D` look important for indoor geometry coverage
- modern geometry models increasingly mix clean synthetic data with refined real datasets

## `Depth Anything V2`

Official sources clearly state:

- the foundation model uses `595K` synthetic labeled images
- the student models use `62M+` real pseudo-labeled images

However, the exact named sources for the full foundation-model synthetic set were not enumerated in the accessible official materials reviewed here.

For the released metric depth models, official model cards explicitly require:

- `Hypersim` for indoor metric models
- `Virtual KITTI 2` for outdoor metric models

Observed implications:

- `Hypersim` is directly confirmed as a metric-depth fine-tuning dataset in a major modern model family
- `Virtual KITTI 2` is a recurring outdoor synthetic metric source

## `Depth Anything 3`

Official sources state:

- all models are trained exclusively on public academic datasets

But the exact training dataset list was not enumerated in the accessible repo/project materials reviewed in this pass.

Observed implications:

- the project direction remains aligned with public academic data
- there is not enough source-backed detail yet to use DA3 as a dataset-ranking signal beyond that

## `Pixel-Perfect Depth`

The paper explicitly states that training uses:

- the official `Hypersim` training split

The paper motivates that choice by emphasizing the need for clean geometry and point clouds without flying 3D artifacts.

Observed implications:

- this is a strong direct signal in favor of `Hypersim`
- it also reinforces your project philosophy that geometric cleanliness matters more than image aesthetics

## Cross-Model Takeaways

### Strongly Recurring Or Strategically Reinforced

- `Hypersim`
- `TartanAir`
- `UrbanSyn`
- `ARKitScenes`
- `IRS`
- `Virtual KITTI 2`

### Notable Indoor Geometry Signals

- `Hypersim`
- `Taskonomy`
- `ARKitScenes`
- `Structured3D`
- `IRS`

### Notable Outdoor / Driving Signals

- `TartanAir`
- `UrbanSyn`
- `Virtual KITTI 2`
- `Synscapes`
- `Waymo`
- `Argoverse2`

## Practical Implication For `depth-collector`

This review strengthens the case for prioritizing datasets that are both:

- geometrically trustworthy
- recurrent in strong model training recipes

At the current stage, this especially boosts the strategic value of:

- `Hypersim`
- `TartanAir`
- `UrbanSyn`

It also suggests that future research should pay closer attention to:

- `ARKitScenes`
- `IRS`
- `Taskonomy`
- `Structured3D`
- `Virtual KITTI 2`

## Sources

- `Depth Pro` repo: https://github.com/apple/ml-depth-pro
- `Depth Pro` project page: https://machinelearning.apple.com/research/depth-pro
- mirrored `Depth Pro` appendix table: https://www.scribd.com/document/787290505/2410-02073v1
- `MoGe` repo: https://github.com/microsoft/MoGe
- `MoGe` project page: https://wangrc.site/MoGePage/
- `MoGe` CVPR page: https://cvpr.thecvf.com/virtual/2025/poster/34233
- mirrored `MoGe` paper text: https://www.researchgate.net/publication/385292347_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_Optimal_Training_Supervision
- `MoGe-2` Microsoft Research page: https://www.microsoft.com/en-us/research/publication/moge-2-accurate-monocular-geometry-with-metric-scale-and-sharp-details/?lang=zh-cn
- mirrored `MoGe-2` appendix table: https://www.researchgate.net/publication/393379073_MoGe-2_Accurate_Monocular_Geometry_with_Metric_Scale_and_Sharp_Details
- `Depth Anything V2` project page: https://depth-anything-v2.github.io/
- `Depth Anything V2` metric model card: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
- `Depth Anything V2` outdoor metric model card: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small
- `Depth Anything 3` repo: https://github.com/ByteDance-Seed/Depth-Anything-3
- `Pixel-Perfect Depth` repo: https://github.com/gangweix/pixel-perfect-depth
- `Pixel-Perfect Depth` paper PDF: https://pixel-perfect-depth.github.io/assets/main_paper_with_supp.pdf
