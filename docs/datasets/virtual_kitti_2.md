# Virtual KITTI 2

- Dataset: `ZhengGuangze/VKITTI2_vlbm`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/ZhengGuangze/VKITTI2_vlbm
- Domain: outdoor street-driving, synthetic
- Projection: perspective driving cameras
- Scale signal: moderate
- Geometry assessment: strategically important synthetic outdoor metric source
- Artifact risk: likely relatively low because of synthetic generation
- Canonical conversion difficulty: medium
- License: not yet confirmed from the reviewed derivative package
- Status: implemented
- Priority tier: P1
- Why it matters: directly used by Depth Anything V2 outdoor metric models and likely valuable for future outdoor/street support
- Known issues: the reviewed source is a derivative VLBM/Flock4D-style package rather than the official raw VKITTI2 release
- Pipeline notes: the derivative package still preserves RGBs, dense metric depths, and per-frame intrinsics/extrinsics, so it is usable for this repository

## Current Pipeline Status

- The repository now includes an initial `VirtualKITTI2Pipeline`.
- The current implementation downloads one archive, `vkitti2_vlbm.tar.gz`, as the complete HF download unit.
- Extraction materializes the dataset root `vkitti2_vlbm/` under the dataset raw directory.
- Processing then enumerates sequence folders such as `Scene06_fog` from the extracted tree.
- The default config still selects only one sequence for the smoke run, but the archive-backed config is intended to support the full dataset as well.

## Current Assumptions

- RGB is read from `<sequence>/rgbs/rgb_<frame>.jpg`.
- Depth is read from `<sequence>/depths/depth_<frame>.npz` and interpreted in meters.
- Per-frame camera intrinsics and extrinsics are loaded from `<sequence>/intrinsics.npy` and `<sequence>/extrinsics.npy`.
- The default `depth_semantics` is `distance`; this remains explicit in config.
- The default archive filename is `vkitti2_vlbm.tar.gz`.
- The default config uses:
  - `sequences: "*"`
  - `sequence_count: 1`
  for a small complete-unit smoke run, but the selectors are intended to scale cleanly to the full dataset.
