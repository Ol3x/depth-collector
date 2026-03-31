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
- `selection: "all"` and ratio selections still operate on the shared archive-backed package, `vkitti2_vlbm.tar.gz`.
- For `selection: "minimum_readable"`, the implementation now materializes a tiny local tarball that contains only one readable sample and the metadata needed for that sample.
- Extraction still materializes the dataset root `vkitti2_vlbm/` under the dataset raw directory, but for minimum-readable builds that extracted tree is a one-sample subset rather than the whole source package.

## Minimum Readable Selection

- `selection: "minimum_readable"` means the smallest source subset that still yields one readable `(image, distance, ray_dir)` sample.
- For VKITTI2, that subset is:
  - one RGB file under `<sequence>/rgbs/`
  - one paired depth file under `<sequence>/depths/`
  - sliced intrinsics and extrinsics for that frame
  - `scene_info.json`
  - a small frame-index map so the pipeline can preserve the original frame id while reading sliced camera arrays
- `selection: "all"` means all extracted sequences.
- A ratio in `(0, 1]` means the corresponding prefix of the ordered sequence pool.

## Current Assumptions

- RGB is read from `<sequence>/rgbs/rgb_<frame>.jpg`.
- Depth is read from `<sequence>/depths/depth_<frame>.npz` and interpreted in meters.
- Per-frame camera intrinsics and extrinsics are loaded from `<sequence>/intrinsics.npy` and `<sequence>/extrinsics.npy`.
- The default `depth_semantics` is `distance`; this remains explicit in config.
- The default archive filename is `vkitti2_vlbm.tar.gz`.
- The default config uses:
  - `sequences: "*"`
  - `selection: "minimum_readable"`
  for a small smoke run that still yields at least one readable `(image, distance, ray_dir)` sample without downloading the full archive payload.
