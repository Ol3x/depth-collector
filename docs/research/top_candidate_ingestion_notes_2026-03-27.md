# Top Candidate Ingestion Notes

Date of review: 2026-03-27

This document narrows the initial inventory into implementation-facing notes for the highest-interest candidates.

The goal is to answer:

- what the Hugging Face packaging appears to look like
- what geometry conversion issues are likely
- what pipeline risks exist
- what interruption-tolerance checkpoints will matter

## `sayakpaul/nyu_depth_v2`

### Observed Hugging Face Packaging

- The HF repository exposes a custom dataset script `nyu_depth_v2.py`.
- The data lives under `data/` as multiple `.tar` shards such as `train-000000.tar` through `train-000011.tar` plus validation shards.
- The dataset script reads `.h5` payloads from those shards.
- Each decoded item yields:
  - `image`
  - `depth_map`

### Geometry Notes

- The HF package appears to expose only RGB and depth, not explicit camera intrinsics or raw sensor metadata.
- The dataset card states that the labeled subset has been preprocessed to fill missing depth labels.
- That makes this package convenient, but also means it may be geometrically less transparent than a raw-sensor export.

### Likely Canonical Conversion Path

- interpret `depth_map` carefully before assuming it is radial distance
- if it is z-depth in the Kinect camera model, convert to radial distance using camera rays
- construct unit `ray_dir` in the camera frame using the chosen intrinsics convention
- decide how much trust to place in filled depth regions

### Main Risks

- camera intrinsics may not be packaged on Hugging Face
- filled-depth preprocessing may hide invalid regions that would otherwise be excluded
- because the package is already a processed export, some source fidelity may be lost

### Resumability Notes

- download checkpoints should track tar shards
- extraction may be avoidable if the pipeline streams `.h5` members directly from the tar files
- processing checkpoints should likely track sample identifiers by shard path plus member path

### Recommendation

- strong candidate for a first pipeline because the HF packaging is operational and the data shape is simple
- treat it as a practical bootstrap dataset, not as the final standard for geometry rigor

## `GaussianWorld/Hypersim`

### Observed Hugging Face Packaging

- The Hugging Face dataset page is gated and did not expose a usable dataset card in this review.
- The original Hypersim project is much richer than a simple RGB-depth export and includes scene-level archives with camera metadata.

### Geometry Notes

- The original Hypersim dataset includes complete camera information for every image.
- Scene layout from the original project includes:
  - per-scene metadata
  - per-trajectory camera orientations and positions
  - nontrivial camera intrinsics that can vary by scene
- Original camera convention is not the same as the project convention here:
  - Hypersim camera space uses positive x right, positive y up, and positive z away from the viewing direction
  - this project wants left, down, forward

### Likely Canonical Conversion Path

- decode HF packaging first and determine whether it preserves scene-level metadata
- if the HF package mirrors the original layout, derive unit `ray_dir` from the provided camera intrinsics system
- convert from Hypersim’s camera convention into the project’s camera convention
- determine whether the available geometric target is radial distance already or needs conversion from another representation

### Main Risks

- gated HF access blocks early inspection
- if the HF mirror drops scene-level camera metadata, the dataset may become much less useful for the canonical representation
- varying scene intrinsics and tilt-shift-like parameters make this geometrically richer but more complex than a standard pinhole export

### Resumability Notes

- checkpointing should likely happen at the scene archive level first
- processing state should then track scene, camera trajectory, and frame identifier
- failed scenes should be recorded distinctly from failed frames because metadata issues could invalidate entire scenes

### Recommendation

- still one of the highest-value targets
- do not start implementation until HF packaging and metadata preservation are verified

## `sayakpaul/diode-subset-train`

### Observed Hugging Face Packaging

- The HF repository appears to contain a single `train_subset.tar.gz`.
- Hugging Face’s own viewer fails and reports that the archive does not conform to WebDataset assumptions.
- The dataset card is minimal and does not describe the internal archive structure.

### Geometry Notes

- The official DIODE dataset uses:
  - RGB `*.png`
  - depth maps `*_depth.npy`
  - depth validity masks `*_depth_mask.npy`
- The official dataset also provides camera intrinsics through its devkit.
- DIODE is attractive because it explicitly separates invalid depth using masks instead of silently filling values.

### Likely Canonical Conversion Path

- inspect the HF tarball layout directly once implementation work begins
- use the packaged images and depth arrays as the primary source
- derive unit `ray_dir` from the camera intrinsics
- use the validity mask conservatively:
  - sky or truly infinite regions may map to `max_dist` if justified
  - unresolved invalid regions should cause sample rejection rather than silent filling

### Main Risks

- the current HF package may omit the metadata needed for robust ray construction
- because the HF package is only a subset, it may not be the best long-term DIODE target
- broken HF split parsing is a warning sign about packaging quality

### Resumability Notes

- checkpoint download state at the archive level
- checkpoint extraction by archive and by extracted top-level directory
- checkpoint processing by sample stem so a bad crop can be skipped without invalidating a full scene or scan

### Recommendation

- high-interest candidate, but only after we inspect the archive contents directly
- better as a second-wave real-data pipeline than as the very first implementation

## `COLE-Ricoh/ToF-360`

### Observed Hugging Face Packaging

- The dataset card describes per-scene folders with modalities grouped as `<scene>/<modality>`.
- Available modalities include:
  - RGB
  - Manhattan-aligned RGB
  - depth
  - XYZ images
  - normals
  - HHA
  - semantic annotations
  - room layout annotations

### Geometry Notes

- This is the cleanest match so far to the project’s distance target among the nonstandard cameras.
- The dataset card explicitly states that depth is defined as distance from the point-center of the camera in the panoramics.
- Depth PNGs are 16-bit, have maximum depth 128 m, and use 0 for missing values.
- The dataset is equirectangular, so `ray_dir` should be derived analytically from image coordinates on the sphere.

### Likely Canonical Conversion Path

- load ERP RGB plus 16-bit depth
- map valid depth values directly to `distance`, clipping to `max_dist` if needed
- convert missing depth value `0` using conservative logic:
  - if semantics or layout justify far-field treatment, map to `max_dist`
  - otherwise reject the sample or preserve only if enough valid structure remains
- generate unit `ray_dir` directly from ERP pixel coordinates in the camera frame
- reconcile the source axis convention with the project convention left, down, forward

### Main Risks

- small scale
- the exact axis convention is still not explicit in the HF card excerpt reviewed
- missing depth handling may need per-scene judgment

### Resumability Notes

- checkpointing should happen naturally at the scene directory level
- processing state should track scene plus frame name
- missing-depth or annotation parsing failures should be recorded per file, not just per scene

### Recommendation

- excellent early geometry-diversity pipeline
- especially valuable once the codebase is ready to formalize non-pinhole `ray_dir` generation

## Cross-Candidate Assessment

If the immediate goal is to bootstrap the framework with one implementable dataset and one ambitious follow-up:

- first practical target: `sayakpaul/nyu_depth_v2`
- first high-value scale target: `GaussianWorld/Hypersim`, pending HF metadata verification
- first non-pinhole target: `COLE-Ricoh/ToF-360`
- first mask-aware real-data target: `sayakpaul/diode-subset-train`, pending archive inspection

## Research Gaps

- inspect actual file trees for `Hypersim` on Hugging Face after access is available
- inspect the internal tarball layout of `diode-subset-train`
- verify axis conventions for `ToF-360`
- determine whether Hugging Face-hosted packages preserve enough metadata to generate `ray_dir` robustly for all target datasets

## Sources

- `sayakpaul/nyu_depth_v2`: https://huggingface.co/datasets/sayakpaul/nyu_depth_v2
- NYU Depth V2 project page: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- `GaussianWorld/Hypersim`: https://huggingface.co/datasets/GaussianWorld/Hypersim
- Hypersim paper: https://arxiv.org/abs/2011.02523
- Hypersim official repository: https://github.com/apple/ml-hypersim
- `sayakpaul/diode-subset-train`: https://huggingface.co/datasets/sayakpaul/diode-subset-train
- DIODE paper: https://arxiv.org/abs/1908.00463
- DIODE official site: https://diode-dataset.org/
- `COLE-Ricoh/ToF-360`: https://huggingface.co/datasets/COLE-Ricoh/ToF-360
- ToF-360 paper summary page: https://www.catalyzex.com/paper/tof-360-a-large-scale-high-resolution-dataset
