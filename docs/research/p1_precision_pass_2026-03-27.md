# P1 Precision Pass

Date of review: 2026-03-27

This note refines the strongest current `P1` candidates from an implementation-risk perspective.

Focus:

- Hugging Face packaging shape
- likely calibration preservation
- likely processing unit
- main interruption-tolerance implications

## `theairlabcmu/tartanair`

### Packaging Shape

The Hugging Face package is highly structured and operationally clear:

- top-level environment folders such as `neighborhood`, `oldtown`, `soulcity`, `seasonsforest_winter`
- each environment contains `Easy` and `Hard`
- each difficulty folder contains modality ZIPs such as:
  - `image_left.zip`
  - `image_right.zip`
  - `depth_left.zip`
  - `depth_right.zip`
  - `seg_left.zip`
  - `seg_right.zip`
  - `flow_flow.zip`
  - `flow_mask.zip`

This is much better than an opaque monolithic tarball.

### Calibration And Geometry Signal

The official TartanAir documentation is unusually explicit:

- camera model: pinhole
- image size: `640 x 640`
- focal length: `320`
- principal point: `(320, 320)`
- distortion coefficients: zero
- stereo baseline: `0.25 m`
- depth is aligned with RGB

The documentation also says the raw dataset uses 12 pinhole cameras arranged as stereo 360-degree rigs, while the Hugging Face package exposes left/right modality ZIPs per environment.

### Likely Processing Unit

A sensible processing unit is:

- one decoded frame for one camera stream

The most practical early subset may be:

- only the left-camera stream
- possibly only `front` view if the extracted file layout is camera-split internally

### Resumability Implications

- download checkpoint unit: one environment/difficulty/modality ZIP
- extraction checkpoint unit: one ZIP archive
- processing checkpoint unit: one frame identifier

### Main Risk

The main risk is not missing calibration. It is operational scale.

This is a very large dataset, so the pipeline must support:

- aggressive subset selection
- partial extraction
- careful state tracking

### Judgment

This is one of the cleanest high-value `P1` candidates because the camera model is explicit and the HF package is logically organized.

## `Pointcept/arkitscenes-compressed`

### Packaging Shape

The Hugging Face repository exposes eight large files:

- `arkitscenes_1.tar.gz`
- ...
- `arkitscenes_8.tar.gz`

Total size is about `76.3 GB`.

This is a chunked archive layout, but less transparently structured than TartanAir’s environment/modality tree.

### Calibration And Geometry Signal

Official ARKitScenes sources clearly state that the dataset includes:

- raw and processed mobile-device RGB-D data
- camera pose
- surface reconstruction
- high-resolution depth maps from stationary laser scans

That is a strong calibration-preservation signal in the original dataset.

What remains unclear in this pass is whether the HF compressed package preserves those assets in an easy-to-consume layout.

### Likely Processing Unit

Probably:

- one RGB-D frame within one capture

But this should only be confirmed after archive inspection, because ARKitScenes is capture-centric rather than simple image-pair-centric.

### Resumability Implications

- download checkpoint unit: one `arkitscenes_*.tar.gz`
- extraction checkpoint unit: one archive or one extracted capture subtree
- processing checkpoint unit: one capture/frame identifier

### Main Risk

The primary risk is metadata preservation and discoverability inside the compressed HF package.

If intrinsics, poses, or registration metadata are hard to recover, pipeline complexity rises sharply.

### Judgment

Still a very strong `P1` candidate, but with materially more inspection risk than TartanAir.

## `Gen3DF/Structured3d-preprocessed`

### Packaging Shape

The strongest HF-hosted package found is a chunked reassembly mirror:

- original file target: `structured3d.tar.gz`
- size: about `307 GB`
- 308 chunk files of about 1 GB each
- helper scripts: `merge.sh`, `download.py`, `extract.sh`

This is workable for archival access, but not friendly for selective incremental processing.

### Calibration And Geometry Signal

Official Structured3D sources make several helpful points:

- it is a large synthetic indoor dataset with 3.5K house designs
- panoramic images were available first
- perspective images were later added
- the dataset is intended for rich 3D structural modeling

This strongly suggests useful geometry and multiple projection modes, but the exact archive-level metadata layout still needs inspection.

### Likely Processing Unit

Probably:

- one rendered frame within one scene

Potentially with two distinct geometry paths:

- perspective path
- panoramic path

### Resumability Implications

- download checkpoint unit: one chunk file
- reassembly checkpoint unit: full archive assembly state
- extraction checkpoint unit: one extracted scene subtree if possible
- processing checkpoint unit: one scene/frame identifier

### Main Risk

The main risk is operational awkwardness:

- huge chunked archive
- likely expensive reassembly step
- less friendly to selective partial ingestion than a scene-structured HF mirror would be

### Judgment

Strategically strong, but operationally heavier than both TartanAir and ARKitScenes.

## Relative Risk Comparison

From lowest to highest implementation uncertainty in the current `P1` set reviewed here:

1. `theairlabcmu/tartanair`
2. `Pointcept/arkitscenes-compressed`
3. `Gen3DF/Structured3d-preprocessed`

This ranking is about packaging and inspectability, not about eventual dataset value.

## Practical Consequence

If we want one `P1` candidate to study in code sooner rather than later, `TartanAir` is currently the cleanest choice.

If we want the strongest real indoor `P1` candidate, `ARKitScenes` is the right target to inspect next.

If we want the strongest large synthetic indoor complement to `Hypersim`, `Structured3D` remains important but operationally heavy.

## Sources

- TartanAir HF tree: https://huggingface.co/datasets/theairlabcmu/tartanair/tree/main
- TartanAir environment example: https://huggingface.co/datasets/theairlabcmu/tartanair/tree/main/neighborhood/Easy
- TartanAir modality docs: https://tartanair.org/modalities.html
- TartanAir examples: https://tartanair.org/examples.html
- `Pointcept/arkitscenes-compressed`: https://huggingface.co/datasets/Pointcept/arkitscenes-compressed/tree/main
- ARKitScenes Apple page: https://machinelearning.apple.com/research/arkitscenes
- ARKitScenes GitHub: https://github.com/apple/ARKitScenes
- `Gen3DF/Structured3d-preprocessed`: https://huggingface.co/datasets/Gen3DF/Structured3d-preprocessed
- Structured3D official site: https://structured3d-dataset.org/
- Structured3D ECCV page: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/890_ECCV_2020_paper.php
