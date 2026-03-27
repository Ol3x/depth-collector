# Recurrent Candidate Expansion

Date of review: 2026-03-27

This note deepens the follow-up candidates highlighted by the SOTA training-data review:

- `ARKitScenes`
- `Taskonomy`
- `Structured3D`
- `Virtual KITTI 2`

The emphasis here is practical:

- is there a Hugging Face-hosted candidate package we could realistically target
- what is the likely geometric value
- how much priority should the dataset get right now

## Summary

| Dataset | Strategic value | HF packaging signal | Near-term assessment |
| --- | --- | --- | --- |
| `ARKitScenes` | high | concrete HF candidate package exists | strong candidate |
| `Structured3D` | high | concrete HF candidate package exists, but very large/gated variants | strong candidate |
| `Virtual KITTI 2` | high | importance is clear, but I did not verify a strong raw HF package in this pass | watch closely |
| `Taskonomy` | high | strategic importance is clear, but I did not verify a strong HF package in this pass | watch closely |

## `ARKitScenes`

### Why It Matters

- strongly reinforced by modern geometry model training recipes
- indoor real RGB-D with LiDAR-backed data
- large and practically relevant
- fits the indoor-first roadmap well

### Hugging Face Signal

The strongest concrete HF-hosted candidate I found is:

- `Pointcept/arkitscenes-compressed`

This repository exposes eight large `arkitscenes_*.tar.gz` files totaling about `76.3 GB`.

There are also derivative or task-specific HF datasets built from ARKitScenes, but they are less directly useful for this project.

### Geometry Signal

The ARKitScenes paper states that the dataset includes:

- raw and processed mobile-device RGB-D data
- high-resolution depth maps from a stationary laser scanner
- indoor real-world scenes at large scale

That is a strong fit for the project’s geometry-first philosophy, assuming the HF package preserves enough calibration and frame metadata.

### Main Risks

- the concrete HF package found is labeled as processed/compressed, not an official raw mirror
- file layout and calibration preservation still need inspection
- real-device depth may include failure modes that need conservative filtering

### Working Judgment

- one of the strongest future indoor real-data targets
- should be treated as a serious candidate once we inspect the compressed HF package layout

## `Structured3D`

### Why It Matters

- strongly reinforced by recurrent use in modern geometry models
- large synthetic indoor dataset
- strong fit for indoor-first development
- likely rich structure and camera metadata

### Hugging Face Signal

The strongest concrete HF-hosted candidate I found is:

- `Gen3DF/Structured3d-preprocessed`

That repository describes:

- a chunked version of `structured3d.tar.gz`
- total size about `307 GB`
- scripts to merge and extract the full archive

There are also other HF repositories derived from Structured3D, but many are downstream task conversions rather than clean raw geometry targets.

### Geometry Signal

The Structured3D paper describes:

- 3,500 synthetic house designs
- photorealistic rendering
- rich 3D structure annotations

This makes it strategically valuable for indoor geometry coverage, especially alongside Hypersim.

### Main Risks

- very large package size
- the best HF-hosted package found is a chunked raw-style archive rather than a convenient structured viewer dataset
- actual camera/depth export details still need inspection before pipeline design

### Working Judgment

- strong synthetic indoor candidate
- likely belongs near the top of the “after Hypersim” indoor backlog

## `Virtual KITTI 2`

### Why It Matters

- directly used by `Depth Anything V2` outdoor metric models
- recurrent in modern synthetic driving-depth workflows
- provides RGB, depth, segmentation, flow, and camera parameters

### Hugging Face Signal

I verified strong evidence of strategic importance from official model cards and the paper, but I did not verify a clearly suitable raw Hugging Face dataset package in this pass.

The concrete HF hits I found were either:

- paper pages
- tiny or unclear mirrors
- indirect references from model cards

So the dataset is important, but its current HF packaging status for this project remains unclear.

### Geometry Signal

The Virtual KITTI 2 paper says the dataset includes:

- RGB
- depth
- segmentation
- flow
- camera parameters and poses

That makes it a strong fit geometrically if a viable HF package is available.

### Main Risks

- HF package discovery is incomplete
- outdoor driving is not the immediate first domain
- still less urgent than the top indoor priorities

### Working Judgment

- strategically important
- keep on the shortlist, but do not promote further until a suitable HF-hosted package is verified

## `Taskonomy`

### Why It Matters

- recurrent in strong modern geometry model training recipes
- very important signal for indoor scale and diversity

### Hugging Face Signal

I did not verify a clear Hugging Face package in this pass that looks suitable for this project’s canonical conversion pipeline.

That does not reduce its strategic value, but it does matter because this project is HF-only for now.

### Geometry Signal

Taskonomy is clearly important as a broad supervision source in modern recipes, but the current pass did not verify enough HF packaging detail to rank it as an immediate pipeline target.

### Main Risks

- HF-hosted package not yet identified clearly
- original dataset structure is broad and task-heavy, so the best geometry-bearing subset may require careful targeting

### Working Judgment

- strategically important for future tracking
- keep as a research target rather than an immediate implementation target

## Practical Ranking Effect

This pass strengthens:

- `ARKitScenes`
- `Structured3D`

It keeps attention high on:

- `Virtual KITTI 2`
- `Taskonomy`

But does not yet justify promoting the latter two in the active HF-only roadmap without a clearer usable HF package.

## Sources

- `ARKitScenes` paper page: https://huggingface.co/papers/2111.08897
- `Pointcept/arkitscenes-compressed`: https://huggingface.co/datasets/Pointcept/arkitscenes-compressed/tree/main
- `Structured3D` paper page: https://huggingface.co/papers/1908.00222
- `Gen3DF/Structured3d-preprocessed`: https://huggingface.co/datasets/Gen3DF/Structured3d-preprocessed
- `Virtual KITTI 2` paper page: https://huggingface.co/papers/2001.10773
- `Depth Anything V2` outdoor metric card: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large
