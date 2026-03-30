# UrbanSyn

- Dataset: `UrbanSyn/UrbanSyn`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/UrbanSyn/UrbanSyn
- Domain: outdoor street, synthetic
- Projection: standard forward-driving perspective
- Scale signal: medium to large
- Geometry assessment: likely strong synthetic geometry
- Artifact risk: unknown from first-pass inspection
- Canonical conversion difficulty: medium
- License on HF card: CC-BY-SA-4.0
- Status: implemented
- Priority tier: P1
- Why it matters: good future street-driving dataset with depth and sky semantic support
- Known issues: lower immediate priority because the project starts indoor-first

## Current Pipeline Status

- The repository now includes a first-pass `UrbanSynPipeline`.
- The pipeline treats one aligned frame triplet as the smallest complete acquisition unit:
  - `rgb/rgb_<frame_id>.png`
  - `depth/depth_<frame_id>.exr`
  - `ss/ss_<frame_id>.png`
- Acquisition is HF-backed and downloads those files directly without an archive extraction stage.

## Current Assumptions

- EXR decoding currently requires the Python `OpenEXR` package.
- The current implementation uses config-driven camera intrinsics rather than discovering them from HF packaging.
- The default config entry records a Cityscapes-like pinhole camera as the initial intrinsic guess.
- The current implementation defaults to `depth_semantics = "distance"` and `depth_unit_meters = 1e-5`.

Those last two points should be treated as implementation-facing assumptions to revisit once the public UrbanSyn camera metadata is exercised directly in a live environment.
