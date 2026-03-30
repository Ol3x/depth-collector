# TopAir

- Dataset: `yaraalaa0/TopAir`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/yaraalaa0/TopAir
- Domain: aerial / drone-like, synthetic
- Projection: nadir top-view perspective
- Scale signal: medium
- Geometry assessment: likely strong synthetic geometry
- Artifact risk: unknown from first-pass inspection
- Canonical conversion difficulty: medium
- License: not confirmed in this pass
- Status: implemented
- Priority tier: P1
- Why it matters: adds aerial coverage and useful viewpoint diversity
- Known issues: not indoor; exact intrinsics are not exposed in the reviewed HF layout, so the current pipeline uses explicit config-driven intrinsics

## Current Pipeline Status

- The repository now includes an initial `TopAirPipeline`.
- The current implementation treats one top-level trajectory folder as the complete download unit.
- Download is HF-backed and materializes the whole selected trajectory folder.
- There is no archive extraction stage in the current implementation.

## Current Assumptions

- RGB is read from `<trajectory>/images/`.
- Depth is read from `<trajectory>/depth/`.
- Semantic IDs are read from `<trajectory>/seg_id/` when semantic masks are enabled.
- The default depth conversion assumes the HF card's documented scale conversion `100.0 / 255.0` meters per encoded depth unit.
- The default `depth_semantics` is currently `distance`, but this remains a first-pass assumption and is exposed in config.
- The default camera intrinsics are config-driven and correspond to a `384x384` pinhole camera with `90°` horizontal field of view:
  - `fx = fy = 192`
  - `cx = cy = 192`
- Semantic sky handling is optional and defaults to treating class ID `0` as sky / invalid far field.
