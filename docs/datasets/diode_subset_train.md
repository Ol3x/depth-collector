# DIODE Subset Train

- Dataset: `sayakpaul/diode-subset-train`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/sayakpaul/diode-subset-train
- Domain: indoor + outdoor, real
- Projection: likely standard perspective RGB-D
- Scale signal: medium subset
- Geometry assessment: promising, based on DIODE’s stated focus on accurate dense long-range depth
- Artifact risk: unknown from HF packaging alone
- Canonical conversion difficulty: medium
- License on HF card: MIT
- Status: implemented
- Priority tier: P1
- Why it matters: potentially strong real-data geometry with cross-domain value
- Known issues: subset packaging only; the reviewed HF package does not surface authoritative camera intrinsics, so the current pipeline uses explicit config-driven intrinsics
- HF packaging notes: current repo appears to expose a single `train_subset.tar.gz`
- Pipeline notes: validity masks are a strong fit for this project’s conservative invalid-data policy, and the current implementation uses one archive as the complete download unit

## Current Pipeline Status

- The repository now includes an initial `DIODEPipeline`.
- The current implementation treats `train_subset.tar.gz` as the complete download unit.
- Download is HF-backed through the shared helper and extraction is tar-based.
- Source items are paired from `*.png`, `*_depth.npy`, and `*_depth_mask.npy`.

## Current Assumptions

- The current implementation uses the reviewed subset archive only.
- The default `depth_semantics` is `distance`; this remains explicit in config.
- Invalid pixels come from the paired `*_depth_mask.npy` files and are mapped conservatively to `max_dist`.
- Camera intrinsics are currently config-driven defaults rather than discovered from the reviewed HF package:
  - `width = 1024`
  - `height = 768`
  - `fx = fy = 512`
  - `cx = 512`
  - `cy = 384`
