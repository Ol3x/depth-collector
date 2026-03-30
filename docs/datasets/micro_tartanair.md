# Micro-TartanAir

- Dataset: `idsia-robotics/micro-tartanair`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/idsia-robotics/micro-tartanair
- Domain: synthetic navigation scenes
- Projection: perspective
- Scale signal: large row count but intentionally tiny resolution
- Geometry assessment: synthetic depth should be clean, but the reduced resolution limits strategic value
- Artifact risk: likely low
- Canonical conversion difficulty: low
- License on HF card: CC-BY-4.0 via TartanAir derivation
- Status: not targeted
- Priority tier: none
- Why it matters: operationally it could serve as a debugging or smoke-test dataset
- Known issues: overlaps with TartanAir content and the 48x48 resolution makes it unattractive for the main unified corpus

## Current Decision

- This repository is not treating `micro-tartanair` as a future pipeline target.
- The reason is not just low resolution; it is also that the package substantially overlaps with already supported `TartanAir` content.
- If a tiny Tartan-family debug source is ever needed later, it should be considered strictly as a convenience derivative, not as a strategic new corpus source.
