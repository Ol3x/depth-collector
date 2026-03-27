# Hypersim

- Dataset: `GaussianWorld/Hypersim`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/GaussianWorld/Hypersim
- Domain: indoor, synthetic
- Projection: likely standard rendered perspective cameras
- Scale signal: high, based on the original Hypersim paper
- Geometry assessment: likely very strong because the original dataset provides dense ground-truth geometry
- Artifact risk: low in principle, but HF packaging still needs inspection
- Canonical conversion difficulty: medium
- License: not clearly documented on the HF page reviewed
- Status: candidate, gated
- Priority tier: P0
- Why it matters: high-value indoor geometry target with strong scale potential
- Known issues: HF repository is gated and has no dataset card, so packaging details are not yet clear
- Pipeline notes: only worth prioritizing if the HF mirror preserves scene-level camera metadata and intrinsics information
