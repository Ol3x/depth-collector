# Structured3D

- Dataset: `Gen3DF/Structured3d-preprocessed`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/Gen3DF/Structured3d-preprocessed
- Domain: indoor, synthetic
- Projection: likely perspective and panoramic renderings depending on the original export layout
- Scale signal: very large
- Geometry assessment: strategically strong synthetic indoor source; reinforced by recurrent use in modern geometry model recipes
- Artifact risk: likely low to moderate, but actual export details still need inspection
- Canonical conversion difficulty: medium to high
- License on HF page: the chunked mirror references original Structured3D terms; related HF packages use restrictive Structured3D-derived terms
- Status: candidate
- Priority tier: P1
- Why it matters: strong complement to Hypersim for large-scale indoor synthetic geometry
- Known issues: very large package; the most promising HF-hosted package is a chunked raw-style archive rather than a convenient structured dataset
- HF packaging notes: 308 ~1 GB chunks reconstructing an original `structured3d.tar.gz` of about 307 GB
- Geometry notes: official sources state that both panoramic and perspective imagery exist, which is strategically valuable for camera-model diversity
- Pipeline notes: attractive, but only after inspecting what image/depth/camera metadata the HF mirror preserves

## Current Roadmap Assessment

- `Structured3D` is now the recommended next target.
- The main reason is not convenience; it is that the dataset remains strategically strong and its HF packaging is at least explicit and verifiable.
- The current reviewed HF mirror appears to make the full reconstructed archive path the true `minimum_readable` download.
- That is operationally expensive, but it is still a better-defined implementation target than `MP3D-FPE`, whose gated repo blocks verification of the real source-backed acquisition path.

## Confirmed Packaging Difficulty

- The reviewed HF mirror exposes `Structured3D` as one original `structured3d.tar.gz` split into 308 approximately 1 GB fragment files.
- Those fragment files are not independently readable sample units.
- That means downloading only a subset of the split parts is not a valid source-backed `minimum_readable` path for this mirror.
- Under the repository definition of `minimum_readable`, the current reviewed HF packaging therefore makes the effective minimum-readable download approximately 307 GB.
