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
