# MP3D-FPE

- Dataset: `EnriqueSolarte/mp3d_fpe`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/EnriqueSolarte/mp3d_fpe
- Domain: indoor, 360 imagery with depth
- Projection: equirectangular panorama
- Scale signal: large storage requirement
- Geometry assessment: promising
- Artifact risk: unknown
- Canonical conversion difficulty: high
- License on HF card: MIT, with additional upstream terms mentioned
- Status: candidate, gated
- Priority tier: P2
- Why it matters: important for panoramic indoor support
- Known issues: gated access, large storage requirement, more complex than early pinhole-style pipelines

## Current Roadmap Assessment

- `MP3D-FPE` should not be the next implementation target yet.
- The main blocker is not only size; it is that the gated HF repo prevents verifying the real downloadable file tree.
- Public evidence suggests one `{SCENE_ID}/{SCENE_VERSION}` directory may be the natural minimum-readable unit, but that is still not verified from actual downloadable contents.
- Until that gated source can be inspected or a better HF mirror is found, the dataset remains a lower-priority candidate than `Structured3D`.
