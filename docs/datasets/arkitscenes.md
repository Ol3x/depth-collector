# ARKitScenes

- Dataset: `Pointcept/arkitscenes-compressed`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/Pointcept/arkitscenes-compressed/tree/main
- Domain: indoor, real mobile RGB-D
- Projection: likely perspective RGB-D with device calibration
- Scale signal: large
- Geometry assessment: strategically strong real indoor geometry source; reinforced by recurrent use in modern geometry model recipes
- Artifact risk: moderate; real-device depth will likely require conservative filtering
- Canonical conversion difficulty: medium to high
- License: not clearly stated on the HF package page reviewed; derivative packages reference ARKitScenes license terms
- Status: candidate
- Priority tier: P1
- Why it matters: large real indoor dataset with stronger long-term value than small legacy RGB-D sets
- Known issues: the current HF package appears processed/compressed rather than a clearly documented raw mirror
- HF packaging notes: eight large `arkitscenes_*.tar.gz` files totaling about 76.3 GB
- Geometry notes: official ARKitScenes sources explicitly mention raw and processed RGB-D data, camera pose, surface reconstruction, and high-resolution stationary-laser depth for a subset
- Pipeline notes: worth prioritizing once the internal archive layout and calibration preservation are inspected
