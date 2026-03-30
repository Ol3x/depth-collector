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
- Status: not targeted
- Priority tier: none
- Why it matters: large real indoor dataset with stronger long-term value than small legacy RGB-D sets
- Known issues: the currently reviewed HF package is the wrong artifact type for this repository
- HF packaging notes: eight large `arkitscenes_*.tar.gz` files totaling about 76.3 GB
- Geometry notes: official ARKitScenes sources explicitly mention raw and processed RGB-D data, camera pose, surface reconstruction, and high-resolution stationary-laser depth for a subset
- Pipeline notes: the reviewed HF package appears to be preprocessed / compressed Pointcept-style data rather than a clean raw RGB-depth-calibration source suitable for this dataset factory

## Current Decision

- This repository is not treating `Pointcept/arkitscenes-compressed` as a future pipeline target.
- The reason is not lack of strategic value; official ARKitScenes remains strategically important.
- The blocker is that the available HF package does not appear to be the right source artifact for this repo's canonical image-distance-ray pipeline.
- Supporting ARKitScenes honestly would likely require the official Apple raw source rather than the currently reviewed HF mirror.
