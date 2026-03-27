# TartanAir

- Dataset: `theairlabcmu/tartanair`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/theairlabcmu/tartanair/tree/main
- Domain: mixed synthetic navigation scenes, including indoor and outdoor environments
- Projection: standard perspective stereo/sequence imagery
- Scale signal: very large
- Geometry assessment: strategically strong synthetic geometry source; recurrent in multiple strong modern model training recipes
- Artifact risk: likely lower than many real datasets, but still requires actual sample inspection
- Canonical conversion difficulty: medium
- License on HF card: BSD-3-Clause
- Status: candidate
- Priority tier: P1
- Why it matters: large scale, repeated SOTA usage signal, and likely strong value for general geometric diversity
- Known issues: HF package is very large and organized as environment/trajectory archives, so interruption-tolerant download and partial processing are essential
- HF packaging notes: Hugging Face hosts a large full dataset with per-environment `Easy` and `Hard` folders and modality archives such as `image_left.zip`, `image_right.zip`, `depth_left.zip`, `depth_right.zip`, `seg_left.zip`, and flow files
- Geometry notes: official docs explicitly specify pinhole cameras with `640x640` resolution, focal length `320`, principal point `(320, 320)`, zero distortion, and stereo baseline `0.25 m`
- Pipeline notes: this is a strong future candidate for shared stereo/disparity-capable geometry tooling as well as monocular distance export
