# TartanGround

- Dataset: `theairlabcmu/TartanGround`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/theairlabcmu/TartanGround
- Domain: mixed synthetic ground-robotics scenes, with strong outdoor emphasis
- Projection: perspective cameras with per-camera modality archives
- Scale signal: medium to high
- Geometry assessment: likely strong synthetic geometry source, but less documented than TartanAir in the current repo notes
- Artifact risk: likely low to medium, pending broader visual inspection
- Canonical conversion difficulty: medium
- License: not yet confirmed from the reviewed Hugging Face packaging
- Status: implemented
- Priority tier: active
- Why it matters: complements TartanAir with a more ground-robotics-specific synthetic layout and tests the shared Tartan-family abstraction
- Known issues: official total sample count and license still need to be written down explicitly; current implementation is validated on a narrow default-project slice
- HF packaging notes: the current pipeline expects archive paths like `<environment>/Data_<version>/<trajectory>/image_<camera>.zip` and `depth_<camera>.zip`
- Geometry notes: current processing treats the provided depth as z-depth and converts it into canonical radial distance using the shared Tartan family geometry path
