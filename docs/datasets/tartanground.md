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

## Minimum Readable Selection

- `selection: "minimum_readable"` means one readable image/depth pair from the first selected environment/version/trajectory/camera group.
- The pipeline now inspects the remote zip archives and materializes tiny local archives containing only:
  - one RGB member from `image_<camera>.zip`
  - one paired depth member from `depth_<camera>.zip`
- That keeps the existing download/extract stage model while making the downloaded source subset actually match one readable `(image, distance, ray_dir)` sample.
- `selection: "all"` means the full ordered pool across configured environments, versions, trajectories, and camera names.
- A ratio in `(0, 1]` means the corresponding prefix of that pool.
