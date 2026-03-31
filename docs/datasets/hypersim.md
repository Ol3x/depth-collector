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
- Status: initial pipeline implemented, gated HF validation still pending
- Priority tier: P0
- Why it matters: high-value indoor geometry target with strong scale potential
- Known issues: HF repository is gated and has no dataset card, so packaging details are not yet clear
- Pipeline notes:
  - the current repo now includes an initial Hypersim scene pipeline
  - it assumes per-scene archives preserve `_detail/metadata_scene.csv`, camera keyframe positions/orientations, and per-frame HDF5 geometry files
  - the current conversion path derives canonical rays from world positions plus camera pose and scene scale, rather than trusting image-plane intrinsics alone
  - Hypersim camera convention is treated as right, up, backward and converted into the repo convention left, down, forward

## Minimum Readable Selection

- `selection: "minimum_readable"` means the smallest source subset that still yields one readable `(image, distance, ray_dir)` sample.
- For Hypersim, that subset is one frame plus the minimum metadata needed to reconstruct geometry:
  - one RGB preview image
  - one paired `depth_meters` file
  - one paired `depth_meters_plane` file
  - sliced camera keyframe orientation and position metadata for the chosen frame
  - scene-level metadata needed for scale and ray reconstruction
- In directory mode the implementation uses a cached manifest and direct file downloads for only those files.
- In archive mode the implementation materializes a tiny scene zip containing only those files and the sliced metadata.
- `selection: "all"` means all selected or discovered scenes.
- A ratio in `(0, 1]` means the corresponding prefix of the ordered scene pool.
