# TODO

## Resume Point

The project can be resumed from repository state alone. A future Codex session does not need the old conversation ID.

Suggested restart prompt:

```text
Continue the depth-collector work from the current repo state.
Current status: the repo uses the micromamba env named depth-collector, exposes the dc CLI, and now has working end-to-end pipelines for TartanAir, TartanGround, Hypersim, TopAir, ToF-360, UrbanSyn, Virtual KITTI 2, DIODE subset train, WMGStereo, and MegaDepth. The current roadmap question is no longer the Tartan family or Hypersim; it is what remaining candidate dataset should be implemented next.
Next target: Structured3D. Current known blocker profile: the reviewed HF mirror appears to expose one ~307 GB reconstructed archive split into 308 ~1 GB fragments, so the true minimum-readable download is still effectively full-archive scale.
```

## Current State

- `tartanair` is working end to end in the default project.
- `tartanground` is working end to end in the default project.
- `dc clean_process --yes` resets only process-stage artifacts and keeps raw extracted data.
- `dc process` now rebuilds correctly after `clean_process`.
- Visualization is in place for processed datasets.
- `ToF-360` is working end to end, including shared equirectangular reprojection visualization.
- `MegaDepth` is implemented, but its live HF packaging can still force bundle-scale minimum-readable acquisition.

## Next Priorities

Based on the current implementation state and [pipeline_prioritization.md](/home/olx2024/repos/depth-collector/docs/research/pipeline_prioritization.md):

1. Build the `Structured3D` pipeline.
   - This is the strongest remaining target with verifiable HF packaging.
   - The main difficulty is that the reviewed HF mirror appears to require full-archive-scale acquisition even for minimum-readable correctness.

2. Keep `MP3D-FPE` below `Structured3D` until access and packaging are verified.
   - The main blocker is gated access rather than geometry value.
   - Do not promote it to the next target until the real downloadable tree can be inspected or a better HF mirror is found.

3. Keep refining family abstractions when new datasets justify them.
   - Do not let one concrete pipeline depend on another concrete pipeline.
   - Introduce family-level abstractions only when they genuinely remove duplication.

## Near-Term Engineering Backlog

1. Add a lightweight `dc doctor` or `dc version` command for faster diagnosis of live CLI/runtime mismatches.
2. Decide whether the default compact `dc process` output should also hide the `selecting ... for process_ratio=...` line unless `--verbose` is used.
3. Remove the remaining Pillow deprecation warning in the TartanGround test fixture.
4. Add a multi-dataset smoke test that exercises both `tartanair` and `tartanground` together through the CLI.
5. Document any dataset-specific depth encoding caveats under `docs/datasets/` as more pipelines are added.
