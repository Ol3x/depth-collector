# TODO

## Resume Point

The project can be resumed from repository state alone. A future Codex session does not need the old conversation ID.

Suggested restart prompt:

```text
Continue the depth-collector work from the current repo state.
Current status: the repo uses the micromamba env named depth-collector, exposes the dc CLI, supports download/extract/process/status/visualize/clean/clean_process, and the default project currently runs both TartanAir and TartanGround end to end with tiny ratios. The Tartan family abstraction is in place, both datasets download from Hugging Face, extract, enumerate paired image/depth data, convert to canonical distance plus ray_dir, write tar shards, and visualize successfully.
Next target: move to the next roadmap priorities after the Tartan family, with Hypersim first.
```

## Current State

- `tartanair` is working end to end in the default project.
- `tartanground` is working end to end in the default project.
- `dc clean_process --yes` resets only process-stage artifacts and keeps raw extracted data.
- `dc process` now rebuilds correctly after `clean_process`.
- Visualization is in place for processed datasets.

## Next Priorities

Based on the current implementation state and [pipeline_prioritization.md](/home/olx2024/repos/depth-collector/docs/research/pipeline_prioritization.md):

1. Build the `Hypersim` pipeline.
   - This is the documented `P0` target and should be the next major dataset integration.
   - Reuse the existing pipeline abstractions instead of adding dataset-specific one-off flows.

2. After `Hypersim`, move to the strongest documented `P1` non-Tartan targets.
   - `Pointcept/arkitscenes-compressed`
   - `Gen3DF/Structured3d-preprocessed`

3. Keep refining family abstractions when new datasets justify them.
   - Do not let one concrete pipeline depend on another concrete pipeline.
   - Introduce family-level abstractions only when they genuinely remove duplication.

## Near-Term Engineering Backlog

1. Add a lightweight `dc doctor` or `dc version` command for faster diagnosis of live CLI/runtime mismatches.
2. Decide whether the default compact `dc process` output should also hide the `selecting ... for process_ratio=...` line unless `--verbose` is used.
3. Remove the remaining Pillow deprecation warning in the TartanGround test fixture.
4. Add a multi-dataset smoke test that exercises both `tartanair` and `tartanground` together through the CLI.
5. Document any dataset-specific depth encoding caveats under `docs/datasets/` as more pipelines are added.
