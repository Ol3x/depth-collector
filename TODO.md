# TODO

## Resume Point

The project can be resumed from repository state alone. A future Codex session does not need the old conversation ID.

Suggested restart prompt:

```text
Continue the depth-collector work from the current repo state.
Current status: docs/spec/contracts are in place, the repo uses the micromamba env named depth-collector via environment.yml, TartanAir downloads archives from Hugging Face Hub, ingests paired image/depth files, converts depth into canonical distance, writes real .pt payloads into tar shards, distinguishes download_ratio from process_ratio with minimum-selection safeguards for tiny runs, and records richer metadata plus non-fatal processing/enumeration errors.
Next target: harden the main scripts around real end-to-end smoke runs and decide whether the shard payload format should stay as single .pt blobs.
```

## Next TartanAir Increment

1. Decide whether shard payloads should stay as `.pt` blobs or move to a stricter WebDataset field layout.
2. Add a smoke-test path for the main scripts that exercises download, extraction, and processing together.
3. Revisit whether `download_ratio` and `process_ratio` should sample by stable ordering, stable hashing, or both at each stage.
4. Consider deduplicating repeated enumeration errors across reruns.
5. Add download-stage retry and failure recording similar to processing-stage handling.
