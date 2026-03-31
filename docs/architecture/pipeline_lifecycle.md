# Pipeline Lifecycle

Every dataset pipeline should fit the same high-level lifecycle.

## Stages

1. Resolve configuration.
2. Prepare local dataset directories.
3. Download a configured fraction of source data.
4. Extract or materialize source files.
5. Iterate source samples.
6. Convert source geometry into the canonical sample contract.
7. Group samples into shard-sized batches.
8. Write `.pt` payloads into WebDataset `.tar` shards.
9. Emit `metadata.json`.
10. Run validation checks on the processed result.

The `.pt` shard format is part of the repository's external contract.
It should be treated as stable unless a user explicitly approves changing the artifact format and downstream compatibility expectations.

## Why This Matters

Different datasets will need different parsing logic, but they should not invent different pipeline shapes. A common lifecycle keeps:

- control flow predictable
- configuration reusable
- validation centralized
- dataset additions easier to review

The future implementation should preserve this lifecycle through a shared `DatasetPipeline` interface rather than leaving orchestration to each dataset module.
