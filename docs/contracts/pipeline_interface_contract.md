# Pipeline Interface Contract

This document defines the expected boundary between dataset-specific pipeline code and the shared runtime.

New dataset integrations should also follow [new_pipeline_guidelines.md](/home/olx2024/repos/depth-collector/docs/contracts/new_pipeline_guidelines.md), especially the requirements around `selection`, minimum readable sample support, full-dataset-capable config design, and stopping when the available source artifact is the wrong type for this repository.

## Dataset Pipeline Responsibilities

Dataset pipelines should:

- define how source data is discovered
- define how raw data is downloaded or opened
- define how source items are parsed
- define dataset-specific camera metadata extraction
- decide how invalid or missing data should be treated when judgment is required
- implement `selection` correctly for `"minimum_readable"`, `"all"`, and ratio values in `(0, 1]`

Dataset pipelines should not:

- reimplement the common end-to-end lifecycle without strong justification
- duplicate generic geometry functions
- invent dataset-local error-record formats
- invent dataset-local resumability mechanisms when shared ones are sufficient
- duplicate common canonical-output validation logic when shared validation bottlenecks are available
- define dataset-local visualization entrypoints, panel layouts, or rendering conventions
- bypass the shared visualization module used by `dc visualize`

## Shared Runtime Responsibilities

The shared runtime should:

- orchestrate the standard pipeline stages
- manage persistent stage-aware state
- write shards
- emit metadata scaffolding
- run shared validation entrypoints
- persist stage-aware error records
- own visualization loading, rendering, grouping, and output layout

The shared runtime should own the main canonical pre-save validation bottleneck.

## Processing Item Boundary

Each dataset pipeline should define a processing item granularity that is:

- meaningful for resumability
- small enough to isolate failures
- stable enough to support persistent identifiers

In many datasets, this should be one frame or one image-depth pair rather than an entire archive.

The pipeline must also define how its candidate pool is ordered so that:

- `"minimum_readable"` yields the smallest readable `(image, distance, ray_dir)` path
- ratio selection yields a deterministic prefix of the ordered candidate pool

## Error Boundary

The pipeline should be able to reject a source item cleanly when:

- geometry is invalid and cannot be repaired meaningfully
- required metadata is missing
- decoding fails

In those cases, the item should be logged through the shared error-record path and excluded from processed outputs.

## Visualization Boundary

Visualization belongs to the shared runtime, not to dataset pipelines.

Dataset pipelines may contribute only the processed sample content and ordinary provenance fields.

They should not:

- implement their own `visualize` method
- import shared visualization helpers directly for dataset-local rendering
- produce dataset-specific visualization outputs outside the shared `dc visualize` flow
