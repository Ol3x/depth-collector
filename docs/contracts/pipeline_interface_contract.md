# Pipeline Interface Contract

This document defines the expected boundary between dataset-specific pipeline code and the shared runtime.

## Dataset Pipeline Responsibilities

Dataset pipelines should:

- define how source data is discovered
- define how raw data is downloaded or opened
- define how source items are parsed
- define dataset-specific camera metadata extraction
- decide how invalid or missing data should be treated when judgment is required

Dataset pipelines should not:

- reimplement the common end-to-end lifecycle without strong justification
- duplicate generic geometry functions
- invent dataset-local error-record formats
- invent dataset-local resumability mechanisms when shared ones are sufficient
- duplicate common canonical-output validation logic when shared validation bottlenecks are available

## Shared Runtime Responsibilities

The shared runtime should:

- orchestrate the standard pipeline stages
- manage persistent stage-aware state
- write shards
- emit metadata scaffolding
- run shared validation entrypoints
- persist stage-aware error records

The shared runtime should own the main canonical pre-save validation bottleneck.

## Processing Item Boundary

Each dataset pipeline should define a processing item granularity that is:

- meaningful for resumability
- small enough to isolate failures
- stable enough to support persistent identifiers

In many datasets, this should be one frame or one image-depth pair rather than an entire archive.

## Error Boundary

The pipeline should be able to reject a source item cleanly when:

- geometry is invalid and cannot be repaired meaningfully
- required metadata is missing
- decoding fails

In those cases, the item should be logged through the shared error-record path and excluded from processed outputs.
