# Pipeline API Draft

This document proposes the future code surface for dataset pipelines.

It is a design document, not an implementation commitment. The goal is to make pipeline code predictable before concrete datasets are added.

## Design Goals

- keep dataset-specific code narrow
- enforce one shared lifecycle across datasets
- support interruption tolerance as a first-class concern
- make error recording systematic
- make multi-dataset projects manageable from one config

## Proposed Module Areas

Under `src/depth_collector/`, the pipeline-related code will likely need responsibilities similar to:

- `core/pipeline.py`
- `core/pipeline_types.py`
- `core/pipeline_stages.py`
- `state/manifests.py`
- `state/errors.py`
- `io/shard_writer.py`
- `validation/pipeline_validation.py`

Exact filenames can change, but the separation of concerns should remain clear.

## Core Pipeline Objects

### `DatasetPipeline`

This should be the main abstract base class for one dataset integration.

It should own:

- dataset identity
- access to the run config
- access to dataset-local directories
- orchestration of the common pipeline lifecycle

It should not own:

- duplicated geometry math
- generic shard-writing logic
- persistent-state implementation details that can be delegated

### `PipelineContext`

Represents runtime context for one dataset inside one multi-dataset project.

Possible fields:

- project config
- dataset config section
- dataset name
- local paths
- logger handle later if needed
- state manager references

This can keep method signatures smaller and more consistent.

### `SampleRecord`

Represents one canonical sample before sharding.

Suggested fields:

- sample identifier
- image
- distance
- ray_dir
- source provenance metadata

This does not need to become a heavy object, but the concept should exist explicitly.

### `ErrorRecord`

Represents one persistent failure event.

Suggested fields:

- stage
- dataset identifier
- item identifier
- error message
- optional traceback text
- timestamp
- retry count
- terminal or non-terminal status

## Abstract Lifecycle Methods

The base `DatasetPipeline` should likely expose a small set of overridable methods aligned with the shared lifecycle.

### Environment And Paths

- `prepare_directories()`

Purpose:

- ensure `raw/`, `processed/`, and `state/` locations exist

### Download

- `enumerate_download_units()`
- `download_unit(unit)`

Purpose:

- let datasets define what a downloadable unit is
- support unit-level resumability

Examples of units:

- one tar shard
- one scene archive
- one split archive

### Extraction Or Materialization

- `enumerate_extraction_units()`
- `extract_unit(unit)`

Purpose:

- allow datasets to define how raw artifacts become processable local inputs
- support skipping already extracted units

This stage should be optional when streaming directly from archives is more appropriate.

### Sample Discovery

- `enumerate_source_items()`

Purpose:

- define the smallest processing item worth checkpointing

Examples:

- one image-depth pair
- one frame in a trajectory
- one archive member

### Sample Processing

- `load_source_item(item)`
- `build_camera_model(item, loaded_item)`
- `build_sample(item, loaded_item, camera_model)`

Purpose:

- isolate source parsing from canonical conversion
- make geometry dependencies explicit

Expected result:

- a canonical `SampleRecord`
- or a controlled rejection with an `ErrorRecord`

### Output And Finalization

- `write_samples(sample_iterator)`
- `build_metadata()`
- `validate_output()`

These may be fully shared in the base implementation if possible, with only metadata hooks overridden by datasets.

## Base Lifecycle Method

The base class will likely need one top-level orchestrator, for example:

- `run()`

Conceptually it should:

1. prepare directories
2. resume state
3. execute download stage
4. execute extraction stage
5. iterate source items
6. build canonical samples
7. write shards
8. emit metadata
9. validate results
10. persist final state

This orchestration should be shared unless a dataset has a very strong reason to override a stage boundary.

## State Interfaces

The pipeline layer should interact with persistent state through explicit helpers rather than ad hoc files.

### `DownloadStateStore`

Responsibilities:

- record completed download units
- record failed download units
- expose whether a unit should be skipped, retried, or resumed

### `ExtractionStateStore`

Responsibilities:

- record completed extraction units
- record failed extraction units

### `ProcessingStateStore`

Responsibilities:

- record completed source items
- record rejected source items
- record shard-writing checkpoints if needed

### `ErrorStore`

Responsibilities:

- persist stage-aware `ErrorRecord` entries
- support appending without corrupting prior state
- support later inspection and possible retry filtering

## Shared vs Dataset-Specific Behavior

Shared behavior should include as much as possible of:

- orchestration
- state checks
- shard batching and writing
- metadata skeleton generation
- validation entrypoints

In particular, canonical sample validation should be centralized before shard writing rather than being reimplemented inside each dataset pipeline.

Dataset-specific behavior should be limited mainly to:

- locating source artifacts
- parsing source metadata
- choosing the right geometry path
- dataset-specific invalid-data judgments

## Multi-Dataset Project Boundary

One config corresponds to one multi-dataset project.

This suggests a future project runner that:

- loads the config once
- instantiates one `DatasetPipeline` per enabled dataset
- runs them in a stable order
- emits project-level status if useful

The dataset pipeline API should therefore assume it is one unit inside a larger project run, not the whole system.

## Early Implementation Priority

The first code version of this API should probably optimize for:

- one base `DatasetPipeline`
- one minimal `PipelineContext`
- one simple persistent state abstraction per stage
- append-only error recording
- shared shard writer integration

The design should stay simple enough that one or two real pipelines can validate it quickly.
