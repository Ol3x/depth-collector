# Rules

This file defines implementation constraints that should remain stable across the project.

## Environment And Dependencies

- The project should use one micromamba environment named `depth-collector`.
- Repository commands and tests should be run explicitly from that environment.
- The environment definition should live in `environment.yml`.
- The repository should expose a `dc` console command from that environment.
- External libraries should be minimized.
- Most data processing should rely on `numpy`.
- Additional dependencies should only be introduced when they remove substantial complexity or risk.

## Configuration

- One pipeline project should be configured from one unique config file.
- One config corresponds to one multi-dataset project.
- The project directory on disk should be derived from that config via `project.name`.
- For the current phase, the repository should support one default config file.
- Development-time partial processing must be configurable through that same config.
- The config must include `max_dist`, the maximum representable camera distance.

## Config-State Reconciliation

- When a user runs a stage script with a given config, the repository should move the on-disk state toward satisfying that config.
- Persistent state files are hints for resumability, not the source of truth over actual required artifacts.
- If a state file says a unit is complete but the required archive or extracted files are missing, the stage should redo the missing work.
- If the required artifact exists but the matching state entry is missing, the state may be healed automatically.
- The practical rule is: filesystem reality and the current config take precedence over stale resumability markers.

## Core Abstractions

- The codebase should define abstract classes for shared pipeline behavior whenever that improves consistency.
- There should be an abstract `DatasetPipeline` base class used by all dataset integrations.
- The codebase may define abstract family pipelines between `DatasetPipeline` and concrete pipelines when several datasets share a source-family structure.
- Dataset-specific implementations should override well-defined lifecycle methods rather than inventing custom end-to-end flows.
- Reusable camera-model and geometry functions should be shared across pipelines instead of reimplemented per dataset.
- Concrete pipelines should not depend on, call into, or subclass other concrete pipelines.
- Shared behavior between related concrete pipelines should live in utilities or abstract family pipeline classes instead.
- Visualization is a shared runtime responsibility, not a dataset-pipeline extension point.
- Concrete dataset pipelines must not define their own visualization methods, contact-sheet builders, or dataset-specific visualization conventions.
- All dataset visual diagnostics must flow through the shared visualization implementation used by `dc visualize`.

## Required Dataset Pipeline Responsibilities

Every concrete dataset pipeline should implement logic to:

- download source data, preferably through Hugging Face Hub when available
- extract or materialize source files and remove no-longer-needed archives when appropriate
- process source samples into the canonical representation using `numpy`-centric logic
- store processed outputs as PyTorch `.pt` payloads packaged into WebDataset `.tar` shards

This rule is strict:

- never assume the user will manually gather source data for a finished dataset integration
- manual local-data staging may be used for exploratory tests, but it is not an acceptable final pipeline state
- if a public Hugging Face Hub dataset or other public mirror exists, the pipeline should acquire the data itself
- building that acquisition path is part of the pipeline job, not user work

Concrete pipelines should delegate generic geometry operations to shared utilities whenever possible. Examples include:

- generating unit `ray_dir` for standard camera models such as pinhole
- generating unit `ray_dir` for spherical or panoramic camera models
- converting z-depth into radial distance
- converting disparity or stereo-derived depth into canonical distance when source calibration is sufficient

## Canonical Geometry Rule

This is the most important invariant in the repository.

- All processed targets must represent distance to the camera, not dataset-specific raw depth unless both are truly equivalent.
- Conversion into distance-to-camera format may require dataset-specific camera-model handling.
- `ray_dir` must be expressed in the camera coordinate frame with convention left, down, forward.
- `ray_dir` must be normalized if the reconstruction rule `point = distance * ray_dir` is used.
- `distance` values must be clipped or mapped so they do not exceed `max_dist`.
- Infinite-depth regions, including sky when present, must be represented at distance `max_dist`.
- For every valid pixel, the corresponding 3D point must be reconstructible as `point = distance * ray_dir`.
- For non-metric datasets, the repository should not pretend to have absolute scale.
- The non-metric rule is: normalize radial distance into `[0, 1]`, with `1` used as the far / max bucket.

## Invalid Data Handling

- The canonical output should not include a validity mask for now.
- Pipelines may repair invalid data only when the repair is well justified by dataset semantics or auxiliary annotations.
- If a sample contains invalid data that cannot be repaired meaningfully, it should be excluded from processed outputs.
- Excluded items should be recorded in persistent error files or logs.
- Error tracking should exist for download, extraction, and processing stages.
- Error records should include at least the file or item that caused the error and the associated error message.

## Canonical Sample Structure

One processed sample must contain:

- one RGB image with shape `(H, W, 3)` and values in `[0, 1]`
- one distance grid with shape `(H, W, 1)`
- one ray-direction tensor with shape `(H, W, 3)`

Different samples may have different `H` and `W`.

All three tensors must contain only finite values.

## Processed Dataset Layout

All projects should live under `data/`.

Each config-defined multi-dataset project should have its own directory:

- `data/<project_name>/`

Each dataset within that project should be partitioned by scale semantics first:

- `data/<project_name>/metric/<dataset_name>/`
- `data/<project_name>/relative/<dataset_name>/`

Metric datasets belong under `metric/`. Non-metric datasets belong under `relative/`.

Each dataset within that metric/relative partition should then use this local structure:

- `raw/`
- `processed/files/`
- `processed/metadata.json`

## Sharding And Metadata

- Processed data should be written as `.tar` shards in WebDataset format.
- Shards should target roughly 1 GB each unless a concrete constraint requires otherwise.
- Each processed dataset should include a `metadata.json` file.
- Metadata should include at least:
  - number of shards
  - number of files or samples per shard
  - suggested training and validation splits derived from a `train_val_split` config value

## Visualization

- The repository must maintain one official visualization path shared by all datasets.
- `dc visualize` should always render processed samples through the shared visualization module.
- Adding a new dataset pipeline does not grant permission to introduce a new visualization method, custom panel format, or dataset-local rendering entrypoint.
- Dataset-specific provenance may affect grouping labels only when the shared visualization module already supports that grouping behavior.

## Development Workflow Constraint

- During development, agents should download and process only a configurable fraction of a dataset.
- The implementation must still be structured so the same code path can scale to full-dataset processing.

## Interruption Tolerance

- The processing system should be designed to tolerate interruptions.
- Pipelines should persist enough state to avoid re-downloading, re-extracting, or re-processing completed work unnecessarily.
- Pipelines should also persist information about files or samples that repeatedly fail so that known errors are not retried blindly.
