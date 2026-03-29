# Config Schema Draft

This document proposes the future configuration surface for `depth-collector`.

It is a design document, not an implementation commitment. The goal is to define a configuration shape that is:

- explicit
- multi-dataset aware
- easy for agents to read and modify
- stable enough to support the shared pipeline runtime

## Design Goals

- one config file per multi-dataset project
- clear separation between project-wide settings and dataset-specific settings
- minimal hidden defaults
- easy enable/disable control per dataset
- enough runtime control for interruption-tolerant processing

## Top-Level Structure

The config should likely have top-level sections similar to:

- `project`
- `runtime`
- `output`
- `datasets`

This keeps the schema readable while allowing the project to grow.

## `project`

This section should define project-wide identity and geometric rules.

Likely fields:

- `name`
- `description`
- `max_dist`
- `train_val_split`

### Notes

- `max_dist` is global for the full multi-dataset project
- `train_val_split` should be global unless there is a strong reason to override it later

## `runtime`

This section should control execution behavior.

Likely fields:

- dataset-specific complete-unit count keys such as `environment_count`, `scene_count`, or `bundle_count`
- `process_ratio`
- `shuffle_seed`
- `resume`
- `skip_known_errors`
- `write_error_traces`
- `target_shard_size_gb`

### Notes

- dataset-specific complete-unit count keys should support development-time reduced download passes without selecting partial units
- `process_ratio` should support development-time partial processing on already-downloaded data
- `resume` should default toward interruption-tolerant behavior
- `skip_known_errors` should default toward not retrying known-bad items blindly

## `output`

This section should define output and storage conventions that are still global to the project.

Likely fields:

- `root_data_dir`
- `processed_subdir_name`
- `raw_subdir_name`
- `state_subdir_name`
- `metadata_filename`

These could remain mostly fixed in code initially, but making the design explicit helps later review.

## `datasets`

This section should be a mapping keyed by dataset identifier.

Each dataset entry should be independently enableable and should carry only the dataset-specific controls.

Likely shared fields per dataset:

- `enabled`
- `hf_dataset_id`
- `revision`
- `split` or `splits`
- candidate-unit lists such as `environments`, `scenes`, or `bundles`
- `notes`

Not every dataset will use every field, but the shape should remain recognizable across integrations.

## Dataset-Specific Extension Blocks

Each dataset entry should be allowed to define a nested dataset-specific block when needed.

Examples:

- `camera`
- `source_layout`
- `conversion`
- `selection`

The shared runtime should ignore unknown dataset-local keys unless a specific pipeline consumes them.

## Suggested Minimal Example

```yaml
project:
  name: default
  description: indoor-first exploratory build
  max_dist: 100.0
  train_val_split: 0.95

runtime:
  process_ratio: 0.01
  shuffle_seed: 0
  resume: true
  skip_known_errors: true
  write_error_traces: true
  target_shard_size_gb: 1.0

output:
  root_data_dir: data
  raw_subdir_name: raw
  processed_subdir_name: processed
  state_subdir_name: state
  metadata_filename: metadata.json

datasets:
  nyu_depth_v2:
    enabled: true
    hf_dataset_id: sayakpaul/nyu_depth_v2

  hypersim:
    enabled: false
    hf_dataset_id: GaussianWorld/Hypersim
```

## What Should Stay Global

These should stay project-wide unless there is a compelling later need otherwise:

- `max_dist`
- `train_val_split`
- shard target size
- resume policy
- default download ratio
- default process ratio

Keeping these global reduces ambiguity in the canonical corpus definition.

## What Can Be Dataset-Specific

These can reasonably vary per dataset:

- whether the dataset is enabled
- source Hugging Face dataset id
- revision or pinned source state
- subset or split selection
- camera-model hints when the source metadata is weak
- dataset-specific conversion options

## Future-Ready Fields

The schema should be designed so future additions fit naturally, including:

- stereo or disparity calibration options
- dataset-specific sky-handling hints
- project-level validation strictness
- per-dataset throttling or parallelism controls

These do not all need to exist in the first implementation, but the schema should leave room for them cleanly.
