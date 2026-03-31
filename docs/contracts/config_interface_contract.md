# Config Interface Contract

This document defines the expected boundary between configuration data and the shared runtime.

## Core Principle

The config should declare intent. The runtime should implement behavior.

That means:

- the config should describe what datasets are enabled and what global rules apply
- the runtime should interpret the config through stable defaults and shared orchestration logic

## Required Global Fields

The configuration should expose at least:

- project name
- global `max_dist`
- global `train_val_split`
- global `process_ratio`
- dataset enablement map

## Required Runtime Controls

The configuration should expose enough runtime intent to support resumable processing:

- whether resume behavior is enabled
- whether known errors should be skipped
- target shard size

## Dataset Entry Contract

Every dataset entry should support at least:

- `enabled`
- `hf_dataset_id`
- `selection`

Optional dataset-local fields should be allowed, but they should remain scoped inside that dataset entry.

Dataset-local selection fields should include:

- candidate-pool fields such as `environments`, `scenes`, `bundles`, `trajectories`, `frames`, or `sequences`

The shared `selection` field is mandatory and must support exactly:

- `"minimum_readable"`
- `"all"`
- a float in `(0, 1]`

`"minimum_readable"` means the pipeline must choose the smallest source subset that still yields at least one readable `(image, distance, ray_dir)` sample.

## Shared Runtime Expectations

The shared runtime should be able to assume:

- one config corresponds to one multi-dataset project
- the project directory on disk is derived from `project.name`
- `max_dist` is global
- disabled datasets can be ignored without side effects
- dataset-local config is passed only to the matching pipeline

## Reconciliation Expectation

Running a stage with a config should make the repository state agree with that config as far as the stage is responsible.

That means:

- stage state files support resumability but do not override missing required artifacts
- if a required archive or extracted directory is missing, the runtime should regenerate it even when stale state says the unit is complete
- if the artifact exists but the state marker is missing, the runtime may heal the state automatically

The config defines desired state. Persistent state files help reach that state efficiently; they do not redefine it.

## Validation Expectations

The config layer should eventually validate at least:

- required top-level sections exist
- required fields have correct basic types
- `process_ratio` is in a valid range
- `train_val_split` is in a valid range
- `max_dist` is strictly positive
- every dataset entry has a valid `selection`

The shared sample validator should enforce at least:

- image values are finite and lie in `[0, 1]`
- distance values are finite and lie in `[0, max_dist]`
- `ray_dir` values are finite and normalized
