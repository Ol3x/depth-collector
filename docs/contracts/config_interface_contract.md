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
- global processing fraction
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

Optional dataset-local fields should be allowed, but they should remain scoped inside that dataset entry.

## Shared Runtime Expectations

The shared runtime should be able to assume:

- one config corresponds to one multi-dataset project
- `max_dist` is global
- disabled datasets can be ignored without side effects
- dataset-local config is passed only to the matching pipeline

## Validation Expectations

The config layer should eventually validate at least:

- required top-level sections exist
- required fields have correct basic types
- `processing_fraction` is in a valid range
- `train_val_split` is in a valid range
- `max_dist` is strictly positive
