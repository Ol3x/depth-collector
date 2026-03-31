# New Pipeline Guidelines

This document defines the working standard for adding a new dataset pipeline to this repository.

It exists to keep new integrations aligned with the repository's role as a dataset factory rather than a collection of one-off ingestion scripts.

## Minimum Readable Definition

IMPORTANT:

In this repository, `selection: "minimum_readable"` is defined in terms of the active source packaging, not in terms of an ideal hypothetical packaging.

That means:

- it is the minimum amount of data that must be downloaded from the current source repository
- so that the pipeline can read and process at least one complete `(image, distance, ray_dir)` sample
- if the current repository packaging forces a large natural acquisition unit, then that large unit is still the correct `minimum_readable` download
- agents must not reinterpret `minimum_readable` as "the smallest subset we wish the upstream repo exposed"

This definition applies first to download behavior, then to extraction and processing behavior built on top of that downloaded source subset.

## Goal

When adding a new pipeline, the goal is:

- support the minimum readable sample path first
- support scaling to the full dataset later without redesigning the pipeline
- preserve the canonical repository contract:
  - image
  - canonical camera distance
  - per-pixel `ray_dir`
  - shared validation
  - shared visualization

The implementation should be good enough for a minimal end-to-end smoke run now and structurally correct for a full-dataset run later.

## Full-Scale Config Requirement

The config for a new dataset must be designed to support full-dataset download and processing, even if the default enabled settings only exercise a minimum readable smoke run.

That means:

- the default config may select the minimum readable subset for a quick test
- but the selector must also allow expansion to the full dataset
- the pipeline must not hardcode "tiny mode" as its only supported operating mode

Every dataset config must include `selection`, and every new pipeline must correctly handle all three allowed values:

- `"minimum_readable"`
- `"all"`
- a float in `(0, 1]`

Candidate-pool fields such as `scenes`, `trajectories`, `environments`, `bundles`, `frames`, or `sequences` remain dataset-specific, but `selection` is now a repository-wide contract.

`"minimum_readable"` specifically means: select the smallest source subset from the active source repository that still yields at least one readable `(image, distance, ray_dir)` sample.

## Required Implementation Process

New dataset integrations should be built in this order:

1. Inspect the actual source packaging first.
2. Identify the minimum readable sample path.
3. Confirm the geometry path and scale semantics.
4. Define explicit download units, extraction units, and source items.
5. Implement the dataset logic inside the shared runtime.
6. Add tests.
7. Add config with full-dataset-capable selectors.
8. Update dataset notes and roadmap docs.

Do not start by writing the pipeline code before the source layout and geometry path are understood well enough to justify the unit model.

## What A New Pipeline Must Do

A finished pipeline should:

- acquire source data from a public source through the shared download helpers
- support resumable download, extract, and process stages through the shared runtime
- emit canonical processed samples
- use shared sharding, validation, metadata, metrics, and visualization
- make assumptions explicit in config and documentation when the source is ambiguous

The dataset module is responsible only for dataset-specific logic:

- source discovery
- source acquisition selection
- decoding
- camera metadata extraction
- geometry interpretation
- source-item pairing
- provenance fields that are specific to the dataset

## What A New Pipeline Must Not Do

A new pipeline must not:

- invent a dataset-specific pipeline lifecycle
- bypass the shared Hugging Face helper methods for normal remote acquisition
- rely on the user to manually fetch source data for a finished integration
- invent a dataset-specific visualization method, layout, or rendering convention
- duplicate generic geometry, validation, sharding, or state logic
- make smoke-test settings the only supported operating mode
- silently guess geometry semantics without exposing the assumption in config and docs
- claim `"minimum_readable"` support without proving that the selected subset yields at least one readable `(image, distance, ray_dir)` sample
- claim a repo is violating the `minimum_readable` rule when the repo's real packaging simply makes the minimum readable download large

## Unit Model Requirement

Every new dataset pipeline should define three boundaries clearly:

- download units
- extraction units
- source items

These boundaries should be:

- meaningful for resumability
- stable enough for persistent identifiers
- small enough to isolate failures
- honest with respect to the source packaging

In many datasets, the source item should be one RGB + depth + metadata frame pair or triplet.

## Geometry Requirement

Before processing samples, the pipeline should determine what the source depth actually means:

- radial camera distance
- z-depth
- disparity
- non-metric relative depth
- or another representation

The pipeline should then convert honestly into the repository's canonical representation.

If the source semantics are not fully confirmed:

- the current assumption must be explicit
- the assumption must be configurable
- the dataset note must call out that the interpretation is provisional

## Testing Requirement

Every new pipeline should add focused tests covering at least:

- remote acquisition through shared helpers
- unit discovery and selection
- source-item enumeration
- sample building and geometry semantics
- a small end-to-end pipeline run that writes real shards and metadata

## Documentation Requirement

Every new pipeline should update:

- the dataset note under `docs/datasets/`
- the dataset inventory table
- the prioritization or roadmap docs if the integration changes what is still a target

The documentation should make these points explicit:

- what the minimum readable sample path is
- what the minimum readable download from the active source repo actually is
- what the dataset-native candidate pool is
- what the default tiny-run selector does
- how to expand the config to the full dataset
- what geometry assumptions are currently provisional

## Decision Boundary

If inspection reveals that the available source artifact is the wrong type for this repository, do not force a pipeline anyway.

Examples:

- a package that contains only preprocessed point clouds when the repo needs image-based RGB-depth geometry
- a derivative package that dropped necessary calibration or pairing information
- a natural complete unit that is so operationally heavy that it no longer fits the intended workflow

In those cases, the correct action is:

- document the blocker clearly
- remove or demote the dataset from the active target list if needed
- move on to a better candidate
