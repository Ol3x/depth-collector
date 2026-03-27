# Project Specification

`depth-collector` is a dataset standardization project. Its purpose is to ingest monocular depth-style datasets with different conventions and export them into one consistent representation for downstream training and evaluation.

## Primary Objective

Build a framework that can transform multiple source datasets into a shared training-ready corpus with:

- consistent geometric meaning
- consistent storage conventions
- consistent metadata
- reproducible processing steps
- interruption-tolerant execution

## Canonical Output

Every exported sample must be representable as:

- `image`: RGB tensor with shape `(H, W, 3)`
- `distance`: distance-to-camera tensor with shape `(H, W, 1)`
- `ray_dir`: ray direction tensor with shape `(H, W, 3)`

The geometric contract is:

- `ray_dir[y, x]` is the normalized direction of the camera ray associated with pixel `(y, x)`
- `distance[y, x]` is the distance traveled along that ray from the camera center to the 3D point
- the corresponding 3D point is computed as `point = distance * ray_dir`

This project standardizes on distance-to-camera, not arbitrary dataset-specific depth definitions.

## Distance Clipping And Far Geometry

The global configuration should define `max_dist`, the maximum representable distance in the processed dataset.

- `distance` values must not exceed `max_dist`
- points beyond that threshold should be mapped to distance `max_dist`
- effectively, the farthest visible geometry lies on a sphere of radius `max_dist` centered at the camera
- infinite-depth regions such as sky should also be modeled at distance `max_dist`

This rule is part of the canonical representation, not just a visualization convenience.

## Coordinate Convention

`ray_dir` should always be expressed in the camera coordinate frame.

The canonical axis convention is:

- left
- down
- forward

World-frame geometry is out of scope for the current phase.

## Invalid Data Policy

The project should not rely on a validity mask in the canonical output for now.

- if invalid or missing source values can be repaired confidently, the pipeline may repair them
- if auxiliary information such as sky masks allows a strong geometric interpretation, those pixels may be mapped to `max_dist`
- if invalid data cannot be resolved meaningfully, the affected sample should not be processed into the final dataset
- such failures should be recorded in error-tracking artifacts for later inspection

## Scope

In scope:

- dataset-specific download and extraction logic
- dataset-specific geometric conversion into the canonical representation
- sharding and metadata generation
- development-time partial processing for rapid iteration
- validation of output shape, dtype, and geometric assumptions where possible
- reusable camera-model and geometry conversion utilities shared across pipelines

Out of scope for the initial phase:

- model training
- benchmarking models
- web services or remote orchestration
- broad dependency-heavy data platforms

Likely future scope:

- disparity and stereo datasets, when they can be converted reliably into canonical `distance` plus unit `ray_dir`

## Design Principles

- Favor correctness of geometric meaning over convenience.
- Favor explicit configuration over hidden defaults.
- Favor a small dependency surface.
- Favor abstract base classes where multiple datasets share lifecycle stages.
- Favor reusable geometry utilities over dataset-specific reimplementation.
- Favor documentation that makes extension work safe for future agents.
- Favor quick integration of new datasets over trying to preserve permanent support for every historical source.

## Live Project Philosophy

The project should assume that the external dataset landscape will change.

- new datasets will appear and should be easy to integrate
- existing datasets, mirrors, or revisions may disappear
- the project does not currently aim to be a permanent archive of all source data
- if a source disappears and some functionality is lost, that is acceptable

The strategic priority is to stay current and extensible rather than archival.

## Early Success Criteria

The project foundation is successful when it can support adding a new dataset pipeline with predictable work in these areas:

- declare config in one file
- implement a dataset-specific pipeline subclass
- run a small-fraction processing pass locally
- validate the processed outputs
- inspect metadata and shard layout without ad hoc scripts

That foundation also requires a clear config schema for multi-dataset projects, not just informal config conventions.
