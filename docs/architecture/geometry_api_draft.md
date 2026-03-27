# Geometry API Draft

This document proposes the future code surface for reusable geometry logic.

It is a design document, not an implementation commitment. The purpose is to make later code additions consistent and reviewable.

## Design Goals

- centralize camera-model math
- keep dataset pipelines thin
- support multiple projection models
- support future stereo and disparity datasets
- keep the API small and composable

## Proposed Module Areas

Under `src/depth_collector/geometry/`, the code should likely converge toward a structure similar to:

- `camera_models.py`
- `ray_generation.py`
- `depth_conversion.py`
- `disparity_conversion.py`
- `frame_conventions.py`
- `projection_types.py`

The exact filenames can change, but the responsibility split should stay clear.

## Core Data Objects

These can be plain dataclasses, lightweight classes, or typed dictionaries. They do not need to be heavyweight.

### `CameraModel`

Represents the source camera model needed to derive rays.

Possible responsibilities:

- identify projection family
- hold intrinsics
- hold distortion or model-specific parameters when supported
- expose enough information for deterministic ray generation

Expected variants over time:

- pinhole / perspective
- equirectangular
- future panoramic or fisheye variants

### `FrameConvention`

Represents an axis convention.

Purpose:

- make source and target camera frames explicit
- avoid silent sign mistakes in geometry conversions

At minimum, the project should distinguish:

- source convention
- canonical convention: left, down, forward

### `GeometrySampleInputs`

Represents the geometry-bearing source inputs available for one sample.

Possible fields:

- image size
- camera model
- source depth
- source disparity
- validity mask if present
- auxiliary semantic cues such as sky labels

This is not necessarily a final runtime object, but it is a useful design concept.

## Abstract Interfaces

These may be abstract base classes, protocols, or just a stable functional boundary.

### `RayGenerator`

Responsibility:

- produce unit `ray_dir` in the canonical camera frame

Core operation:

- `generate(camera_model, image_height, image_width) -> ray_dir`

Notes:

- output should be `(H, W, 3)`
- output should be unit-normalized
- projection-specific details should be encapsulated inside the implementation

Expected concrete implementations:

- `PinholeRayGenerator`
- `EquirectangularRayGenerator`

### `DepthConverter`

Responsibility:

- convert source depth-like quantities into canonical radial distance

Core operation:

- `to_distance(depth_like, ray_dir, camera_model, ...) -> distance`

Possible supported cases:

- radial distance already provided
- z-depth in camera coordinates
- depth definitions requiring projection-aware correction

### `DisparityConverter`

Responsibility:

- convert disparity or stereo measurements into depth before canonical distance conversion

Core operation:

- `disparity_to_depth(disparity, calibration, ...) -> depth`

This layer should exist even if stereo datasets are added later.

### `FrameTransformer`

Responsibility:

- map vectors or camera parameters between source frame conventions and the canonical frame

Core operations:

- `transform_vectors(vectors, source_convention, target_convention) -> vectors`
- `transform_camera_model(camera_model, source_convention, target_convention) -> camera_model`

## Functional Helpers

Not every geometry operation needs a class. Some should likely stay as small pure functions.

Good candidates:

- `normalize_rays`
- `clip_distance_to_max_dist`
- `distance_to_points`
- `depth_to_points`
- `safe_inverse_disparity`
- `make_pinhole_intrinsics_matrix`

## Pipeline Dependency Pattern

Dataset pipelines should ideally work like this:

1. parse source files and metadata
2. instantiate or decode the appropriate `CameraModel`
3. choose the matching shared ray-generation path
4. convert source depth or disparity into canonical `distance`
5. apply dataset-specific invalid-data policy
6. emit canonical sample fields

This keeps dataset modules focused on source-specific parsing and judgment rather than geometry formulas.

## Validation Expectations

The geometry layer should be easy to validate independently of any dataset.

Expected testable properties:

- generated `ray_dir` has unit norm
- canonical frame orientation is correct
- z-depth to radial-distance conversion behaves as expected
- disparity conversion handles edge cases safely
- clipping to `max_dist` is deterministic

## Early Implementation Priority

The first geometry surface should probably support only:

- pinhole cameras
- equirectangular panoramas
- canonical frame transforms
- z-depth to radial-distance conversion
- distance clipping to `max_dist`

That is enough to cover the current most relevant candidates while keeping the design extensible.
