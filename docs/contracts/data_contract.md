# Data Contract

This document defines the canonical meaning of a processed sample.

## Sample Fields

Every processed sample must contain:

- `image`
- `distance`
- `ray_dir`

## Shapes

- `image`: `(H, W, 3)`
- `distance`: `(H, W, 1)`
- `ray_dir`: `(H, W, 3)`

`H` and `W` may vary from sample to sample.

## Geometry Semantics

The target is distance to the camera center along the pixel ray, not merely z-depth in camera coordinates unless the dataset definition makes those values identical.

For every valid pixel:

- `ray_dir` should encode the normalized viewing ray direction associated with that pixel
- `distance` should encode the scalar distance traveled from the camera center to the 3D point
- the 3D point reconstruction rule is `point = distance * ray_dir`

For non-metric datasets:

- `distance` should still preserve camera-center radial semantics
- but the numeric range should be normalized into `[0, 1]` rather than pretending to be metric
- `distance = 1` should act as the far / max bucket, including sky or unknown-far regions when such semantics are available

## Maximum Distance

The global config should provide `max_dist`.

- `distance` must lie in the closed interval `[0, max_dist]` for valid samples
- geometry farther than `max_dist` should be projected onto the sphere of radius `max_dist`
- infinite-distance regions, such as sky, should also be encoded with `distance = max_dist`

For non-metric datasets, this repository currently uses the normalized variant of the same rule:

- `distance` must lie in `[0, 1]`
- `distance = 1` is the far / max bucket

## Coordinate Frame

`ray_dir` must be expressed in the camera frame with axis convention:

- left
- down
- forward

Pipeline implementations must document any dataset-specific conversion used to derive this representation.

These invariants should be enforced primarily through centralized pre-save validation rather than repeated ad hoc assertions throughout pipeline code.
