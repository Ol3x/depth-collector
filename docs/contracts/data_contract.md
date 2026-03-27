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

## Maximum Distance

The global config should provide `max_dist`.

- `distance` must lie in the closed interval `[0, max_dist]` for valid samples
- geometry farther than `max_dist` should be projected onto the sphere of radius `max_dist`
- infinite-distance regions, such as sky, should also be encoded with `distance = max_dist`

## Coordinate Frame

`ray_dir` must be expressed in the camera frame with axis convention:

- left
- down
- forward

Pipeline implementations must document any dataset-specific conversion used to derive this representation.

These invariants should be enforced primarily through centralized pre-save validation rather than repeated ad hoc assertions throughout pipeline code.
