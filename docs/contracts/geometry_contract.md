# Geometry Contract

This document defines the expectations for reusable geometry logic in the repository.

## Shared Operations

The repository should provide shared implementations for recurring operations where possible.

Examples:

- pinhole pixel-to-ray conversion
- equirectangular pixel-to-ray conversion
- z-depth to radial-distance conversion
- disparity to depth conversion
- canonical camera-frame conversion

## Output Expectations

Geometry utilities should support producing:

- unit `ray_dir` in the project camera frame
- canonical `distance`
- intermediate forms only when needed for a justified conversion path

## Stereo And Disparity Readiness

The architecture should be designed so that future stereo or disparity datasets can plug into the same canonical pipeline.

Typical path:

1. derive depth from disparity using source calibration
2. derive unit `ray_dir` from the source camera model
3. convert depth into canonical radial distance when depth is not already radial distance

This future support should influence the abstraction design even if such datasets are not implemented immediately.
