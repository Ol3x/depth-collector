# Geometry Reuse

The project should not reimplement common geometry logic inside each dataset pipeline.

## Principle

Whenever multiple datasets share the same camera-model operation, that operation should live in reusable geometry utilities or abstract helper layers.

Examples include:

- generating unit `ray_dir` for pinhole cameras
- generating unit `ray_dir` for equirectangular panoramas
- converting z-depth to radial distance
- converting disparity to depth
- converting depth to canonical distance
- changing axis conventions between source camera frames and the project camera frame

## Why This Matters

This improves:

- correctness, because fewer geometry formulas are duplicated
- reviewability, because camera math lives in one place
- speed of integration, because new pipelines can reuse existing camera-model code
- future support for stereo and disparity datasets

## Pipeline Boundary

Dataset pipelines should be responsible for:

- locating the relevant source files
- reading the source metadata
- selecting the appropriate shared geometry path
- applying dataset-specific judgment for invalid data

Shared geometry modules should be responsible for:

- camera-model math
- unit-ray generation
- depth/disparity conversion helpers
- coordinate-frame transforms
