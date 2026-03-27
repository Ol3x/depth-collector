# Geometry Interface Contract

This document describes the expected boundary between dataset pipelines and shared geometry code.

## Pipeline Responsibilities

Dataset pipelines should:

- parse source files
- read source metadata and calibration
- decide which shared geometry path applies
- make conservative decisions about invalid or missing data

Dataset pipelines should not:

- duplicate standard camera-model math
- duplicate common depth-to-distance logic
- hide coordinate-frame assumptions inside dataset-local code without documentation

## Geometry Layer Responsibilities

The shared geometry layer should:

- generate unit `ray_dir` for supported camera models
- convert supported depth definitions into canonical radial distance
- convert disparity into depth when calibration is sufficient
- transform source camera conventions into the canonical frame

## Canonical Output Boundary

The shared geometry layer should help the pipeline produce:

- `ray_dir` in canonical camera frame
- `distance` clipped or mapped to `max_dist`

The final decision to reject a sample because of unresolved invalid data still belongs to the dataset pipeline.
