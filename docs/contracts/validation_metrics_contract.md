# Validation Metrics Contract

This document defines the expected non-visual validation outputs of the project.

## Main Principle

The project should not rely only on pass/fail contract checks.

It should also compute quantitative metrics that help detect suspicious geometry and summarize dataset health.

## Sample-Level Metrics

The validation layer should eventually support metrics such as:

- `distance` summary statistics
- fraction of pixels at `max_dist`
- fraction of pixels near zero distance
- `ray_dir` norm statistics
- local geometric outlier scores

## Dataset-Level Metrics

The validation layer should aggregate metrics across runs and datasets, including:

- rejection rate
- warning rate
- distributions of key sample metrics
- counts of stage-specific failures

## Intended Use

These metrics should support:

- agent-readable debugging
- threshold tuning
- detection of systematic pipeline errors

They are not all hard rejection criteria.
