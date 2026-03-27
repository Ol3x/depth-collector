# Validation Metrics Strategy

This document defines the project’s non-visual validation strategy.

Because LLM agents cannot reliably inspect images and depth maps by eye, the project should provide machine-checkable statistics that expose whether processed samples are likely valid.

## Goal

The validation system should give agents feedback that is:

- quantitative
- comparable across samples and datasets
- informative enough to catch broken geometry without human visualization

## Validation Layers

The validation stack should have three layers.

### Layer 1: Hard Contract Checks

These are the required canonical validations already defined elsewhere:

- shape correctness
- dtype compatibility
- finite-value requirements
- `distance` range compliance
- `ray_dir` normalization

These answer:

- is the sample structurally valid

### Layer 2: Sample-Level Metrics

These should characterize whether a sample looks geometrically plausible.

Examples:

- fraction of pixels with `distance == max_dist` within tolerance
- fraction of pixels with near-zero distance
- mean, std, min, max of `distance`
- mean, std, min, max of `ray_dir` norm
- fraction of non-finite values before rejection
- local depth-gradient statistics
- local 3D neighbor-distance statistics
- isolated-point or outlier score in reconstructed local point neighborhoods

These answer:

- does the sample look suspicious even if it satisfies the hard contract

### Layer 3: Dataset-Level Aggregation

These should summarize a pipeline run.

Examples:

- distribution of `distance` summary stats
- distribution of `max_dist` saturation fraction
- distribution of ray-norm error
- sample rejection rate
- per-stage failure counts
- fraction of samples flagged by suspicious-geometry heuristics

These answer:

- does the dataset output look globally healthy
- did the pipeline likely make a systematic mistake

## Flying-Point Detection Heuristics

One of the main project risks is spurious 3D points floating between surfaces.

The validation system should therefore include local geometric heuristics aimed at catching suspicious isolated points.

Possible statistics:

- for each pixel, reconstruct a local 3D point
- compare it to neighboring reconstructed points
- estimate local neighbor-distance dispersion
- estimate whether a point is isolated relative to its neighborhood
- aggregate robust statistics such as median and high-percentile outlier score

These should remain heuristics, not absolute truth, but they are valuable because they can catch obvious geometric corruption without visual inspection.

## Image-Geometry Relationship Metrics

The project should also consider weak alignment metrics between image structure and geometry structure.

Examples:

- local image-gradient magnitude vs local depth-gradient magnitude
- fraction of extreme depth discontinuities occurring in very low-texture regions
- unusual mismatch between image edges and geometry edges

These should be treated as soft diagnostics, not hard rejection rules.

## Output Artifacts

The validation system should emit machine-readable summaries.

Likely outputs:

- per-sample metric records for flagged samples only, or for debug mode
- per-shard summary files
- per-dataset aggregated summary files
- human-readable textual reports

This is important because agents need compact evidence, not raw dumps of every pixel statistic.

## Decision Policy

The validation metrics layer should distinguish:

- hard failures
- suspicious warnings
- informational summaries

Suggested policy:

- hard failures reject a sample
- suspicious warnings do not automatically reject a sample unless thresholds are clearly exceeded
- dataset-level summaries help detect systemic issues and guide later threshold tuning

## Configurability

Metric computation should remain controllable because some checks may be expensive.

Potential runtime controls:

- enable or disable expensive sample metrics
- choose whether to compute point-cloud neighborhood diagnostics
- set warning thresholds
- set escalation thresholds from warning to rejection

## Early Implementation Priority

The first useful metrics layer should probably include:

- `distance` min/mean/max
- fraction at `max_dist`
- fraction near zero
- ray-norm statistics
- simple local 3D neighbor-distance dispersion
- per-dataset aggregation of those metrics

That is enough to start giving LLM agents meaningful feedback before more advanced heuristics are added.
