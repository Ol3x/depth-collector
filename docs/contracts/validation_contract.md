# Validation Contract

This document defines the expected validation boundary in the project.

## Main Rule

The project should centralize most canonical output checks in shared validation code rather than spreading them across dataset pipelines.

## Required Pre-Save Validation

Before writing a processed sample, shared validation should check at least:

- required fields exist
- canonical shapes are correct
- shared spatial dimensions match
- `ray_dir` is normalized within tolerance
- `distance` is within `[0, max_dist]` within tolerance
- values satisfy finiteness requirements

## Dataset-Specific Validation

Dataset pipelines may add source-specific checks when needed, but they should not duplicate the common canonical checks.

## Quantitative Diagnostics

Beyond hard pass/fail checks, the shared validation layer should also support quantitative sample and dataset metrics that help detect suspicious geometry without manual visualization.

## Assertion Philosophy

Use plain `assert` primarily for internal implementation invariants and programmer errors.

Use controlled validation with structured failures for dataset-driven problems and invalid samples.
