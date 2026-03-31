# Validation And Assertion Strategy

This document defines how the project should use asserts and validation checks.

The goal is to keep invariants explicit without filling the codebase with ad hoc assertions.

## Core Principle

The project should prefer:

- a small number of strong validation bottlenecks
- explicit validation functions
- stage-aware rejection and error recording

The project should avoid:

- scattered asserts throughout dataset-specific code
- repeated shape checks at every helper boundary
- hidden assumptions that are not documented centrally

## Where Validation Should Be Strongest

The strongest validation should happen:

1. after a sample has been converted into the canonical representation
2. before the sample is serialized or written into shard outputs

This is the main bottleneck where the system should verify that the output really satisfies the project contract.

## Canonical Pre-Save Validation

Before saving a processed sample as a PyTorch payload, the system should validate at least:

- required fields exist
- `image`, `distance`, and `ray_dir` shapes are correct
- `image`, `distance`, and `ray_dir` share the same `H` and `W`
- `ray_dir` has last dimension `3`
- `distance` has last dimension `1`
- `ray_dir` is normalized within tolerance
- `distance` lies within `[0, max_dist]` within tolerance
- arrays have expected dtypes or convertible dtypes
- values are finite where finiteness is required

This validation point protects a stable artifact contract.
Changing the serialized payload format is a product-level contract change and should not be done implicitly as part of an implementation or environment fix.

This should be one centralized validation path, not a repeated pattern in each dataset pipeline.

## Earlier Validation

Earlier stages may still validate, but they should do so selectively.

Good uses of earlier assertions or checks:

- verifying required source metadata exists before conversion
- verifying camera-model parameters are internally coherent
- verifying source files decode correctly
- verifying a shared geometry utility received inputs that are impossible to interpret safely

These earlier checks should focus on:

- unrecoverable programmer errors
- impossible geometry states
- corrupted source inputs

They should not duplicate the full canonical sample validation repeatedly.

## Assert vs Controlled Validation

The project should distinguish between two kinds of failures.

### Use `assert` Sparingly For Internal Invariants

Appropriate uses:

- impossible internal states
- programmer assumptions inside shared geometry/runtime code
- conditions that indicate a bug in the implementation, not a bad dataset sample

Examples:

- a supposedly normalized ray-generation path returns the wrong last dimension
- a state-store function receives an invalid stage enum

### Use Controlled Validation For Data And Sample Checks

Appropriate uses:

- source file corruption
- missing source metadata
- invalid sample geometry
- out-of-range canonical outputs

These should not crash the whole process by default. They should produce structured failures and error records.

## Bottleneck Design

The validation design should have a few well-defined bottlenecks.

Recommended bottlenecks:

- source-item decode validation
- geometry-conversion validation
- canonical pre-save validation
- post-write dataset-level validation

The most important one is canonical pre-save validation.

The project should also maintain a separate quantitative metrics layer so agents can inspect suspicious geometry statistically rather than visually.

## Centralization Rule

If many pipelines need the same check, it should live in shared validation code.

Examples:

- shape validation
- `ray_dir` normalization checks
- `distance` range checks
- finite-value checks
- metadata schema checks

Dataset pipelines should add only the checks that are truly specific to their source format.

## Configurability

Validation strictness should remain controllable.

The runtime should eventually support project-level controls such as:

- strict mode vs normal mode
- whether expensive numerical checks are enabled
- whether warnings should escalate to sample rejection

This keeps the code clean while allowing deeper debugging when needed.

## Error Handling Consequence

When a validation bottleneck fails for one sample:

- the sample should be rejected
- the failure should be recorded through the shared error path
- the pipeline should continue unless the failure indicates a broader runtime bug

The project should default toward isolating bad samples rather than crashing long runs.
