# Repository Growth Plan

The project should grow in layers.

## Phase 1: Foundation

- project-level specifications
- repository structure
- config conventions
- abstract pipeline interfaces
- validation strategy

## Phase 2: Core Runtime

- base pipeline implementation
- shard writing utilities
- metadata generation
- geometry helpers

## Phase 3: First Dataset Integrations

- one relatively simple dataset pipeline
- one more complex pipeline with nontrivial geometry conversion
- tests that prove the abstractions are sufficient

## Phase 4: Hardening

- stronger validation coverage
- resumability
- better logging and failure handling
- reproducibility checks

This staged approach is deliberate. The project should not accumulate dataset-specific code before its contracts are stable.
