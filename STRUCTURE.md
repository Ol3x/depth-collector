# Repository Structure

This repository should grow around a small number of top-level concerns.

## Top-Level Layout

- `configs/`: project configuration files; initially one default config
- `dc`: primary command-line entrypoint
- `data/`: local dataset storage
- `docs/`: deeper technical documentation and dataset notes
- `src/`: library code
- `tests/`: unit and integration tests

## Intended Code Layout

Under `src/`, the code should converge toward:

- `depth_collector/core/`: abstract pipeline interfaces, shared lifecycle logic, and base data contracts
- `depth_collector/config/`: config loading and schema definitions
- `depth_collector/datasets/`: one submodule per source dataset
- `depth_collector/geometry/`: camera and geometry conversion utilities
- `depth_collector/io/`: shard writing, metadata writing, and local filesystem helpers
- `depth_collector/state/`: persistent progress tracking, caches, and error registries
- `depth_collector/validation/`: structural and semantic validation routines

The `geometry/` area should centralize reusable operations such as:

- ray-direction generation for supported camera models
- camera-convention transforms
- depth-to-distance conversion
- disparity-to-depth conversion
- depth-to-point and distance-to-point helpers

The `datasets/` area may also contain source-family abstractions. For example:

- `DatasetPipeline`
- `TartanPipeline`
- `TartanAirPipeline`
- `TartanGroundPipeline`

The intended rule is:

- family abstractions may sit between the root pipeline base class and concrete datasets
- concrete pipelines should not depend on other concrete pipelines

## Project And Dataset Directory Layout

Each config-defined multi-dataset project should live under `data/<project_name>/`.

Each dataset within that project should then live under `data/<project_name>/<dataset_name>/` with this local structure:

- `raw/`: downloaded archives or extracted original assets
- `processed/files/`: exported `.tar` shards
- `processed/metadata.json`: summary metadata for processed output
- `state/`: interruption-tolerance artifacts such as manifests, caches, and error records

The `state/` directory should be able to hold stage-specific error artifacts for download, extraction, and processing work.

This layout encodes a project-level philosophy:

- one config corresponds to one multi-dataset project
- one project gets one top-level directory under `data/`
- datasets live inside that project directory instead of sharing one flat global namespace

That separation keeps different projects from colliding on resumability state, raw archives, extracted files, and processed outputs.

## Documentation Layout

The repository should keep documentation split by role:

- root documents for stable project-wide definitions
- `docs/architecture/` for design decisions and subsystem boundaries
- `docs/datasets/` for source-dataset-specific notes and caveats
- `docs/contracts/` for canonical schemas and invariants
- `docs/research/` for cross-dataset landscape analysis, prioritization, and roadmap inputs

Important design surfaces such as the future geometry API should be documented before implementation so new agents can extend them consistently.

This separation matters because the code will be extended by agents that need a clear distinction between global rules and local dataset exceptions.

The structure should also make it cheap to add, remove, or revise dataset integrations as external data sources evolve.
