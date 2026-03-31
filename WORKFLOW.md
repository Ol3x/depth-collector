# Workflow

This repository is organized around one explicit stage sequence:

1. `dc download <project>`
2. `dc extract <project>`
3. `dc process <project>`

Supporting commands:

- `dc status <project>`: inspect current local state
- `dc visualize <project>`: render processed-sample diagnostics
- `dc clean_process <project> --yes`: remove only process-stage artifacts
- `dc clean <project> --yes`: remove a project's local data
- `dc projects`: list project configs discoverable under `configs/`

The legacy wrappers `s01_download.py`, `s02_extract_remove_archives.py`, and `s03_process.py` still exist, but `dc` is the intended interface.

The `dc` command is provided by the package console-script entry point, so the expected setup is:

```bash
mamba activate depth-collector
python -m pip install -e .
```

Project names resolve to `configs/<project>.json` by default, with `--config <path>` available as an override on stage commands.

## Stage Responsibilities

`dc download <project>`

- reads the selected config
- determines which download units are required by that config
- downloads the required archives
- does not extract archives
- for a supported dataset integration, this stage is expected to acquire data from a public source directly rather than relying on the user to pre-stage files

`dc extract <project>`

- reads the selected config
- determines which archives must be extracted
- for each archive: extract it, remove the archive, continue to the next archive
- removes the dataset-local Hugging Face cache after a successful extraction pass unless `--keep-cache` is used
- does not process samples into output shards

`dc process <project>`

- reads the selected config
- enumerates already extracted source items
- processes the selected sample fraction into canonical outputs
- writes shards, metadata, metrics, and run reports
- defaults to compact logging; use `--verbose` for detailed preflight progress during cached replay, selection, and state coverage

`dc status <project>`

- reads the selected config
- reports compact per-dataset local state
- does not modify data

`dc clean <project> --yes`

- removes the local directory for that configured project
- is intentionally explicit and guarded by `--yes`

`dc clean_process <project> --yes`

- removes `processed/` outputs for each enabled dataset
- removes dataset `visualizations/`
- removes `state/processed.jsonl`
- removes `state/enumeration_manifest.json`
- removes `enumeration` and `processing` stage error records
- keeps downloaded archives, extracted raw data, and download/extraction state intact

Hugging Face cache policy:

- each dataset stores its HF cache under `data/<project>/<metric-or-relative>/<dataset>/.hf_cache/`
- no dataset should rely on the user's global `~/.cache/huggingface` directory
- `dc extract` removes that dataset-local HF cache by default after a successful extraction pass
- new HF-backed dataset pipelines should use the shared `DatasetPipeline` helper methods (`hf_list_repo_files`, `hf_hub_download`, `hf_snapshot_download`) so this policy applies automatically

Acquisition contract for concrete dataset pipelines:

- a finished pipeline must implement remote acquisition itself; it must not assume the user manually downloaded source files first
- optional local mirrors may still exist for tests or smoke workflows, but they are not a substitute for the remote acquisition path
- HF-backed pipelines are expected to acquire data through the shared HF helper methods on `DatasetPipeline`
- acquisition-contract tests should fail if a dataset module bypasses those shared helpers or uses ad hoc alternate download clients

`dc visualize <project>`

- reads processed samples from dataset shards
- reconstructs colored point clouds using `point = distance * ray_dir`
- renders that reconstruction from the camera at the origin
- renders a canonical distance map from `distance`
- renders a z-depth map from the reconstructed `Z` coordinate
- writes contact-sheet PNGs under `data/<project>/<metric-or-relative>/<dataset>/visualizations/`
- accepts either a bounded sample count (`--max-samples <N>`) or the full processed dataset (`--all`)
- packs sample panels into a dense grid, controlled by `--samples-per-image` and `--sample-columns`
- is the only supported visualization path for processed datasets in this repository
- must use the shared visualization module for every dataset rather than dataset-specific rendering logic
- may evolve centrally over time, but new dataset integrations are not allowed to invent alternate visualization conventions

## Project Model

One config corresponds to one multi-dataset project.

That project is stored under:

```text
data/<project_name>/
```

where `<project_name>` comes from `project.name` in the config.

Datasets are then nested under that project directory:

```text
data/<project_name>/metric/<dataset_name>/
data/<project_name>/relative/<dataset_name>/
```

Metric datasets live under `metric/`. Non-metric datasets live under `relative/`.

For geometry semantics:

- metric datasets keep metric camera distance
- non-metric datasets still use camera-distance semantics, but normalize `distance` into `[0, 1]`
- for non-metric datasets, `distance = 1` is the far / max bucket

## Reconciliation Rule

The repository treats the config as the definition of desired state.

Stage state files exist to support resumability, but they do not override missing required artifacts.

Examples:

- if a download unit is marked complete but the archive is missing, the download stage should fetch it again
- if an extraction unit is marked complete but the extracted files are missing, the extraction stage should extract again
- if the required artifact exists but the state marker is missing, the runtime may heal the state automatically

The practical goal is:

- when a user runs a stage command with a config, the current on-disk state should move toward satisfying that config

This also applies to acquisition:

- a completed dataset pipeline should make `dc download` materially move the project toward the configured dataset state
- requiring the user to fetch source files manually is not considered a finished integration

## Tiny-Run Behavior

The runtime distinguishes:

- dataset-specific `selection` controls
- `download_workers`: number of archive downloads to run concurrently
- `process_ratio`: fraction of extracted source items to process

`download_workers` may be set globally under `runtime` and overridden per dataset under `datasets.<name>.download_workers`.

For dataset download selection:

- every dataset config must provide `selection`
- `selection: "minimum_readable"` means the pipeline must select the smallest candidate set that still yields at least one readable `(image, distance, ray_dir)` sample
- `selection: "all"` means the pipeline must use the full candidate pool
- a float in `(0, 1]` means the pipeline must use that fraction of the ordered candidate pool, rounded up so a non-empty candidate pool still yields at least one useful unit
- candidate-unit lists may use `"*"` or `"all"` to request all discoverable units, with `selection` then deciding how much of that pool is used

For tiny successful processing runs:

- the repository still materializes one train shard and one val shard so the result is consumable as a minimal dataset

## Dataset-Specific Notes

Current TartanAir behavior:

- `dc download` expects archive paths like `<environment>/<difficulty>/image_left.zip` and `<environment>/<difficulty>/depth_left.zip`
- if the dataset repo has an extra leading directory, set `datasets.tartanair.hf_path_prefix`
- for offline smoke testing, `datasets.tartanair.local_archive_root` can mirror the same archive layout and act as the download source
- use `datasets.tartanair.environments` to list candidate environments and `datasets.tartanair.selection` to decide whether to keep the minimum readable slice, the whole pool, or a ratio of the pool
- configured difficulties contribute to one shared candidate pool; `"minimum_readable"` selects the first complete RGB+depth slice rather than forcing every configured difficulty for one environment

Current TartanGround behavior:

- `dc download` expects archive paths like `<environment>/Data_<version>/<trajectory>/image_<camera>.zip` and `depth_<camera>.zip`
- the default config enables `tartanground` with a minimal `AbandonedCable / omni / P0000 / lcam_front` selection
- `TartanGroundPipeline` should reuse shared Tartan-family geometry, validation, and sharding logic rather than reimplementing TartanAir internals
- use `datasets.tartanground.environments` to list candidate environments and `datasets.tartanground.selection` to decide whether to keep the minimum readable slice, the whole pool, or a ratio of the pool
- configured versions, trajectories, and camera names contribute to one shared candidate pool; `"minimum_readable"` selects the first complete RGB+depth slice rather than forcing every configured variation for one environment

Current Hypersim behavior:

- `dc download` expects per-scene archives like `<scene_name>.zip`
- extraction expects the original scene-style layout with `_detail/` camera metadata and `images/scene_<camera>_*_hdf5/` frame files
- the pipeline uses scene `meters_per_asset_unit`, per-camera keyframe positions and orientations, `position.hdf5`, and `depth_meters.hdf5` to derive canonical `ray_dir` and distance
- Hypersim stays disabled by default in `configs/default.json` until the gated HF packaging is exercised more broadly
- use `datasets.hypersim.scenes` to list candidate scenes and `datasets.hypersim.selection` to choose the minimum readable subset, full set, or a ratio of scenes

Current MegaDepth behavior:

- the current initial pipeline is non-metric, so it belongs under `relative/`
- non-metric MegaDepth exports normalize radial distance into `[0, 1]`
- `distance = 1` acts as the far / max bucket
- use `datasets.megadepth.bundles` to list candidate bundles and `datasets.megadepth.selection` to choose the minimum readable subset, full set, or a ratio of bundles

Implementation note:

- related datasets may share a family pipeline abstraction instead of duplicating logic in each concrete pipeline
- for the Tartan family, shared logic should live in a `TartanPipeline` family class rather than inside one concrete dataset pipeline
- new dataset integrations should follow [docs/contracts/new_pipeline_guidelines.md](/home/olx2024/repos/depth-collector/docs/contracts/new_pipeline_guidelines.md), especially the rule that default tiny-run selectors must still scale cleanly to full-dataset operation
