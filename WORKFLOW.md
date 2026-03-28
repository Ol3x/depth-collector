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

`dc extract <project>`

- reads the selected config
- determines which archives must be extracted
- for each archive: extract it, remove the archive, continue to the next archive
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

`dc visualize <project>`

- reads processed samples from dataset shards
- reconstructs colored point clouds using `point = distance * ray_dir`
- renders that reconstruction from the camera at the origin
- renders a z-depth map from the reconstructed `Z` coordinate
- writes contact-sheet PNGs under `data/<project>/<dataset>/visualizations/`
- accepts either a bounded sample count (`--max-samples <N>`) or the full processed dataset (`--all`)
- packs sample panels into a dense grid, controlled by `--samples-per-image` and `--sample-columns`

## Project Model

One config corresponds to one multi-dataset project.

That project is stored under:

```text
data/<project_name>/
```

where `<project_name>` comes from `project.name` in the config.

Datasets are then nested under that project directory:

```text
data/<project_name>/<dataset_name>/
```

## Reconciliation Rule

The repository treats the config as the definition of desired state.

Stage state files exist to support resumability, but they do not override missing required artifacts.

Examples:

- if a download unit is marked complete but the archive is missing, the download stage should fetch it again
- if an extraction unit is marked complete but the extracted files are missing, the extraction stage should extract again
- if the required artifact exists but the state marker is missing, the runtime may heal the state automatically

The practical goal is:

- when a user runs a stage command with a config, the current on-disk state should move toward satisfying that config

## Tiny-Run Behavior

The runtime distinguishes:

- `download_ratio`: fraction of remote archive units to download
- `download_workers`: number of archive downloads to run concurrently
- `process_ratio`: fraction of extracted source items to process

`download_workers` may be set globally under `runtime` and overridden per dataset under `datasets.<name>.download_workers`.

For both:

- if the requested ratio would otherwise yield zero useful work, the runtime still selects a minimum of one useful unit

For tiny successful processing runs:

- the repository still materializes one train shard and one val shard so the result is consumable as a minimal dataset

## Dataset-Specific Notes

Current TartanAir behavior:

- `dc download` expects archive paths like `<environment>/<difficulty>/image_left.zip` and `<environment>/<difficulty>/depth_left.zip`
- if the dataset repo has an extra leading directory, set `datasets.tartanair.hf_path_prefix`
- for offline smoke testing, `datasets.tartanair.local_archive_root` can mirror the same archive layout and act as the download source

Current TartanGround behavior:

- `dc download` expects archive paths like `<environment>/Data_<version>/<trajectory>/image_<camera>.zip` and `depth_<camera>.zip`
- the default config enables `tartanground` with a minimal `AbandonedCable / omni / P0000 / lcam_front` selection
- `TartanGroundPipeline` should reuse shared Tartan-family geometry, validation, and sharding logic rather than reimplementing TartanAir internals

Implementation note:

- related datasets may share a family pipeline abstraction instead of duplicating logic in each concrete pipeline
- for the Tartan family, shared logic should live in a `TartanPipeline` family class rather than inside one concrete dataset pipeline
