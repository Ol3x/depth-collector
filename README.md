# depth-collector
A dataset factory for Monocular Depth Estimation resarch.

It builds pipelines to Hugging Face Hub and other public datasets in order to create an immense public, high quality, standardized monocular depth dataset with different camera models.

## Environment

This repository should be used from the `depth-collector` micromamba environment.

Create or update it from [environment.yml](/home/olx2024/repos/depth-collector/environment.yml), then run commands through that env explicitly:

```bash
mamba env create --strict-channel-priority -f environment.yml
mamba activate depth-collector
python -m pip install -e .
python -m pytest -q
```

If the environment already exists, update it with:

```bash
mamba env update --strict-channel-priority -n depth-collector -f environment.yml
mamba activate depth-collector
python -m pip install -e .
```

The base environment is solved from conda-forge, while PyTorch is installed from the CUDA 12.8 pip wheels via `environment.yml`.
That keeps the project on real PyTorch `.pt` payloads while avoiding the broken conda PyTorch builds we hit on this machine.

## CLI

Use the repository through the `dc` command after activating the environment and installing the editable package:

```bash
dc projects
dc status default
dc download default
dc extract default
dc process default
dc process default --verbose
dc visualize default
dc clean_process default --yes
```

Project names resolve to `configs/<project>.json` by default.

If you want to point at an explicit config file instead, use `--config`:

```bash
dc process --config configs/default.json
```

Main commands:

- `dc projects`: list available project configs
- `dc status <project>`: show current archive / extraction / shard state
- `dc download <project>`: download dataset archives
- `dc extract <project>`: extract downloaded archives and remove them after extraction
- `dc process <project>`: process extracted data into shards and metadata
- `dc visualize <project>`: render diagnostic image sheets from processed samples
- `dc visualize <project> --dataset <name>`: render diagnostics for one enabled dataset only
- `dc clean_process <project> --yes`: remove processed outputs and process-stage state, while keeping raw extracted data
- `dc clean <project> --yes`: remove that project's local data directory

## Acquisition Policy

Supported dataset integrations are expected to download their own source data from Hugging Face Hub or another public mirror.

This is an explicit project rule:

- users should not be expected to manually gather source data for a finished pipeline
- temporary local staging may be used during development, but it is not the target product behavior
- implementing the acquisition path is part of building the pipeline

Visualization modes:

- `dc visualize <project> --max-samples 24`: visualize a bounded sample count
- `dc visualize <project> --all`: visualize the full processed dataset
- `dc visualize <project> --dataset virtual_kitti_2`: visualize one enabled dataset only
- `dc visualize <project> --samples-per-image 24 --sample-columns 4`: control grid density per output image

## Typical Workflow

For a normal run:

```bash
dc status default
dc download default
dc extract default
dc process default
dc visualize default
dc status default
```

For development smoke runs, the default config now uses dataset-specific `selection: "minimum_readable"` plus a small `process_ratio` so you can test the full workflow on a tiny subset first.

The default config now includes both `tartanair` and `tartanground`, each with a narrow selection and small ratios so the multi-dataset workflow stays testable. It also includes a disabled `hypersim` entry for the next high-priority integration.

If you want faster downloads and your connection can sustain it, increase `runtime.download_workers` in the project config. You can also override it per dataset with `datasets.<name>.download_workers`. The default is conservative.

Download selection is now dataset-specific through `datasets.<name>.selection`.

- `"minimum_readable"` means "choose the smallest dataset-native selection that still yields at least one readable `(image, distance, ray_dir)` sample"
- `"all"` means "use the full discovered candidate pool"
- a float in `(0, 1]` means "use that fraction of the ordered candidate pool", rounded up so a non-empty pool still yields at least one candidate

Candidate-unit lists such as `environments`, `scenes`, `bundles`, `trajectories`, `frames`, or `sequences` still define the pool being selected from.
For Tartan-family datasets, `selection` trims the shared pool of complete candidate slices, so `"minimum_readable"` selects the first complete RGB+depth slice from that pool rather than forcing every configured variation for one environment.

## Configs

Project configs live in `configs/`.

The default project is:

```text
configs/default.json
```

That means these are equivalent:

```bash
dc process default
dc process --config configs/default.json
```

## Notes

- The legacy scripts `s01_download.py`, `s02_extract_remove_archives.py`, and `s03_process.py` still exist as compatibility wrappers.
- `dc extract` removes archives and the dataset-local Hugging Face cache after successful extraction by default. Use `--keep-archives` and/or `--keep-cache` if needed.
- `dc process` is compact by default. Use `dc process <project> --verbose` for detailed cached-replay and selection progress.
- `dc clean_process <project> --yes` removes `processed/`, `visualizations/`, `processed.jsonl`, `enumeration_manifest.json`, and process-stage error records, but keeps downloaded/extracted raw data.
- `dc visualize` writes dense contact-sheet PNGs under `data/<project>/<metric-or-relative>/<dataset>/visualizations/`.
- Each sample panel currently shows RGB, same-camera reprojection, a canonical distance map, and a z-depth map computed from `ray_dir * distance`.
- `dc visualize` accepts either `--max-samples <N>` or `--all`.
- `dc visualize --dataset <name>` restricts rendering to one enabled dataset; without it, visualization runs across all enabled datasets in the selected config.
- By default, `dc visualize` behaves like `--max-samples 24 --samples-per-image 24 --sample-columns 4`.
- Datasets are now partitioned on disk as `data/<project>/metric/<dataset>/...` or `data/<project>/relative/<dataset>/...` depending on scale semantics.
- Hugging Face cache is stored per dataset under `data/<project>/<metric-or-relative>/<dataset>/.hf_cache/` instead of the global user cache.
- See [WORKFLOW.md](/home/olx2024/repos/depth-collector/WORKFLOW.md) for the project model, stage semantics, and config-state reconciliation rules.
