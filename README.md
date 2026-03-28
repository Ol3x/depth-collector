# depth-collector
Builds pipelines to hugging face hub datasets in order to create an immense public, high quality, standardized, monocular depth dataset.

## Environment

This repository should be used from the `depth-collector` micromamba environment.

Create or update it from [environment.yml](/home/olx2024/repos/depth-collector/environment.yml), then run commands through that env explicitly:

```bash
mamba env create -f environment.yml
mamba activate depth-collector
python -m pip install -e .
python -m pytest -q
```

If the environment already exists, update it with:

```bash
mamba env update -n depth-collector -f environment.yml
mamba activate depth-collector
python -m pip install -e .
```

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
- `dc clean_process <project> --yes`: remove processed outputs and process-stage state, while keeping raw extracted data
- `dc clean <project> --yes`: remove that project's local data directory

Visualization modes:

- `dc visualize <project> --max-samples 24`: visualize a bounded sample count
- `dc visualize <project> --all`: visualize the full processed dataset
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

For TartanAir development smoke runs, the default config already uses small `download_ratio` and `process_ratio` values so you can test the full workflow on a tiny subset first.

The default config now includes both `tartanair` and `tartanground`, each with a narrow selection and small ratios so the multi-dataset workflow stays testable.

If you want faster downloads and your connection can sustain it, increase `runtime.download_workers` in the project config. You can also override it per dataset with `datasets.<name>.download_workers`. The default is conservative.

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
- `dc extract` removes archives after successful extraction by default. Use `--keep-archives` if needed.
- `dc process` is compact by default. Use `dc process <project> --verbose` for detailed cached-replay and selection progress.
- `dc clean_process <project> --yes` removes `processed/`, `visualizations/`, `processed.jsonl`, `enumeration_manifest.json`, and process-stage error records, but keeps downloaded/extracted raw data.
- `dc visualize` writes dense contact-sheet PNGs under `data/<project>/<dataset>/visualizations/`.
- Each sample panel currently shows RGB, same-camera reprojection, and a z-depth map.
- `dc visualize` accepts either `--max-samples <N>` or `--all`.
- By default, `dc visualize` behaves like `--max-samples 24 --samples-per-image 24 --sample-columns 4`.
- The default config currently enables both `tartanair` and `tartanground` with minimal-scope selections.
- See [WORKFLOW.md](/home/olx2024/repos/depth-collector/WORKFLOW.md) for the project model, stage semantics, and config-state reconciliation rules.
