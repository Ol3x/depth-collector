from __future__ import annotations

import json
from pathlib import Path

from .models import DatasetConfig, OutputConfig, ProjectConfig, RootConfig, RuntimeConfig
from .validator import validate_config


def _parse_dataset_config(data: dict[str, object]) -> DatasetConfig:
    known = {"enabled", "hf_dataset_id", "revision", "split"}
    options = {key: value for key, value in data.items() if key not in known}
    return DatasetConfig(
        enabled=bool(data["enabled"]),
        hf_dataset_id=str(data["hf_dataset_id"]),
        revision=None if data.get("revision") is None else str(data["revision"]),
        split=None if data.get("split") is None else str(data["split"]),
        options=options,
    )


def load_config(path: str | Path) -> RootConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text())

    project = ProjectConfig(**data["project"])
    runtime = RuntimeConfig(**data["runtime"])
    output = OutputConfig(**data["output"])
    datasets = {
        name: _parse_dataset_config(dataset_data)
        for name, dataset_data in data["datasets"].items()
    }
    config = RootConfig(project=project, runtime=runtime, output=output, datasets=datasets)
    validate_config(config)
    return config
