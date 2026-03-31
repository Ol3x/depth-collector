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
    runtime_payload = dict(data["runtime"])
    runtime_payload.setdefault("download_workers", 2)
    runtime_payload.setdefault("max_relative_far_distance_fraction", 0.98)
    runtime_payload.setdefault("min_metric_distance_std_m", 0.1)
    runtime_payload.setdefault("max_relative_distance_std", 0.3)
    runtime_payload.pop("download_ratio", None)
    if "process_ratio" not in runtime_payload and "processing_fraction" in runtime_payload:
        runtime_payload["process_ratio"] = runtime_payload.pop("processing_fraction")
    runtime = RuntimeConfig(**runtime_payload)
    output = OutputConfig(**data["output"])
    datasets = {
        name: _parse_dataset_config(dataset_data)
        for name, dataset_data in data["datasets"].items()
    }
    config = RootConfig(project=project, runtime=runtime, output=output, datasets=datasets)
    validate_config(config)
    return config
