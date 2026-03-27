from __future__ import annotations

from depth_collector.config.models import RootConfig


def validate_config(config: RootConfig) -> None:
    if config.project.max_dist <= 0.0:
        raise ValueError("project.max_dist must be strictly positive")
    if not 0.0 < config.project.train_val_split < 1.0:
        raise ValueError("project.train_val_split must lie strictly between 0 and 1")
    if not 0.0 < config.runtime.processing_fraction <= 1.0:
        raise ValueError("runtime.processing_fraction must lie in (0, 1]")
    if config.runtime.target_shard_size_gb <= 0.0:
        raise ValueError("runtime.target_shard_size_gb must be strictly positive")
    if not config.datasets:
        raise ValueError("at least one dataset entry must be configured")

    for dataset_name, dataset_config in config.datasets.items():
        if not dataset_name:
            raise ValueError("dataset names must be non-empty")
        if not dataset_config.hf_dataset_id:
            raise ValueError(f"datasets.{dataset_name}.hf_dataset_id must be non-empty")
