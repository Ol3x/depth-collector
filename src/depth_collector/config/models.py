from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    description: str
    max_dist: float
    train_val_split: float


@dataclass(frozen=True)
class RuntimeConfig:
    download_workers: int
    process_ratio: float
    shuffle_seed: int
    resume: bool
    skip_known_errors: bool
    write_error_traces: bool
    target_shard_size_gb: float
    max_relative_far_distance_fraction: float = 0.98
    min_metric_distance_std_m: float = 0.1
    max_relative_distance_std: float = 0.3


@dataclass(frozen=True)
class OutputConfig:
    root_data_dir: str
    raw_subdir_name: str
    processed_subdir_name: str
    state_subdir_name: str
    metadata_filename: str


@dataclass(frozen=True)
class DatasetConfig:
    enabled: bool
    hf_dataset_id: str
    revision: str | None = None
    split: str | None = None
    options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RootConfig:
    project: ProjectConfig
    runtime: RuntimeConfig
    output: OutputConfig
    datasets: dict[str, DatasetConfig]
