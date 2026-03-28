from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from depth_collector.config import DatasetConfig, RootConfig


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    raw: Path
    processed: Path
    processed_files: Path
    state: Path
    metadata: Path
    run_report: Path


@dataclass(frozen=True)
class PipelineContext:
    config: RootConfig
    dataset_name: str
    dataset_config: DatasetConfig
    paths: DatasetPaths


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image: np.ndarray
    distance: np.ndarray
    ray_dir: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorRecord:
    stage: str
    dataset_name: str
    item_id: str
    error_message: str
    traceback_text: str | None = None
    retry_count: int = 0
    terminal: bool = True


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: tuple[ValidationIssue, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMetricsSummary:
    sample_count: int
    metric_means: dict[str, float] = field(default_factory=dict)
    metric_mins: dict[str, float] = field(default_factory=dict)
    metric_maxs: dict[str, float] = field(default_factory=dict)
