from __future__ import annotations

import json
from pathlib import Path

from depth_collector.config import RootConfig, load_config
from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.datasets import PIPELINE_TYPES


def build_enabled_pipelines(config: RootConfig) -> list[DatasetPipeline]:
    pipelines: list[DatasetPipeline] = []
    for dataset_name, dataset_config in config.datasets.items():
        if not dataset_config.enabled:
            continue
        pipeline_type = PIPELINE_TYPES.get(dataset_name)
        if pipeline_type is None:
            raise ValueError(f"no pipeline registered for enabled dataset: {dataset_name}")
        pipelines.append(pipeline_type(config, dataset_name))
    return pipelines


def load_enabled_pipelines(config_path: str = "configs/default.json") -> list[DatasetPipeline]:
    return build_enabled_pipelines(load_config(config_path))


def resolve_project_config_path(project_or_path: str, configs_dir: str | Path = "configs") -> Path:
    candidate = Path(project_or_path)
    if candidate.exists():
        return candidate
    config_path = Path(configs_dir) / f"{project_or_path}.json"
    if config_path.exists():
        return config_path
    raise FileNotFoundError(f"no config found for project '{project_or_path}' under {configs_dir}")


def list_project_configs(configs_dir: str | Path = "configs") -> list[tuple[str, Path]]:
    configs_root = Path(configs_dir)
    if not configs_root.exists():
        return []
    projects: list[tuple[str, Path]] = []
    for path in sorted(configs_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        project = payload.get("project", {})
        if not isinstance(project, dict):
            continue
        project_name = project.get("name")
        if isinstance(project_name, str) and project_name:
            projects.append((project_name, path))
    return projects
