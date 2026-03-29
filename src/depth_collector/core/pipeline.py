from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Iterable, Iterator

from tqdm.auto import tqdm

from depth_collector.config import RootConfig
from depth_collector.core.pipeline_types import (
    DatasetPaths,
    ErrorRecord,
    PipelineContext,
    SampleRecord,
    ValidationReport,
)
from depth_collector.state import FileDownloadStateStore, FileExtractionStateStore, FileProcessingStateStore, JsonlErrorStore
from depth_collector.validation.metrics import summarize_metrics
from depth_collector.validation.validator import CanonicalSampleValidator


class DatasetPipeline(ABC):
    """Shared lifecycle for dataset-specific processing."""

    def __init__(self, config: RootConfig, dataset_name: str) -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_config = config.datasets[dataset_name]
        self.paths = self._build_paths()
        self.context = PipelineContext(
            config=config,
            dataset_name=dataset_name,
            dataset_config=self.dataset_config,
            paths=self.paths,
        )
        self.validator = CanonicalSampleValidator(max_dist=config.project.max_dist)
        self.download_state = FileDownloadStateStore(self.paths.state / "downloads.jsonl")
        self.extraction_state = FileExtractionStateStore(self.paths.state / "extractions.jsonl")
        self.processing_state = FileProcessingStateStore(self.paths.state / "processed.jsonl")
        self.error_store = JsonlErrorStore(self.paths.state / "errors.jsonl")
        self._metric_records: list[dict[str, float]] = []
        self._selected_source_items_cache: list[object] | None = None
        self._run_started_at = datetime.now(timezone.utc)
        self.verbose = False
        self._run_stats: dict[str, int] = {
            "selected_download_unit_count": 0,
            "download_error_count": 0,
            "selected_extraction_unit_count": 0,
            "extraction_error_count": 0,
            "available_source_item_count": 0,
            "selected_source_item_count": 0,
            "skipped_by_process_ratio_count": 0,
            "valid_sample_count": 0,
            "invalid_sample_count": 0,
            "processing_error_count": 0,
        }

    def reset_source_selection_cache(self) -> None:
        self._selected_source_items_cache = None

    def is_metric_scale(self) -> bool:
        return True

    def scale_group_name(self) -> str:
        return "metric" if self.is_metric_scale() else "relative"

    def _build_paths(self) -> DatasetPaths:
        project_root = Path(self.config.output.root_data_dir) / self.config.project.name
        root = project_root / self.scale_group_name() / self.dataset_name
        processed = root / self.config.output.processed_subdir_name
        return DatasetPaths(
            root=root,
            raw=root / self.config.output.raw_subdir_name,
            hf_cache=root / ".hf_cache",
            processed=processed,
            processed_files=processed / "files",
            state=root / self.config.output.state_subdir_name,
            metadata=processed / self.config.output.metadata_filename,
            run_report=processed / "run_report.json",
        )

    def run(self) -> None:
        self.prepare_directories()
        self.run_download_stage()
        self.run_extraction_stage()
        self.write_samples(self.iter_valid_samples())
        self.build_metrics_summary()
        self.build_metadata()
        self.build_run_report()
        self.validate_output()

    def prepare_directories(self) -> None:
        for path in (
            self.paths.raw,
            self.paths.processed,
            self.paths.processed_files,
            self.paths.state,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def use_hf_cache(self) -> Iterator[None]:
        self.paths.hf_cache.mkdir(parents=True, exist_ok=True)
        hub_cache = self.paths.hf_cache / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        prior_env = {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        }
        os.environ["HF_HOME"] = str(self.paths.hf_cache)
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
        try:
            yield
        finally:
            for key, value in prior_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def clear_hf_cache(self) -> bool:
        if not self.paths.hf_cache.exists():
            return False
        shutil.rmtree(self.paths.hf_cache)
        return True

    def hf_list_repo_files(self, repo_id: str, *, repo_type: str = "dataset") -> list[str]:
        from huggingface_hub import list_repo_files

        with self.use_hf_cache():
            return list(list_repo_files(repo_id=repo_id, repo_type=repo_type))

    def hf_hub_download(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str = "dataset",
        revision: str | None = None,
        local_dir: str | Path | None = None,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        with self.use_hf_cache():
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    revision=revision,
                    local_dir=None if local_dir is None else str(local_dir),
                )
            )

    def hf_snapshot_download(
        self,
        *,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str | None = None,
        local_dir: str | Path | None = None,
        allow_patterns: list[str] | None = None,
        tqdm_class: type[object] | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        kwargs: dict[str, object] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
        }
        if local_dir is not None:
            kwargs["local_dir"] = str(local_dir)
        if allow_patterns is not None:
            kwargs["allow_patterns"] = allow_patterns
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class

        with self.use_hf_cache():
            return Path(snapshot_download(**kwargs))

    def run_download_stage(self) -> None:
        selected_units = self._iter_selected_download_units()
        self._run_stats["selected_download_unit_count"] = len(selected_units)
        for unit in selected_units:
            unit_id = self.get_download_unit_id(unit)
            if self.is_download_unit_satisfied(unit):
                if not self.download_state.is_complete(unit_id):
                    self.download_state.mark_complete(unit_id)
                continue
            try:
                self.download_unit(unit)
                self.download_state.mark_complete(unit_id)
            except Exception as exc:
                self.handle_stage_exception(stage="download", item_id=unit_id, exc=exc)

    def run_extraction_stage(self) -> None:
        extraction_units = list(self.enumerate_extraction_units())
        self._run_stats["selected_extraction_unit_count"] = len(extraction_units)
        for unit in extraction_units:
            unit_id = self.get_extraction_unit_id(unit)
            if self.is_extraction_unit_satisfied(unit):
                if not self.extraction_state.is_complete(unit_id):
                    self.extraction_state.mark_complete(unit_id)
                continue
            try:
                self.extract_unit(unit)
                self.extraction_state.mark_complete(unit_id)
            except Exception as exc:
                self.handle_stage_exception(stage="extraction", item_id=unit_id, exc=exc)

    def run_extraction_cleanup_stage(self) -> None:
        extraction_units = list(self.enumerate_extraction_units())
        self._run_stats["selected_extraction_unit_count"] = len(extraction_units)
        for unit in extraction_units:
            unit_id = self.get_extraction_unit_id(unit)
            if self.is_extraction_unit_satisfied(unit):
                if not self.extraction_state.is_complete(unit_id):
                    self.extraction_state.mark_complete(unit_id)
                continue
            try:
                self.extract_unit(unit)
                self.remove_download_artifact(unit)
                self.extraction_state.mark_complete(unit_id)
            except Exception as exc:
                self.handle_stage_exception(stage="extraction", item_id=unit_id, exc=exc)

    def iter_valid_samples(self) -> Iterator[SampleRecord]:
        progress = self._create_processing_progress(self.get_selected_source_items())
        try:
            for item in progress:
                item_id = self.get_source_item_id(item)
                if self.processing_state.is_complete(item_id):
                    self._update_processing_progress(progress)
                    continue
                try:
                    loaded_item = self.load_source_item(item)
                    camera_model = self.build_camera_model(item, loaded_item)
                    sample = self.build_sample(item, loaded_item, camera_model)
                    report = self.validator.validate(sample)
                    if not report.valid:
                        self.handle_invalid_sample(item_id, report)
                        self._update_processing_progress(progress)
                        continue
                    self._metric_records.append(report.metrics)
                    self.processing_state.mark_complete(item_id)
                    self._run_stats["valid_sample_count"] += 1
                    self._update_processing_progress(progress)
                    yield sample
                except Exception as exc:
                    self.handle_processing_exception(item_id, exc)
                    self._update_processing_progress(progress)
        finally:
            self._close_processing_progress(progress)

    def handle_invalid_sample(self, item_id: str, report: ValidationReport) -> None:
        messages = "; ".join(issue.message for issue in report.issues)
        self._run_stats["invalid_sample_count"] += 1
        self.record_error(stage="processing", item_id=item_id, error_message=messages)

    def handle_processing_exception(self, item_id: str, exc: Exception) -> None:
        self._run_stats["processing_error_count"] += 1
        self.record_error(stage="processing", item_id=item_id, error_message=str(exc), exc=exc)

    def handle_stage_exception(self, stage: str, item_id: str, exc: Exception) -> None:
        stat_key = f"{stage}_error_count"
        if stat_key in self._run_stats:
            self._run_stats[stat_key] += 1
        self.record_error(stage=stage, item_id=item_id, error_message=str(exc), exc=exc)

    def get_download_unit_id(self, unit: object) -> str:
        return str(unit)

    def get_extraction_unit_id(self, unit: object) -> str:
        return str(unit)

    def get_source_item_id(self, item: object) -> str:
        return str(item)

    def _iter_selected_download_units(self) -> list[object]:
        return list(self.enumerate_download_units())

    def get_selected_download_units(self) -> list[object]:
        return self._iter_selected_download_units()

    def is_partial_download_build(self) -> bool:
        return len(self.get_selected_download_units()) < len(list(self.enumerate_download_units()))

    def _should_process_item(self, item_id: str) -> bool:
        fraction = self.config.runtime.process_ratio
        if fraction >= 1.0:
            return True
        seed = self.config.runtime.shuffle_seed
        digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
        return bucket < fraction

    def _iter_selected_source_items(self) -> list[object]:
        items = list(self.enumerate_source_items())
        self._run_stats["available_source_item_count"] = len(items)
        if not items:
            self._run_stats["selected_source_item_count"] = 0
            self._run_stats["skipped_by_process_ratio_count"] = 0
            return []
        fraction = self.config.runtime.process_ratio
        if fraction >= 1.0:
            self._run_stats["selected_source_item_count"] = len(items)
            self._run_stats["skipped_by_process_ratio_count"] = 0
            return items

        selected_count = max(1, math.ceil(len(items) * fraction))
        if items and sys.stdout.isatty():
            print(
                f"[{self.dataset_name}] selecting {selected_count} of {len(items)} source items "
                f"for process_ratio={fraction}"
            )
        scored_items: list[tuple[int, int, object]] = []
        selection_progress = self._create_selection_progress(items)
        try:
            for index, item in enumerate(selection_progress):
                scored_items.append((self._item_selection_score(self.get_source_item_id(item)), index, item))
        finally:
            self._close_selection_progress(selection_progress)
        scored_items.sort(key=lambda row: (row[0], row[1]))
        selected_ids = {self.get_source_item_id(item) for _, _, item in scored_items[:selected_count]}
        selected_items = [item for item in items if self.get_source_item_id(item) in selected_ids]
        self._run_stats["selected_source_item_count"] = len(selected_items)
        self._run_stats["skipped_by_process_ratio_count"] = len(items) - len(selected_items)
        return selected_items

    def get_selected_source_items(self) -> list[object]:
        if self._selected_source_items_cache is None:
            self._selected_source_items_cache = self._iter_selected_source_items()
        return self._selected_source_items_cache

    def _create_processing_progress(self, items: list[object]) -> Iterable[object]:
        if not items or not sys.stdout.isatty():
            return items
        progress = tqdm(
            items,
            desc=f"{self.dataset_name} process",
            unit="sample",
            leave=False,
        )
        self._update_processing_progress(progress)
        return progress

    def _create_selection_progress(self, items: list[object]) -> Iterable[object]:
        if not items or not self.verbose or not sys.stdout.isatty():
            return items
        return self._iter_logged_progress(
            items,
            label="selection",
            unit="sample",
        )

    def _update_processing_progress(self, progress: Iterable[object]) -> None:
        if not isinstance(progress, tqdm):
            return
        progress.set_postfix(
            valid=self._run_stats["valid_sample_count"],
            invalid=self._run_stats["invalid_sample_count"],
            errors=self._run_stats["processing_error_count"],
            refresh=False,
        )

    def _close_processing_progress(self, progress: Iterable[object]) -> None:
        if isinstance(progress, tqdm):
            progress.close()

    def _close_selection_progress(self, progress: Iterable[object]) -> None:
        if isinstance(progress, tqdm):
            progress.close()

    def _iter_logged_progress(
        self,
        items: list[object],
        label: str,
        unit: str,
        every_items: int = 5000,
        every_seconds: float = 2.0,
    ) -> Iterable[object]:
        if not self.verbose:
            return items
        return self._iter_verbose_logged_progress(
            items,
            label=label,
            unit=unit,
            every_items=every_items,
            every_seconds=every_seconds,
        )

    def _iter_verbose_logged_progress(
        self,
        items: list[object],
        label: str,
        unit: str,
        every_items: int = 5000,
        every_seconds: float = 2.0,
    ) -> Iterable[object]:
        total = len(items)
        last_print = time.monotonic()
        for index, item in enumerate(items, start=1):
            now = time.monotonic()
            if index == 1 or index == total or index % every_items == 0 or (now - last_print) >= every_seconds:
                print(f"[{self.dataset_name}] {label}: {index}/{total} {unit}s")
                last_print = now
            yield item

    def _item_selection_score(self, item_id: str) -> int:
        seed = self.config.runtime.shuffle_seed
        digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big")

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        return ()

    def remove_download_artifact(self, unit: object) -> None:
        del unit

    def get_download_artifact_path(self, unit: object) -> Path | None:
        del unit
        return None

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        del unit
        return None

    def is_download_unit_satisfied(self, unit: object) -> bool:
        return self.download_state.is_complete(self.get_download_unit_id(unit))

    def is_extraction_unit_satisfied(self, unit: object) -> bool:
        return self.extraction_state.is_complete(self.get_extraction_unit_id(unit))

    def record_error(self, stage: str, item_id: str, error_message: str, exc: Exception | None = None) -> None:
        traceback_text = None
        if exc is not None and self.config.runtime.write_error_traces:
            traceback_text = "".join(traceback.format_exception(exc))
        error = ErrorRecord(
            stage=stage,
            dataset_name=self.dataset_name,
            item_id=item_id,
            error_message=error_message,
            traceback_text=traceback_text,
        )
        if self.config.runtime.skip_known_errors and self.error_store.has_matching_record(error):
            return
        self.error_store.record(error)

    def build_metrics_summary(self) -> None:
        summary = summarize_metrics(self._metric_records)
        path = self.paths.processed / "metrics_summary.json"
        self._write_json_atomic(
            path,
            {
                "sample_count": summary.sample_count,
                "metric_means": summary.metric_means,
                "metric_mins": summary.metric_mins,
                "metric_maxs": summary.metric_maxs,
            },
        )

    def build_run_report(self) -> None:
        metadata_payload = {}
        if self.paths.metadata.exists():
            metadata_payload = json.loads(self.paths.metadata.read_text())

        recent_errors: list[dict[str, object]] = []
        error_stage_counts: dict[str, int] = {}
        errors_path = self.paths.state / "errors.jsonl"
        if errors_path.exists():
            for line in errors_path.read_text().splitlines():
                if not line:
                    continue
                payload = json.loads(line)
                stage = str(payload["stage"])
                error_stage_counts[stage] = error_stage_counts.get(stage, 0) + 1
                recent_errors.append(
                    {
                        "stage": stage,
                        "item_id": payload["item_id"],
                        "error_message": payload["error_message"],
                    }
                )

        report = {
            "dataset": self.dataset_name,
            "run_started_at": self._run_started_at.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata_file": self.paths.metadata.name,
            "metrics_summary_file": "metrics_summary.json",
            "error_log_file": errors_path.name,
            "run_stats": self._run_stats,
            "shard_count": metadata_payload.get("shard_count", 0),
            "valid_sample_count": metadata_payload.get("valid_sample_count", self._run_stats["valid_sample_count"]),
            "error_stage_counts": error_stage_counts,
            "recent_errors": recent_errors[-5:],
        }
        self._write_json_atomic(self.paths.run_report, report)

    def _write_json_atomic(self, path: Path, payload: dict[str, object]) -> None:
        partial_path = path.with_name(f"{path.name}.partial")
        partial_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        partial_path.replace(path)

    @abstractmethod
    def enumerate_download_units(self) -> Iterable[object]:
        raise NotImplementedError

    @abstractmethod
    def download_unit(self, unit: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def enumerate_extraction_units(self) -> Iterable[object]:
        raise NotImplementedError

    @abstractmethod
    def extract_unit(self, unit: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def enumerate_source_items(self) -> Iterable[object]:
        raise NotImplementedError

    @abstractmethod
    def load_source_item(self, item: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def build_camera_model(self, item: object, loaded_item: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def build_sample(self, item: object, loaded_item: object, camera_model: object) -> SampleRecord:
        raise NotImplementedError

    @abstractmethod
    def write_samples(self, sample_iterator: Iterable[SampleRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_metadata(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_output(self) -> None:
        raise NotImplementedError
