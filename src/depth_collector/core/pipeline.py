from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Iterable, Iterator

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

    def _build_paths(self) -> DatasetPaths:
        root = Path(self.config.output.root_data_dir) / self.dataset_name
        processed = root / self.config.output.processed_subdir_name
        return DatasetPaths(
            root=root,
            raw=root / self.config.output.raw_subdir_name,
            processed=processed,
            processed_files=processed / "files",
            state=root / self.config.output.state_subdir_name,
            metadata=processed / self.config.output.metadata_filename,
        )

    def run(self) -> None:
        self.prepare_directories()
        self.run_download_stage()
        self.run_extraction_stage()
        self.write_samples(self.iter_valid_samples())
        self.build_metrics_summary()
        self.build_metadata()
        self.validate_output()

    def prepare_directories(self) -> None:
        for path in (
            self.paths.raw,
            self.paths.processed,
            self.paths.processed_files,
            self.paths.state,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def run_download_stage(self) -> None:
        for unit in self.enumerate_download_units():
            unit_id = self.get_download_unit_id(unit)
            if self.download_state.is_complete(unit_id):
                continue
            self.download_unit(unit)
            self.download_state.mark_complete(unit_id)

    def run_extraction_stage(self) -> None:
        for unit in self.enumerate_extraction_units():
            unit_id = self.get_extraction_unit_id(unit)
            if self.extraction_state.is_complete(unit_id):
                continue
            self.extract_unit(unit)
            self.extraction_state.mark_complete(unit_id)

    def iter_valid_samples(self) -> Iterator[SampleRecord]:
        for item in self.enumerate_source_items():
            item_id = self.get_source_item_id(item)
            if self.processing_state.is_complete(item_id):
                continue
            loaded_item = self.load_source_item(item)
            camera_model = self.build_camera_model(item, loaded_item)
            sample = self.build_sample(item, loaded_item, camera_model)
            report = self.validator.validate(sample)
            if not report.valid:
                self.handle_invalid_sample(item_id, report)
                continue
            self._metric_records.append(report.metrics)
            self.processing_state.mark_complete(item_id)
            yield sample

    def handle_invalid_sample(self, item_id: str, report: ValidationReport) -> None:
        messages = "; ".join(issue.message for issue in report.issues)
        self.error_store.record(
            ErrorRecord(
                stage="processing",
                dataset_name=self.dataset_name,
                item_id=item_id,
                error_message=messages,
            )
        )

    def get_download_unit_id(self, unit: object) -> str:
        return str(unit)

    def get_extraction_unit_id(self, unit: object) -> str:
        return str(unit)

    def get_source_item_id(self, item: object) -> str:
        return str(item)

    def build_metrics_summary(self) -> None:
        summary = summarize_metrics(self._metric_records)
        path = self.paths.processed / "metrics_summary.json"
        path.write_text(
            json.dumps(
                {
                    "sample_count": summary.sample_count,
                    "metric_means": summary.metric_means,
                    "metric_mins": summary.metric_mins,
                    "metric_maxs": summary.metric_maxs,
                },
                indent=2,
                sort_keys=True,
            )
        )

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
