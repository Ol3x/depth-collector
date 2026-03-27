from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import PinholeCameraModel


@dataclass(frozen=True)
class TartanAirArchiveUnit:
    environment: str
    difficulty: str
    modality: str

    @property
    def filename(self) -> str:
        return f"{self.modality}.zip"


@dataclass(frozen=True)
class TartanAirSourceItem:
    environment: str
    difficulty: str
    modality: str
    relative_path: str


class TartanAirPipeline(DatasetPipeline):
    """Narrow first slice of a TartanAir pipeline."""

    DEFAULT_DIFFICULTIES = ("Easy", "Hard")
    DEFAULT_MODALITIES = ("image_left",)
    DEFAULT_ENVIRONMENTS = ()

    def _get_option_list(self, key: str, default: tuple[str, ...]) -> list[str]:
        value = self.dataset_config.options.get(key, default)
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _selected_environments(self) -> list[str]:
        configured = self._get_option_list("environments", self.DEFAULT_ENVIRONMENTS)
        if configured:
            return configured

        if not self.paths.raw.exists():
            return []
        return sorted(path.name for path in self.paths.raw.iterdir() if path.is_dir())

    def _selected_difficulties(self) -> list[str]:
        return self._get_option_list("difficulties", self.DEFAULT_DIFFICULTIES)

    def _selected_modalities(self) -> list[str]:
        return self._get_option_list("modalities", self.DEFAULT_MODALITIES)

    def enumerate_download_units(self) -> Iterable[TartanAirArchiveUnit]:
        for environment in self._selected_environments():
            for difficulty in self._selected_difficulties():
                for modality in self._selected_modalities():
                    yield TartanAirArchiveUnit(
                        environment=environment,
                        difficulty=difficulty,
                        modality=modality,
                    )

    def download_unit(self, unit: TartanAirArchiveUnit) -> None:
        # Real HF download support will be added later.
        archive_path = self._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

    def enumerate_extraction_units(self) -> Iterable[TartanAirArchiveUnit]:
        return self.enumerate_download_units()

    def extract_unit(self, unit: TartanAirArchiveUnit) -> None:
        # Real archive extraction will be added later.
        extracted_dir = self._extracted_dir(unit)
        extracted_dir.mkdir(parents=True, exist_ok=True)

    def enumerate_source_items(self) -> Iterable[TartanAirSourceItem]:
        for environment in self._selected_environments():
            for difficulty in self._selected_difficulties():
                for modality in self._selected_modalities():
                    extracted_dir = self._extracted_dir(
                        TartanAirArchiveUnit(
                            environment=environment,
                            difficulty=difficulty,
                            modality=modality,
                        )
                    )
                    if not extracted_dir.exists():
                        continue
                    for path in sorted(extracted_dir.rglob("*")):
                        if path.is_file():
                            yield TartanAirSourceItem(
                                environment=environment,
                                difficulty=difficulty,
                                modality=modality,
                                relative_path=str(path.relative_to(extracted_dir)),
                            )

    def load_source_item(self, item: TartanAirSourceItem) -> dict[str, object]:
        return {
            "path": self._extracted_dir(
                TartanAirArchiveUnit(
                    environment=item.environment,
                    difficulty=item.difficulty,
                    modality=item.modality,
                )
            )
            / item.relative_path
        }

    def build_camera_model(self, item: TartanAirSourceItem, loaded_item: object) -> PinholeCameraModel:
        del item, loaded_item
        return PinholeCameraModel(
            width=640,
            height=640,
            fx=320.0,
            fy=320.0,
            cx=320.0,
            cy=320.0,
        )

    def build_sample(self, item: TartanAirSourceItem, loaded_item: object, camera_model: PinholeCameraModel) -> SampleRecord:
        del loaded_item, camera_model
        image = np.zeros((640, 640, 3), dtype=np.float32)
        distance = np.ones((640, 640, 1), dtype=np.float32)
        ray_dir = np.zeros((640, 640, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance={
                "environment": item.environment,
                "difficulty": item.difficulty,
                "modality": item.modality,
                "relative_path": item.relative_path,
            },
        )

    def write_samples(self, sample_iterator: Iterable[SampleRecord]) -> None:
        count = sum(1 for _ in sample_iterator)
        summary_path = self.paths.processed / "sample_counts.json"
        summary_path.write_text(json.dumps({"sample_count": count}, indent=2, sort_keys=True))

    def build_metadata(self) -> None:
        metadata = {
            "dataset": self.dataset_name,
            "hf_dataset_id": self.dataset_config.hf_dataset_id,
            "sample_count_file": "sample_counts.json",
            "train_val_split": self.config.project.train_val_split,
        }
        self.paths.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def validate_output(self) -> None:
        if not self.paths.metadata.exists():
            raise ValueError("metadata.json was not created")

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, TartanAirArchiveUnit)
        return f"{unit.environment}/{unit.difficulty}/{unit.modality}"

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, TartanAirSourceItem)
        return f"{item.environment}/{item.difficulty}/{item.modality}/{item.relative_path}"

    def _archive_path(self, unit: TartanAirArchiveUnit) -> Path:
        return self.paths.raw / unit.environment / unit.difficulty / unit.filename

    def _extracted_dir(self, unit: TartanAirArchiveUnit) -> Path:
        return self.paths.raw / unit.environment / unit.difficulty / unit.modality
