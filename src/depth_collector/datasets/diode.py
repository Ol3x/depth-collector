from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import tarfile
from typing import Iterable

import numpy as np
from PIL import Image

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import PinholeCameraModel, clip_distance_to_max_dist, generate_pinhole_rays, z_depth_to_distance
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class DIODEArchiveUnit:
    archive_name: str


@dataclass(frozen=True)
class DIODESourceItem:
    split_name: str
    scene_type: str
    relative_stem: str
    image_relative_path: str
    depth_relative_path: str
    depth_mask_relative_path: str


class DIODEPipeline(DatasetPipeline):
    """Archive-based DIODE pipeline with RGB, metric depth, and validity masks."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    DEFAULT_CAMERA_INTRINSICS = {
        "width": 1024.0,
        "height": 768.0,
        "fx": 512.0,
        "fy": 512.0,
        "cx": 512.0,
        "cy": 384.0,
    }

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []

    def _archive_filename(self) -> str:
        return str(self.dataset_config.options.get("archive_filename", "train_subset.tar.gz"))

    def _selected_scene_types(self) -> list[str]:
        configured = self.dataset_config.options.get("scene_types", "*")
        if isinstance(configured, str):
            if configured.strip().lower() in self.ALL_SELECTOR_VALUES:
                return []
            return [configured]
        return [
            str(scene_type)
            for scene_type in configured
            if str(scene_type).strip().lower() not in self.ALL_SELECTOR_VALUES
        ]

    def _selected_split_names(self) -> list[str]:
        configured = self.dataset_config.options.get("splits", ["train"])
        if isinstance(configured, str):
            return [configured]
        return [str(split_name) for split_name in configured]

    def _camera_intrinsics(self) -> dict[str, float]:
        configured = self.dataset_config.options.get("camera_intrinsics", {})
        assert isinstance(configured, dict)
        values = dict(self.DEFAULT_CAMERA_INTRINSICS)
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if key in configured:
                values[key] = float(configured[key])
        return values

    def _archive_path(self) -> Path:
        return self.paths.raw / self._archive_filename()

    def _extracted_root(self) -> Path:
        configured = self.dataset_config.options.get("extracted_root")
        if configured:
            return self.paths.raw / str(configured)
        return self.paths.raw / Path(self._archive_filename()).stem.removesuffix(".tar")

    def enumerate_download_units(self) -> Iterable[DIODEArchiveUnit]:
        yield DIODEArchiveUnit(archive_name=self._archive_filename())

    def download_unit(self, unit: DIODEArchiveUnit) -> None:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        archive_path = self._archive_path()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if local_archive_root:
            source_path = Path(str(local_archive_root)) / unit.archive_name
            if not source_path.exists():
                raise FileNotFoundError(f"missing local DIODE archive source: {source_path}")
            shutil.copy2(source_path, archive_path)
            return

        downloaded_path = self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            filename=unit.archive_name,
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
        )
        downloaded_path = Path(downloaded_path)
        if downloaded_path.resolve() != archive_path.resolve():
            shutil.copy2(downloaded_path, archive_path)

    def enumerate_extraction_units(self) -> Iterable[DIODEArchiveUnit]:
        return self.enumerate_download_units()

    def extract_unit(self, unit: DIODEArchiveUnit) -> None:
        archive_path = self._archive_path()
        if not archive_path.exists():
            raise FileNotFoundError(f"missing DIODE archive for extraction: {archive_path}")
        self.paths.raw.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(self.paths.raw, filter="data")

    def enumerate_source_items(self) -> Iterable[DIODESourceItem]:
        split_names = set(self._selected_split_names())
        selected_scene_types = set(self._selected_scene_types())
        extracted_root = self._extracted_root()
        if not extracted_root.exists():
            self.record_error("enumeration", self.dataset_name, f"missing extracted DIODE root: {extracted_root}")
            return
        default_split_name = next(iter(split_names)) if split_names else "train"
        for image_path in sorted(extracted_root.rglob("*.png")):
            if image_path.name.endswith(("_depth.png", "_depth_mask.png", "_normal.png")):
                continue
            relative_path = image_path.relative_to(extracted_root)
            parts = relative_path.parts
            if len(parts) < 3:
                continue
            if parts[0] in split_names and len(parts) >= 4:
                split_name = parts[0]
                scene_type = parts[1]
            else:
                split_name = default_split_name
                scene_type = parts[0]
            if selected_scene_types and scene_type not in selected_scene_types:
                continue
            relative_stem = str(relative_path.with_suffix(""))
            depth_relative_path = f"{relative_stem}_depth.npy"
            depth_mask_relative_path = f"{relative_stem}_depth_mask.npy"
            depth_path = extracted_root / depth_relative_path
            depth_mask_path = extracted_root / depth_mask_relative_path
            if not depth_path.exists():
                self.record_error(
                    "enumeration",
                    relative_stem,
                    f"missing paired DIODE depth file: {depth_relative_path}",
                )
                continue
            if not depth_mask_path.exists():
                self.record_error(
                    "enumeration",
                    relative_stem,
                    f"missing paired DIODE depth mask file: {depth_mask_relative_path}",
                )
                continue
            yield DIODESourceItem(
                split_name=split_name,
                scene_type=scene_type,
                relative_stem=relative_stem,
                image_relative_path=str(relative_path),
                depth_relative_path=depth_relative_path,
                depth_mask_relative_path=depth_mask_relative_path,
            )

    def load_source_item(self, item: DIODESourceItem) -> dict[str, object]:
        extracted_root = self._extracted_root()
        image = np.asarray(Image.open(extracted_root / item.image_relative_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.asarray(np.load(extracted_root / item.depth_relative_path), dtype=np.float32)
        depth_mask = np.asarray(np.load(extracted_root / item.depth_mask_relative_path))
        return {
            "image": image,
            "depth": depth,
            "depth_mask": depth_mask,
        }

    def build_camera_model(self, item: DIODESourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        height, width = image.shape[:2]
        intrinsics = self._camera_intrinsics()
        scale_x = width / intrinsics["width"]
        scale_y = height / intrinsics["height"]
        return PinholeCameraModel(
            width=width,
            height=height,
            fx=intrinsics["fx"] * scale_x,
            fy=intrinsics["fy"] * scale_y,
            cx=intrinsics["cx"] * scale_x,
            cy=intrinsics["cy"] * scale_y,
        )

    def build_sample(self, item: DIODESourceItem, loaded_item: object, camera_model: PinholeCameraModel) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth = np.asarray(loaded_item["depth"], dtype=np.float32)
        depth_mask = np.asarray(loaded_item["depth_mask"])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError("DIODE depth must decode to a 2D array")
        if image.shape[:2] != depth.shape:
            raise ValueError("DIODE image and depth shapes must match")
        if depth_mask.ndim == 3 and depth_mask.shape[-1] == 1:
            depth_mask = depth_mask[..., 0]
        if depth_mask.ndim != 2:
            raise ValueError("DIODE depth mask must decode to a 2D array")
        if depth_mask.shape != depth.shape:
            raise ValueError("DIODE depth mask and depth shapes must match")

        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        depth_semantics = str(self.dataset_config.options.get("depth_semantics", "distance")).strip().lower()
        if depth_semantics == "distance":
            distance = depth
        elif depth_semantics == "z_depth":
            distance = z_depth_to_distance(depth[..., None], ray_dir)[..., 0]
        else:
            raise ValueError("DIODE depth_semantics must be 'distance' or 'z_depth'")

        valid_mask = depth_mask.astype(bool)
        valid_mask &= np.isfinite(distance)
        valid_mask &= distance > 0.0
        distance = distance[..., None]
        distance[..., 0][~valid_mask] = self.config.project.max_dist
        distance = clip_distance_to_max_dist(distance, self.config.project.max_dist)

        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance.astype(np.float32),
            ray_dir=ray_dir,
            provenance={
                "split_name": item.split_name,
                "scene_type": item.scene_type,
                "relative_stem": item.relative_stem,
                "image_relative_path": item.image_relative_path,
                "depth_relative_path": item.depth_relative_path,
                "depth_mask_relative_path": item.depth_mask_relative_path,
                "projection": "pinhole",
            },
        )

    def write_samples(self, sample_iterator: Iterable[SampleRecord]) -> None:
        shard_writer = ShardWriter(
            output_dir=self.paths.processed_files,
            target_shard_size_bytes=max(1, int(self.config.runtime.target_shard_size_gb * (1024**3))),
            ensure_split_pair=self.config.runtime.process_ratio < 1.0,
        )
        self._written_shards = shard_writer.write(sample_iterator)

    def build_metadata(self) -> None:
        shard_names = [str(shard["shard_name"]) for shard in self._written_shards]
        sample_counts = {str(shard["shard_name"]): int(shard["sample_count"]) for shard in self._written_shards}
        train_shards, val_shards = self._suggest_shard_splits(shard_names)
        metadata = {
            "dataset": self.dataset_name,
            "hf_dataset_id": self.dataset_config.hf_dataset_id,
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shard_count": len(self._written_shards),
            "shard_names": shard_names,
            "samples_per_shard": sample_counts,
            "train_val_split": self.config.project.train_val_split,
            "suggested_train_shards": train_shards,
            "suggested_val_shards": val_shards,
            "process_ratio": self.config.runtime.process_ratio,
            "partial_build": self.is_partial_download_build() or self.config.runtime.process_ratio < 1.0,
            "selected_download_unit_count": self._run_stats["selected_download_unit_count"],
            "download_error_count": self._run_stats["download_error_count"],
            "selected_extraction_unit_count": self._run_stats["selected_extraction_unit_count"],
            "extraction_error_count": self._run_stats["extraction_error_count"],
            "available_source_item_count": self._run_stats["available_source_item_count"],
            "selected_source_item_count": self._run_stats["selected_source_item_count"],
            "skipped_by_process_ratio_count": self._run_stats["skipped_by_process_ratio_count"],
            "valid_sample_count": self._run_stats["valid_sample_count"],
            "invalid_sample_count": self._run_stats["invalid_sample_count"],
            "processing_error_count": self._run_stats["processing_error_count"],
        }
        self._write_json_atomic(self.paths.metadata, metadata)

    def validate_output(self) -> None:
        self.validate_processed_output_structure()

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, DIODEArchiveUnit)
        return unit.archive_name

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, DIODESourceItem)
        return item.relative_stem

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        yield self._archive_path()

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, DIODEArchiveUnit)
        return self._archive_path()

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, DIODEArchiveUnit)
        del unit
        return self._extracted_root()

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, DIODEArchiveUnit)
        archive_path = self._archive_path()
        if archive_path.exists():
            archive_path.unlink()

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, DIODEArchiveUnit)
        archive_path = self._archive_path()
        return archive_path.exists() and archive_path.is_file()

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(math.floor(len(shard_names) * self.config.project.train_val_split))
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "DIODEArchiveUnit",
    "DIODEPipeline",
    "DIODESourceItem",
]
