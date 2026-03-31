from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
from PIL import Image

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import EquirectangularCameraModel, clip_distance_to_max_dist, generate_equirectangular_rays
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class ToF360SceneUnit:
    scene_name: str


@dataclass(frozen=True)
class ToF360SourceItem:
    scene_name: str
    frame_id: str
    image_relative_path: str
    depth_relative_path: str


class ToF360Pipeline(DatasetPipeline):
    """Scene-folder ToF-360 pipeline for equirectangular RGB-D captures."""

    ALL_SELECTOR_VALUES = {"*", "all"}

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(repo_id=self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _scene_option(self) -> list[str]:
        configured = self.dataset_config.options.get("scenes", "*")
        if isinstance(configured, str):
            if configured.strip().lower() in self.ALL_SELECTOR_VALUES:
                return []
            return [configured]
        return [
            str(scene)
            for scene in configured
            if str(scene).strip().lower() not in self.ALL_SELECTOR_VALUES
        ]

    def _discover_scene_names_from_root(self, root: Path) -> list[str]:
        if not root.exists():
            return []
        scene_names: list[str] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if self._resolve_rgb_root(path) is None:
                continue
            if not (path / self._depth_dir_name()).exists():
                continue
            scene_names.append(path.name)
        return scene_names

    def _discover_scene_names_from_remote(self) -> list[str]:
        scene_names: set[str] = set()
        for repo_path in self._remote_repo_files():
            parts = Path(repo_path).parts
            if len(parts) >= 2:
                scene_names.add(parts[0])
        return sorted(scene_names)

    def _selected_scene_names(self) -> list[str]:
        configured = self._scene_option()
        if configured:
            scenes = configured
        else:
            local_archive_root = self.dataset_config.options.get("local_archive_root")
            if local_archive_root:
                scenes = self._discover_scene_names_from_root(Path(str(local_archive_root)))
            else:
                scenes = []
            if not scenes:
                scenes = self._discover_scene_names_from_root(self.paths.raw)
            if not scenes:
                scenes = self._discover_scene_names_from_remote()
        return [str(scene_name) for scene_name in self.apply_dataset_selection(scenes)]

    def _rgb_dir_name(self) -> str:
        return str(self.dataset_config.options.get("rgb_dir", "rgb"))

    def _depth_dir_name(self) -> str:
        return str(self.dataset_config.options.get("depth_dir", "depth"))

    def _candidate_rgb_dir_names(self) -> tuple[str, ...]:
        configured = self.dataset_config.options.get("rgb_dir_candidates")
        if isinstance(configured, str):
            return (configured,)
        if isinstance(configured, list):
            names = tuple(str(name) for name in configured if str(name))
            if names:
                return names
        return ("rgb", "manhattan", "manhattan_rgb", "rgb_manhattan", "color", "albedo")

    def _resolve_rgb_root(self, scene_root: Path) -> Path | None:
        configured_root = scene_root / self._rgb_dir_name()
        if configured_root.exists():
            return configured_root
        for name in self._candidate_rgb_dir_names():
            candidate = scene_root / name
            if candidate.exists():
                return candidate
        return None

    def enumerate_download_units(self) -> Iterable[ToF360SceneUnit]:
        for scene_name in self._selected_scene_names():
            yield ToF360SceneUnit(scene_name=scene_name)

    def download_unit(self, unit: ToF360SceneUnit) -> None:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_root = Path(str(local_archive_root)) / unit.scene_name
            if not source_root.exists():
                raise FileNotFoundError(f"missing local ToF-360 scene source: {source_root}")
            shutil.copytree(source_root, self.paths.raw / unit.scene_name, dirs_exist_ok=True)
            return

        self.hf_snapshot_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
            allow_patterns=[f"{unit.scene_name}/**"],
        )

    def enumerate_extraction_units(self) -> Iterable[object]:
        return ()

    def extract_unit(self, unit: object) -> None:
        del unit
        raise RuntimeError("ToF-360 does not use an extract stage")

    def enumerate_source_items(self) -> Iterable[ToF360SourceItem]:
        for scene_name in self._selected_scene_names():
            scene_root = self.paths.raw / scene_name
            rgb_root = self._resolve_rgb_root(scene_root)
            depth_root = scene_root / self._depth_dir_name()
            if rgb_root is None:
                available_dirs = sorted(path.name for path in scene_root.iterdir() if path.is_dir()) if scene_root.exists() else []
                self.record_error(
                    "enumeration",
                    scene_name,
                    f"missing ToF-360 RGB directory under {scene_root}; available_dirs={available_dirs}",
                )
                continue
            if not depth_root.exists():
                self.record_error("enumeration", scene_name, f"missing ToF-360 depth directory: {depth_root}")
                continue

            depth_by_frame_id = {
                self._frame_id_from_depth_path(path): path
                for path in sorted(depth_root.rglob("*"))
                if path.is_file() and path.suffix.lower() == ".png"
            }
            for image_path in sorted(rgb_root.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                frame_id = self._frame_id_from_image_path(image_path)
                depth_path = depth_by_frame_id.get(frame_id)
                if depth_path is None:
                    self.record_error(
                        "enumeration",
                        f"{scene_name}/{image_path.name}",
                        f"missing paired ToF-360 depth file for image: {image_path.name} (frame_id={frame_id})",
                    )
                    continue
                yield ToF360SourceItem(
                    scene_name=scene_name,
                    frame_id=frame_id,
                    image_relative_path=str(image_path.relative_to(self.paths.raw)),
                    depth_relative_path=str(depth_path.relative_to(self.paths.raw)),
                )

    def _frame_id_from_depth_path(self, path: Path) -> str:
        return self._normalize_frame_stem(path.stem, suffixes=("_depth",))

    def _frame_id_from_image_path(self, path: Path) -> str:
        return self._normalize_frame_stem(
            path.stem,
            suffixes=("_rgb", "_manhattan", "_albedo", "_color", "_image"),
        )

    @staticmethod
    def _normalize_frame_stem(stem: str, *, suffixes: tuple[str, ...]) -> str:
        normalized = stem
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[: -len(suffix)]
                    changed = True
        return normalized

    def load_source_item(self, item: ToF360SourceItem) -> dict[str, object]:
        image = np.asarray(Image.open(self.paths.raw / item.image_relative_path).convert("RGB"), dtype=np.float32) / 255.0
        depth_png = np.asarray(Image.open(self.paths.raw / item.depth_relative_path), dtype=np.uint16)
        return {
            "image": image,
            "depth_png": depth_png,
        }

    def build_camera_model(self, item: ToF360SourceItem, loaded_item: object) -> EquirectangularCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        height, width = image.shape[:2]
        return EquirectangularCameraModel(width=width, height=height)

    def build_sample(self, item: ToF360SourceItem, loaded_item: object, camera_model: EquirectangularCameraModel) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth_png = np.asarray(loaded_item["depth_png"], dtype=np.uint16)
        if depth_png.ndim == 3:
            depth_png = depth_png[..., 0]
        if depth_png.ndim != 2:
            raise ValueError("ToF-360 depth must decode to a 2D array")
        if image.shape[:2] != depth_png.shape:
            raise ValueError("ToF-360 image and depth shapes must match")

        depth_scale_divisor = float(self.dataset_config.options.get("depth_scale_divisor", 512.0))
        if depth_scale_divisor <= 0.0:
            raise ValueError("ToF-360 depth_scale_divisor must be positive")

        ray_dir = generate_equirectangular_rays(camera_model).astype(np.float32)
        distance = depth_png.astype(np.float32) / depth_scale_divisor
        invalid_mask = depth_png == 0
        missing_depth_policy = str(self.dataset_config.options.get("missing_depth_policy", "max_dist"))
        if missing_depth_policy == "reject" and np.any(invalid_mask):
            raise ValueError("ToF-360 sample contains missing depth values")
        if missing_depth_policy not in {"max_dist", "reject"}:
            raise ValueError("ToF-360 missing_depth_policy must be 'max_dist' or 'reject'")
        distance = distance[..., None]
        if np.any(invalid_mask):
            distance[..., 0][invalid_mask] = self.config.project.max_dist
        distance = clip_distance_to_max_dist(distance, self.config.project.max_dist)

        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance.astype(np.float32),
            ray_dir=ray_dir,
            provenance={
                "scene_name": item.scene_name,
                "frame_id": item.frame_id,
                "image_relative_path": item.image_relative_path,
                "depth_relative_path": item.depth_relative_path,
                "projection": "equirectangular",
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
        assert isinstance(unit, ToF360SceneUnit)
        return unit.scene_name

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, ToF360SourceItem)
        return f"{item.scene_name}/{item.frame_id}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self.enumerate_download_units():
            yield self.paths.raw / unit.scene_name

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, ToF360SceneUnit)
        return self.paths.raw / unit.scene_name

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, ToF360SceneUnit)
        return self.paths.raw / unit.scene_name

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, ToF360SceneUnit)
        scene_root = self.paths.raw / unit.scene_name
        if scene_root.exists():
            shutil.rmtree(scene_root)

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, ToF360SceneUnit)
        scene_root = self.paths.raw / unit.scene_name
        rgb_root = self._resolve_rgb_root(scene_root)
        depth_root = scene_root / self._depth_dir_name()
        return (
            rgb_root is not None
            and depth_root.exists()
            and any(path.is_file() for path in rgb_root.rglob("*"))
            and any(path.is_file() for path in depth_root.rglob("*"))
        )

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(len(shard_names) * self.config.project.train_val_split)
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "ToF360Pipeline",
    "ToF360SceneUnit",
    "ToF360SourceItem",
]
