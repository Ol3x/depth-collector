from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import (
    PinholeCameraModel,
    clip_distance_to_max_dist,
    generate_pinhole_rays,
    z_depth_to_distance,
)
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class MegaDepthDownloadUnit:
    unit_name: str
    unit_type: str = "scene"


@dataclass(frozen=True)
class MegaDepthSourceItem:
    scene_name: str
    image_index: int
    image_relative_path: str
    depth_relative_path: str
    intrinsics: tuple[float, float, float, float]


class _ConcatBinaryReader:
    def __init__(self, part_paths: list[Path]) -> None:
        self._part_paths = part_paths
        self._index = 0
        self._current: BinaryIO | None = None

    def _advance(self) -> bool:
        if self._current is not None:
            self._current.close()
            self._current = None
        if self._index >= len(self._part_paths):
            return False
        self._current = self._part_paths[self._index].open("rb")
        self._index += 1
        return True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            chunks: list[bytes] = []
            while True:
                chunk = self.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            if self._current is None and not self._advance():
                break
            assert self._current is not None
            chunk = self._current.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
                continue
            if not self._advance():
                break
        return b"".join(chunks)

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None


class MegaDepthPipeline(DatasetPipeline):
    """MegaDepth pipeline with HF-backed acquisition and non-metric distance normalization."""

    ALL_SELECTOR_VALUES = {"*", "all"}

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None
        self._remote_mode_cache: str | None = None

    def is_metric_scale(self) -> bool:
        return False

    def _selected_scenes(self) -> list[str]:
        configured = self.dataset_config.options.get("scenes")
        if isinstance(configured, str) and configured.strip().lower() not in self.ALL_SELECTOR_VALUES:
            return [str(scene) for scene in self.apply_dataset_selection([configured])]
        if isinstance(configured, list):
            scenes = [
                str(scene)
                for scene in configured
                if str(scene) and str(scene).strip().lower() not in self.ALL_SELECTOR_VALUES
            ]
            if scenes:
                return [str(scene) for scene in self.apply_dataset_selection(scenes)]
        scene_info_root = self.paths.raw / self._scene_info_dir_name()
        discovered = sorted(path.stem for path in scene_info_root.glob("*.npz"))
        if discovered:
            return [str(scene) for scene in self.apply_dataset_selection(discovered)]
        remote_scene_files = self._remote_scene_names()
        if remote_scene_files:
            return [str(scene) for scene in self.apply_dataset_selection(remote_scene_files)]
        raise ValueError(
            "MegaDepth requires configured `datasets.megadepth.scenes`, remote scene-info `.npz` files, "
            "or existing local scene-info `.npz` files"
        )

    def _uses_local_archive_root(self) -> bool:
        return bool(self.dataset_config.options.get("local_archive_root"))

    def _bundle_unit_name(self) -> str:
        return str(self.dataset_config.options.get("bundle_unit_name", "megadepth_bundle"))

    def _selected_bundles(self) -> list[str]:
        configured = self.dataset_config.options.get("bundles")
        if isinstance(configured, str):
            bundles = [configured]
        elif isinstance(configured, list):
            bundles = [str(bundle) for bundle in configured if str(bundle)]
        else:
            bundles = [self._bundle_unit_name()]
        return [str(bundle) for bundle in self.apply_dataset_selection(bundles)]

    def _scene_info_dir_name(self) -> str:
        return str(self.dataset_config.options.get("scene_info_dir", "prep_scene_info"))

    def _archive_part_prefix(self) -> str:
        return str(self.dataset_config.options.get("archive_part_prefix", "MegaDepth_v1.tar.gz_part"))

    def _downloads_root(self) -> Path:
        return self.paths.raw / "_downloads"

    def _bundle_download_root(self) -> Path:
        return self._downloads_root() / self._bundle_unit_name()

    def _scene_info_path(self, scene_name: str) -> Path:
        return self.paths.raw / self._scene_info_dir_name() / f"{scene_name}.npz"

    def _scene_info_repo_path(self, scene_name: str) -> str:
        return f"{self._scene_info_dir_name()}/{scene_name}.npz"

    def _configured_scene_names(self) -> list[str]:
        configured = self.dataset_config.options.get("scenes")
        if isinstance(configured, str):
            normalized = configured.strip().lower()
            if normalized in self.ALL_SELECTOR_VALUES or not configured.strip():
                return []
            return [configured]
        if isinstance(configured, list):
            return [
                str(scene)
                for scene in configured
                if str(scene) and str(scene).strip().lower() not in self.ALL_SELECTOR_VALUES
            ]
        return []

    def _remote_scene_names(self) -> list[str]:
        return sorted(
            Path(path).stem
            for path in self._remote_repo_files()
            if Path(path).parent.as_posix() == self._scene_info_dir_name() and Path(path).suffix == ".npz"
        )

    def _selected_scenes_if_available(self) -> list[str]:
        configured_scenes = self._configured_scene_names()
        if configured_scenes:
            return configured_scenes
        scene_info_root = self.paths.raw / self._scene_info_dir_name()
        discovered = sorted(path.stem for path in scene_info_root.glob("*.npz"))
        if discovered:
            return discovered
        return self._remote_scene_names()

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self._list_hf_files(self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _has_local_scene_info_files(self) -> bool:
        configured_scenes = self._configured_scene_names()
        if configured_scenes:
            return all(self._scene_info_path(scene_name).exists() for scene_name in configured_scenes)
        return self._has_any_local_scene_info_files()

    def _remote_download_mode(self) -> str:
        if self._remote_mode_cache is not None:
            return self._remote_mode_cache

        if self._has_local_scene_info_files():
            self._remote_mode_cache = "scene_files"
            return self._remote_mode_cache

        repo_files = set(self._remote_repo_files())
        matching_parts = [path for path in repo_files if path.startswith(self._archive_part_prefix())]
        if matching_parts:
            self._remote_mode_cache = "bundle"
            return self._remote_mode_cache

        configured_scenes = self._configured_scene_names()
        if configured_scenes:
            scene_info_paths = [self._scene_info_repo_path(scene_name) for scene_name in configured_scenes]
            if all(path in repo_files for path in scene_info_paths):
                self._remote_mode_cache = "scene_files"
                return self._remote_mode_cache

        if self._remote_scene_names():
            self._remote_mode_cache = "scene_files"
            return self._remote_mode_cache

        raise FileNotFoundError(
            "MegaDepth remote source does not expose either per-scene files or the configured multipart bundle"
        )

    def _bundle_repo_part_paths(self) -> list[str]:
        return sorted(path for path in self._remote_repo_files() if path.startswith(self._archive_part_prefix()))

    def enumerate_download_units(self) -> Iterable[MegaDepthDownloadUnit]:
        if self._uses_local_archive_root():
            for scene_name in self._selected_scenes():
                yield MegaDepthDownloadUnit(unit_name=scene_name)
            return
        if self._remote_download_mode() == "scene_files":
            for scene_name in self._selected_scenes():
                yield MegaDepthDownloadUnit(unit_name=scene_name)
            return
        for bundle_name in self._selected_bundles():
            yield MegaDepthDownloadUnit(unit_name=bundle_name, unit_type="bundle")

    def download_unit(self, unit: MegaDepthDownloadUnit) -> None:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if self._uses_local_archive_root():
            self._download_from_local_archive_root(unit)
            return
        if self._remote_download_mode() == "scene_files":
            self._download_remote_scene_files(unit)
            return
        self._download_remote_bundle(unit)

    def enumerate_extraction_units(self) -> Iterable[MegaDepthDownloadUnit]:
        if self._status_prefers_scene_file_mode():
            return ()
        return (MegaDepthDownloadUnit(unit_name=self._bundle_unit_name(), unit_type="bundle_extract"),)

    def extract_unit(self, unit: MegaDepthDownloadUnit) -> None:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if self._uses_local_archive_root():
            return
        expected_parts = self._expected_bundle_part_paths()
        part_paths = self._bundle_part_paths()
        if not part_paths:
            raise FileNotFoundError("MegaDepth archive parts are missing; run `dc download` first")
        if len(part_paths) != len(expected_parts):
            raise FileNotFoundError(
                f"MegaDepth extraction requires the complete multipart bundle; "
                f"found {len(part_paths)}/{len(expected_parts)} parts"
            )
        self._extract_multipart_tar_gz(part_paths, self.paths.raw)

    def enumerate_source_items(self) -> Iterable[MegaDepthSourceItem]:
        scene_info_root = self.paths.raw / self._scene_info_dir_name()
        for scene_name in self._selected_scenes():
            scene_info_path = scene_info_root / f"{scene_name}.npz"
            if not scene_info_path.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=scene_name,
                    error_message=f"missing MegaDepth scene info file: {scene_info_path}",
                )
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            image_paths = self._as_string_list(scene_info["image_paths"])
            depth_paths = self._as_string_list(scene_info["depth_paths"])
            intrinsics_array = np.asarray(scene_info["intrinsics"], dtype=np.float32)
            if intrinsics_array.ndim != 3 or intrinsics_array.shape[1:] != (3, 3):
                raise ValueError("MegaDepth intrinsics must have shape (N, 3, 3)")
            count = min(len(image_paths), len(depth_paths), intrinsics_array.shape[0])
            for index in range(count):
                image_relative_path = image_paths[index]
                depth_relative_path = depth_paths[index]
                if not (self.paths.raw / image_relative_path).exists():
                    self.record_error(
                        stage="enumeration",
                        item_id=f"{scene_name}/{index}",
                        error_message=f"missing MegaDepth image file: {image_relative_path}",
                    )
                    continue
                if not (self.paths.raw / depth_relative_path).exists():
                    self.record_error(
                        stage="enumeration",
                        item_id=f"{scene_name}/{index}",
                        error_message=f"missing MegaDepth depth file: {depth_relative_path}",
                    )
                    continue
                intrinsic = intrinsics_array[index]
                yield MegaDepthSourceItem(
                    scene_name=scene_name,
                    image_index=index,
                    image_relative_path=image_relative_path,
                    depth_relative_path=depth_relative_path,
                    intrinsics=(
                        float(intrinsic[0, 0]),
                        float(intrinsic[1, 1]),
                        float(intrinsic[0, 2]),
                        float(intrinsic[1, 2]),
                    ),
                )

    def load_source_item(self, item: MegaDepthSourceItem) -> dict[str, object]:
        image = self._load_image(self.paths.raw / item.image_relative_path)
        depth = self._load_depth(self.paths.raw / item.depth_relative_path)
        return {"image": image, "depth": depth}

    def build_camera_model(self, item: MegaDepthSourceItem, loaded_item: object) -> PinholeCameraModel:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        height, width = image.shape[:2]
        fx, fy, cx, cy = item.intrinsics
        return PinholeCameraModel(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    def build_sample(
        self,
        item: MegaDepthSourceItem,
        loaded_item: object,
        camera_model: PinholeCameraModel,
    ) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth = np.asarray(loaded_item["depth"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if image.shape[:2] != depth.shape:
            raise ValueError("MegaDepth image and depth shapes must match")
        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        distance = z_depth_to_distance(depth[..., None].astype(np.float32), ray_dir).astype(np.float32)
        distance = self._normalize_non_metric_distance(distance)
        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance={
                "scene_name": item.scene_name,
                "image_index": item.image_index,
                "image_relative_path": item.image_relative_path,
                "depth_relative_path": item.depth_relative_path,
                "scale_semantics": "scene-relative",
                "distance_normalization": "[0, 1]",
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
            "metric_scale": False,
            "distance_normalization": "[0, 1]",
        }
        self._write_json_atomic(self.paths.metadata, metadata)

    def validate_output(self) -> None:
        self.validate_processed_output_structure()

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, MegaDepthDownloadUnit)
        return f"{unit.unit_type}:{unit.unit_name}"

    def get_extraction_unit_id(self, unit: object) -> str:
        assert isinstance(unit, MegaDepthDownloadUnit)
        return f"{unit.unit_type}:{unit.unit_name}"

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, MegaDepthSourceItem)
        return f"{item.scene_name}/{item.image_index:06d}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        if self._status_prefers_scene_file_mode():
            for unit in self.enumerate_download_units():
                assert isinstance(unit, MegaDepthDownloadUnit)
                yield self._scene_info_path(unit.unit_name)
            return
        yield self._bundle_download_root()

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if self._uses_local_archive_root() or unit.unit_type == "scene":
            scene_info_path = self._scene_info_path(unit.unit_name)
            if scene_info_path.exists():
                scene_info = np.load(scene_info_path, allow_pickle=True)
                image_paths = self._as_string_list(scene_info["image_paths"])
                depth_paths = self._as_string_list(scene_info["depth_paths"])
                for relative_path in image_paths + depth_paths:
                    path = self.paths.raw / relative_path
                    if path.exists():
                        path.unlink()
                scene_info_path.unlink()
            return
        if self._bundle_download_root().exists():
            shutil.rmtree(self._bundle_download_root())

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if self._uses_local_archive_root() or unit.unit_type == "scene":
            return self._scene_info_path(unit.unit_name)
        return self._bundle_download_root()

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        del unit
        if self._uses_local_archive_root():
            return None
        return self.paths.raw

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if self._uses_local_archive_root() or unit.unit_type == "scene":
            scene_info_path = self._scene_info_path(unit.unit_name)
            if not scene_info_path.exists():
                return False
            scene_info = np.load(scene_info_path, allow_pickle=True)
            image_paths = self._as_string_list(scene_info["image_paths"])
            depth_paths = self._as_string_list(scene_info["depth_paths"])
            return all((self.paths.raw / relative_path).exists() for relative_path in image_paths + depth_paths)
        expected_parts = self._expected_bundle_part_paths()
        if len(self._bundle_part_paths()) != len(expected_parts):
            return False
        selected_scenes = self._selected_scenes_if_available()
        if selected_scenes and not all(self._scene_info_path(scene_name).exists() for scene_name in selected_scenes):
            return False
        return True

    def is_extraction_unit_satisfied(self, unit: object) -> bool:
        del unit
        if self._uses_local_archive_root():
            return True
        scene_info_root = self.paths.raw / self._scene_info_dir_name()
        return scene_info_root.exists() and any(scene_info_root.glob("*.npz"))

    def _status_prefers_scene_file_mode(self) -> bool:
        if self._uses_local_archive_root():
            return True
        if self._has_any_local_scene_info_files():
            return True
        if self.dataset_config.options.get("bundles") is not None:
            return False
        return self._remote_download_mode() == "scene_files"

    def _has_any_local_scene_info_files(self) -> bool:
        scene_info_root = self.paths.raw / self._scene_info_dir_name()
        return scene_info_root.exists() and any(scene_info_root.glob("*.npz"))

    def _download_from_local_archive_root(self, unit: MegaDepthDownloadUnit) -> None:
        source_root = Path(str(self.dataset_config.options["local_archive_root"]))
        destination_root = self.paths.raw
        destination_root.mkdir(parents=True, exist_ok=True)

        source_scene_info = source_root / self._scene_info_dir_name() / f"{unit.unit_name}.npz"
        if not source_scene_info.exists():
            raise FileNotFoundError(f"missing MegaDepth scene info for scene {unit.unit_name}: {source_scene_info}")
        scene_info = np.load(source_scene_info, allow_pickle=True)
        image_paths = self._as_string_list(scene_info["image_paths"])
        depth_paths = self._as_string_list(scene_info["depth_paths"])

        (destination_root / self._scene_info_dir_name()).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_scene_info, destination_root / self._scene_info_dir_name() / source_scene_info.name)
        for relative_path in image_paths + depth_paths:
            source_path = source_root / relative_path
            target_path = destination_root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

    def _download_remote_scene_files(self, unit: MegaDepthDownloadUnit) -> None:
        scene_info_path = self._scene_info_path(unit.unit_name)
        if not scene_info_path.exists():
            self._download_hf_file(
                self.dataset_config.hf_dataset_id,
                self._scene_info_repo_path(unit.unit_name),
                scene_info_path,
            )

        scene_info = np.load(scene_info_path, allow_pickle=True)
        image_paths = self._as_string_list(scene_info["image_paths"])
        depth_paths = self._as_string_list(scene_info["depth_paths"])
        repo_files = set(self._remote_repo_files())
        missing_paths = [path for path in image_paths + depth_paths if path not in repo_files]
        if missing_paths:
            raise FileNotFoundError(
                f"MegaDepth repo is missing direct scene files for {unit.unit_name}: {missing_paths[0]}"
            )

        for relative_path in image_paths + depth_paths:
            target_path = self.paths.raw / relative_path
            if target_path.exists():
                continue
            self._download_hf_file(self.dataset_config.hf_dataset_id, relative_path, target_path)

    def _download_remote_bundle(self, unit: MegaDepthDownloadUnit) -> None:
        del unit
        bundle_root = self._bundle_download_root()
        bundle_root.mkdir(parents=True, exist_ok=True)
        matching_parts = self._bundle_repo_part_paths()
        if not matching_parts:
            raise FileNotFoundError(
                f"no MegaDepth archive parts matching `{self._archive_part_prefix()}` were found in "
                f"`{self.dataset_config.hf_dataset_id}`"
            )
        for repo_path in matching_parts:
            target_path = bundle_root / Path(repo_path).name
            if target_path.exists():
                continue
            self._download_hf_file(self.dataset_config.hf_dataset_id, repo_path, target_path)
        for scene_name in self._selected_scenes_if_available():
            scene_info_path = self._scene_info_path(scene_name)
            if scene_info_path.exists():
                continue
            self._download_hf_file(
                self.dataset_config.hf_dataset_id,
                self._scene_info_repo_path(scene_name),
                scene_info_path,
            )

    def get_download_progress_plan(self, unit: object) -> dict[str, object] | None:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if self._uses_local_archive_root() or self._remote_download_mode() == "scene_files":
            return None

        total_files = len(self._bundle_repo_part_paths())
        total_files += len(self._selected_scenes_if_available())
        return {
            "label": unit.unit_name,
            "root": self._downloads_root(),
            "total_files": max(1, total_files),
            "mode": "bundle_parts",
        }

    def get_download_progress_status(self, unit: object) -> dict[str, object] | None:
        assert isinstance(unit, MegaDepthDownloadUnit)
        if unit.unit_type != "bundle":
            return None

        completed_files = len(self._bundle_part_paths())
        completed_files += sum(
            1 for scene_name in self._selected_scenes_if_available() if self._scene_info_path(scene_name).exists()
        )
        current_file = ""
        current_size_bytes = 0
        for repo_path in self._bundle_repo_part_paths():
            path = self._bundle_download_root() / Path(repo_path).name
            if path.exists():
                continue
            partial_path = path.with_suffix(path.suffix + ".incomplete")
            if partial_path.exists():
                current_file = partial_path.name
                current_size_bytes = partial_path.stat().st_size
            else:
                current_file = path.name
            break
        if not current_file:
            for scene_name in self._selected_scenes_if_available():
                scene_info_path = self._scene_info_path(scene_name)
                if scene_info_path.exists():
                    continue
                current_file = scene_info_path.name
                break
        return {
            "completed_files": completed_files,
            "current_file": current_file,
            "current_size_bytes": current_size_bytes,
        }

    def _expected_bundle_part_paths(self) -> list[Path]:
        return [self._bundle_download_root() / Path(repo_path).name for repo_path in self._bundle_repo_part_paths()]

    def _bundle_part_paths(self) -> list[Path]:
        expected_paths = self._expected_bundle_part_paths()
        return [path for path in expected_paths if path.exists()]

    def _list_hf_files(self, repo_id: str) -> list[str]:
        return self.hf_list_repo_files(repo_id=repo_id, repo_type="dataset")

    def _download_hf_file(self, repo_id: str, repo_path: str, target_path: Path) -> Path:
        cached_path = self.hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            local_dir=self.paths.hf_cache / "downloads",
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, target_path)
        return target_path

    def _extract_multipart_tar_gz(self, part_paths: list[Path], destination_root: Path) -> None:
        reader = _ConcatBinaryReader(part_paths)
        try:
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    archive.extract(member, path=destination_root)
        finally:
            reader.close()

    def _extract_tar_archive(self, archive_path: Path, destination_root: Path) -> None:
        mode = "r"
        if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
            mode = "r:gz"
        elif archive_path.suffixes and archive_path.suffixes[-1] == ".gz":
            mode = "r:gz"
        with tarfile.open(archive_path, mode) as archive:
            archive.extractall(path=destination_root)

    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0

    def _load_depth(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        if suffix == ".npz":
            payload = np.load(path)
            if "data" in payload:
                return np.asarray(payload["data"], dtype=np.float32)
            return np.asarray(payload[payload.files[0]], dtype=np.float32)
        if suffix in {".h5", ".hdf5"}:
            return self._load_hdf5_array(path)
        raise ValueError(f"unsupported MegaDepth depth file suffix: {path.suffix}")

    def _normalize_non_metric_distance(self, distance: np.ndarray) -> np.ndarray:
        valid_mask = np.isfinite(distance) & (distance > 1e-6)
        normalized = np.ones_like(distance, dtype=np.float32)
        if not np.any(valid_mask):
            return normalized
        max_distance = float(np.max(distance[valid_mask]))
        if max_distance <= 1e-6:
            return normalized
        normalized[valid_mask] = distance[valid_mask] / max_distance
        return clip_distance_to_max_dist(normalized.astype(np.float32), 1.0)

    def _load_hdf5_array(self, path: Path) -> np.ndarray:
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("MegaDepth processing requires h5py for HDF5 depth maps") from exc
        with h5py.File(path, "r") as handle:
            if len(handle.keys()) == 1:
                return np.asarray(handle[next(iter(handle.keys()))], dtype=np.float32)
            if "dataset" in handle:
                return np.asarray(handle["dataset"], dtype=np.float32)
            raise ValueError(f"could not determine dataset key in HDF5 file: {path}")

    def _as_string_list(self, values: object) -> list[str]:
        array = np.asarray(values)
        return [str(item) for item in array.tolist()]


    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(len(shard_names) * self.config.project.train_val_split)
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "MegaDepthDownloadUnit",
    "MegaDepthPipeline",
    "MegaDepthSourceItem",
]
