from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import shutil
import zipfile
from typing import Iterable

import numpy as np
from PIL import Image

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import PinholeCameraModel, clip_distance_to_max_dist, generate_pinhole_rays
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class UnrealStereo4KArchiveUnit:
    archive_name: str
    repo_path: str


@dataclass(frozen=True)
class UnrealStereo4KSourceItem:
    scene_name: str
    frame_key: str
    archive_name: str
    image_relative_path: str
    disparity_relative_path: str


class UnrealStereo4KPipeline(DatasetPipeline):
    """Archive-backed UnrealStereo4K-Q pipeline using non-metric inverse-disparity depth."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    DEFAULT_CAMERA_INTRINSICS = {
        "width": 960.0,
        "height": 540.0,
        "fx": 960.0,
        "fy": 960.0,
        "cx": 480.0,
        "cy": 270.0,
    }
    DEFAULT_MINIMUM_READABLE_ARCHIVE = "00008.zip"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None

    def is_metric_scale(self) -> bool:
        return False

    def _downloads_root(self) -> Path:
        return self.paths.raw / "_downloads"

    def _archive_root(self) -> Path:
        return self._downloads_root()

    def _camera_intrinsics(self) -> dict[str, float]:
        configured = self.dataset_config.options.get("camera_intrinsics", {})
        assert isinstance(configured, dict)
        values = dict(self.DEFAULT_CAMERA_INTRINSICS)
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if key in configured:
                values[key] = float(configured[key])
        return values

    def _configured_archive_names(self) -> list[str]:
        configured = self.dataset_config.options.get("archives", "*")
        if isinstance(configured, str):
            normalized = configured.strip().lower()
            if normalized in self.ALL_SELECTOR_VALUES or not normalized:
                return []
            return [configured]
        return [
            str(archive_name)
            for archive_name in configured
            if str(archive_name).strip().lower() not in self.ALL_SELECTOR_VALUES and str(archive_name).strip()
        ]

    def _minimum_readable_archive_name(self) -> str:
        return str(
            self.dataset_config.options.get("minimum_readable_archive", self.DEFAULT_MINIMUM_READABLE_ARCHIVE)
        ).strip()

    def _image_dir_candidates(self) -> tuple[str, ...]:
        configured = self.dataset_config.options.get("image_dir")
        if isinstance(configured, str) and configured.strip():
            return (configured.strip().strip("/"),)
        return ("Image0", "frames_cleanpass/left")

    def _disparity_dir_candidates(self) -> tuple[str, ...]:
        configured = self.dataset_config.options.get("disparity_dir")
        if isinstance(configured, str) and configured.strip():
            return (configured.strip().strip("/"),)
        return ("Disp0", "disparity/left")

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(self.dataset_config.hf_dataset_id, repo_type="dataset"))
        return self._remote_repo_files_cache

    def _available_remote_archive_paths(self) -> list[str]:
        return sorted(path for path in self._remote_repo_files() if Path(path).suffix.lower() == ".zip")

    def _available_local_archive_paths(self) -> list[Path]:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if not local_archive_root:
            return []
        root = Path(str(local_archive_root))
        if not root.exists():
            return []
        return sorted(root.rglob("*.zip"))

    def _available_extracted_scene_roots(self) -> list[Path]:
        if not self.paths.raw.exists():
            return []
        return sorted(
            path
            for path in self.paths.raw.iterdir()
            if path.is_dir() and path.name != "_downloads" and not path.name.startswith(".")
        )

    def _selected_archive_units(self) -> list[UnrealStereo4KArchiveUnit]:
        downloaded_archives = sorted(self._archive_root().glob("*.zip"))
        if downloaded_archives:
            archive_by_name = {path.name: path for path in downloaded_archives}
            configured_names = self._configured_archive_names()
            if configured_names:
                selected_names = [name for name in configured_names if name in archive_by_name]
            elif self.dataset_selection() == self.MINIMUM_READABLE_SELECTION:
                selected_names = [
                    name for name in [self._minimum_readable_archive_name()] if name in archive_by_name
                ] or sorted(archive_by_name)
            else:
                selected_names = sorted(archive_by_name)
            selected_names = [str(name) for name in self.apply_dataset_selection(selected_names)]
            return [
                UnrealStereo4KArchiveUnit(archive_name=name, repo_path=str(archive_by_name[name]))
                for name in selected_names
            ]

        available_local = self._available_local_archive_paths()
        if available_local:
            archive_by_name = {path.name: path for path in available_local}
            configured_names = self._configured_archive_names()
            if configured_names:
                selected_names = [name for name in configured_names if name in archive_by_name]
            elif self.dataset_selection() == self.MINIMUM_READABLE_SELECTION:
                selected_names = [
                    name for name in [self._minimum_readable_archive_name()] if name in archive_by_name
                ] or sorted(archive_by_name)
            else:
                selected_names = sorted(archive_by_name)
            selected_names = [str(name) for name in self.apply_dataset_selection(selected_names)]
            return [
                UnrealStereo4KArchiveUnit(archive_name=name, repo_path=str(archive_by_name[name]))
                for name in selected_names
            ]

        extracted_roots = self._available_extracted_scene_roots()
        if extracted_roots:
            archive_by_name = {f"{path.name}.zip": path for path in extracted_roots}
            configured_names = self._configured_archive_names()
            if configured_names:
                selected_names = [name for name in configured_names if name in archive_by_name]
            elif self.dataset_selection() == self.MINIMUM_READABLE_SELECTION:
                selected_names = [
                    name for name in [self._minimum_readable_archive_name()] if name in archive_by_name
                ] or sorted(archive_by_name)
            else:
                selected_names = sorted(archive_by_name)
            selected_names = [str(name) for name in self.apply_dataset_selection(selected_names)]
            return [
                UnrealStereo4KArchiveUnit(archive_name=name, repo_path=str(archive_by_name[name]))
                for name in selected_names
            ]

        available_remote = self._available_remote_archive_paths()
        archive_by_name = {Path(repo_path).name: repo_path for repo_path in available_remote}
        configured_names = self._configured_archive_names()
        if configured_names:
            selected_names = [name for name in configured_names if name in archive_by_name]
        elif self.dataset_selection() == self.MINIMUM_READABLE_SELECTION:
            selected_names = [
                name for name in [self._minimum_readable_archive_name()] if name in archive_by_name
            ] or sorted(archive_by_name)
        else:
            selected_names = sorted(archive_by_name)
        selected_names = [str(name) for name in self.apply_dataset_selection(selected_names)]
        return [
            UnrealStereo4KArchiveUnit(archive_name=name, repo_path=archive_by_name[name])
            for name in selected_names
        ]

    def enumerate_download_units(self) -> Iterable[UnrealStereo4KArchiveUnit]:
        return self._selected_archive_units()

    def _archive_path(self, unit: UnrealStereo4KArchiveUnit) -> Path:
        return self._archive_root() / unit.archive_name

    def _resolve_local_source_archive_path(self, unit: UnrealStereo4KArchiveUnit) -> Path:
        local_archive_root = Path(str(self.dataset_config.options["local_archive_root"]))
        candidate_paths = [
            Path(unit.repo_path),
            local_archive_root / unit.archive_name,
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"missing local UnrealStereo4K archive source for {unit.archive_name}")

    def download_unit(self, unit: UnrealStereo4KArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if self.dataset_selection() == self.MINIMUM_READABLE_SELECTION:
            if local_archive_root:
                source_path = self._resolve_local_source_archive_path(unit)
                self._write_minimum_readable_archive_from_local_source(source_path, archive_path)
                return
            self._write_minimum_readable_archive_from_remote_source(unit.repo_path, archive_path)
            return
        if local_archive_root:
            source_path = self._resolve_local_source_archive_path(unit)
            shutil.copy2(source_path, archive_path)
            return
        downloaded_path = self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            filename=unit.repo_path,
            repo_type="dataset",
            revision=self.dataset_config.revision,
            local_dir=self.paths.hf_cache / "downloads",
        )
        shutil.copy2(downloaded_path, archive_path)

    def _write_minimum_readable_archive_from_local_source(self, source_path: Path, target_path: Path) -> None:
        with zipfile.ZipFile(source_path) as archive:
            self._write_minimum_readable_archive_from_zip(archive, target_path)

    def _write_minimum_readable_archive_from_remote_source(self, repo_path: str, target_path: Path) -> None:
        with self.hf_open_remote_zip(
            repo_id=self.dataset_config.hf_dataset_id,
            filename=repo_path,
            repo_type="dataset",
            revision=self.dataset_config.revision,
        ) as archive:
            self._write_minimum_readable_archive_from_zip(archive, target_path)

    def _write_minimum_readable_archive_from_zip(self, archive: zipfile.ZipFile, target_path: Path) -> None:
        image_members: dict[str, str] = {}
        disparity_members: dict[str, str] = {}
        for member_name in sorted(archive.namelist()):
            descriptor = self._classify_member(member_name)
            if descriptor is None:
                continue
            frame_key, label = descriptor
            if label == "image" and frame_key not in image_members:
                image_members[frame_key] = member_name
            elif label == "disparity" and frame_key not in disparity_members:
                disparity_members[frame_key] = member_name
            if frame_key in image_members and frame_key in disparity_members:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as output_archive:
                    for required_name in (image_members[frame_key], disparity_members[frame_key]):
                        output_archive.writestr(required_name, archive.read(required_name))
                return
        raise ValueError(
            f"{self.dataset_name} could not identify a readable UnrealStereo4K sample in {target_path.name}: "
            "no archive member set contained paired left image and disparity files"
        )

    def _classify_member(self, member_name: str) -> tuple[str, str] | None:
        path = Path(member_name)
        if path.name.startswith(".") or member_name.endswith("/"):
            return None
        normalized = path.as_posix().strip("/")
        if normalized.endswith((".png", ".jpg", ".jpeg")) and self._path_contains_dir(normalized, self._image_dir_candidates()):
            return self._frame_key(path), "image"
        if normalized.endswith((".npy", ".npz")) and self._path_contains_dir(normalized, self._disparity_dir_candidates()):
            return self._frame_key(path), "disparity"
        return None

    def _path_contains_dir(self, normalized_path: str, candidates: tuple[str, ...]) -> bool:
        path_parts = tuple(part.lower() for part in Path(normalized_path).parts)
        for candidate in candidates:
            candidate_parts = tuple(part.lower() for part in Path(candidate).parts)
            if len(candidate_parts) > len(path_parts):
                continue
            for index in range(len(path_parts) - len(candidate_parts) + 1):
                if path_parts[index : index + len(candidate_parts)] == candidate_parts:
                    return True
        return False

    def _frame_key(self, path: Path) -> str:
        return str(path.with_suffix("").name)

    def enumerate_extraction_units(self) -> Iterable[UnrealStereo4KArchiveUnit]:
        return self.enumerate_download_units()

    def extract_unit(self, unit: UnrealStereo4KArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        if not archive_path.exists():
            raise FileNotFoundError(f"missing UnrealStereo4K archive for extraction: {archive_path}")
        self.paths.raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(self.paths.raw)

    def enumerate_source_items(self) -> Iterable[UnrealStereo4KSourceItem]:
        for scene_root in self._scene_roots():
            image_root = self._find_existing_subdir(scene_root, self._image_dir_candidates())
            disparity_root = self._find_existing_subdir(scene_root, self._disparity_dir_candidates())
            if not image_root.exists() or not disparity_root.exists():
                self.record_error(
                    "enumeration",
                    scene_root.name,
                    f"incomplete UnrealStereo4K frame layout under {scene_root}",
                )
                continue
            disparity_by_key = self._files_by_frame_key(disparity_root)
            archive_name = self._scene_archive_name(scene_root.name)
            for image_path in sorted(self._iter_supported_files(image_root)):
                frame_key = self._frame_key(image_path)
                disparity_path = disparity_by_key.get(frame_key)
                if disparity_path is None:
                    self.record_error(
                        "enumeration",
                        f"{scene_root.name}/{frame_key}",
                        f"missing paired UnrealStereo4K disparity file for frame {frame_key}",
                    )
                    continue
                yield UnrealStereo4KSourceItem(
                    scene_name=scene_root.name,
                    frame_key=frame_key,
                    archive_name=archive_name,
                    image_relative_path=str(image_path.relative_to(self.paths.raw)),
                    disparity_relative_path=str(disparity_path.relative_to(self.paths.raw)),
                )

    def _scene_roots(self) -> list[Path]:
        return self._available_extracted_scene_roots()

    def _find_existing_subdir(self, root: Path, candidates: tuple[str, ...]) -> Path:
        for candidate in candidates:
            candidate_path = root / candidate
            if candidate_path.exists():
                return candidate_path
        return root / candidates[0]

    def _scene_archive_name(self, scene_name: str) -> str:
        configured_names = self._configured_archive_names()
        if len(configured_names) == 1:
            return configured_names[0]
        local_archive_names = sorted(path.name for path in self._archive_root().glob("*.zip"))
        for archive_name in local_archive_names:
            if Path(archive_name).stem == scene_name:
                return archive_name
        return f"{scene_name}.zip"

    def _iter_supported_files(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            return ()
        return (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".npz"}
        )

    def _files_by_frame_key(self, root: Path) -> dict[str, Path]:
        return {self._frame_key(path): path for path in self._iter_supported_files(root)}

    def load_source_item(self, item: UnrealStereo4KSourceItem) -> dict[str, object]:
        image = np.asarray(Image.open(self.paths.raw / item.image_relative_path).convert("RGB"), dtype=np.float32) / 255.0
        disparity = self._load_disparity(self.paths.raw / item.disparity_relative_path)
        return {
            "image": image,
            "disparity": disparity,
        }

    def build_camera_model(self, item: UnrealStereo4KSourceItem, loaded_item: object) -> PinholeCameraModel:
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

    def build_sample(
        self,
        item: UnrealStereo4KSourceItem,
        loaded_item: object,
        camera_model: PinholeCameraModel,
    ) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        disparity = np.asarray(loaded_item["disparity"], dtype=np.float32)
        if disparity.ndim == 3 and disparity.shape[-1] == 1:
            disparity = disparity[..., 0]
        if disparity.ndim != 2:
            raise ValueError("UnrealStereo4K disparity must decode to a 2D array")
        if image.shape[:2] != disparity.shape:
            raise ValueError("UnrealStereo4K image and disparity shapes must match")

        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        distance = self._relative_distance_from_disparity(disparity)
        invalid_mask = ~np.isfinite(distance[..., 0]) | (distance[..., 0] <= 1e-6)
        distance = self._normalize_non_metric_distance(distance, invalid_mask)
        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance={
                "scene_name": item.scene_name,
                "frame_key": item.frame_key,
                "archive_name": item.archive_name,
                "projection": "pinhole",
                "scale_semantics": "scene-relative",
                "depth_semantics": "inverse_disparity_relative",
                "distance_normalization": "[0, 1]",
            },
        )

    def _relative_distance_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        distance = np.full((*disparity.shape, 1), np.nan, dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > 1e-6)
        distance[..., 0][valid] = 1.0 / disparity[valid]
        return distance

    def _normalize_non_metric_distance(self, distance: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
        distance_2d = np.asarray(distance[..., 0], dtype=np.float32)
        valid_mask = np.isfinite(distance_2d) & (distance_2d > 1e-6) & ~invalid_mask
        normalized = np.ones((*distance.shape[:2], 1), dtype=np.float32)
        if not np.any(valid_mask):
            return normalized
        max_distance = float(np.max(distance_2d[valid_mask]))
        if max_distance <= 1e-6:
            return normalized
        normalized[..., 0][valid_mask] = distance_2d[valid_mask] / max_distance
        normalized[..., 0][invalid_mask] = 1.0
        return clip_distance_to_max_dist(normalized, 1.0)

    def _load_disparity(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        if suffix == ".npz":
            with np.load(path, allow_pickle=True) as payload:
                if "data" in payload:
                    return np.asarray(payload["data"], dtype=np.float32)
                return np.asarray(payload[payload.files[0]], dtype=np.float32)
        raise ValueError(f"unsupported UnrealStereo4K disparity file suffix: {path.suffix}")

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
        assert isinstance(unit, UnrealStereo4KArchiveUnit)
        return unit.archive_name

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, UnrealStereo4KSourceItem)
        return f"{item.scene_name}/{item.frame_key}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self.enumerate_download_units():
            yield self._archive_path(unit)

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, UnrealStereo4KArchiveUnit)
        return self._archive_path(unit)

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, UnrealStereo4KArchiveUnit)
        del unit
        return self.paths.raw

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, UnrealStereo4KArchiveUnit)
        archive_path = self._archive_path(unit)
        if archive_path.exists():
            archive_path.unlink()

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, UnrealStereo4KArchiveUnit)
        archive_path = self._archive_path(unit)
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
    "UnrealStereo4KArchiveUnit",
    "UnrealStereo4KPipeline",
    "UnrealStereo4KSourceItem",
]
