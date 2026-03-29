from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
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
class WMGStereoArchiveUnit:
    category: str
    archive_name: str
    repo_path: str


@dataclass(frozen=True)
class WMGStereoSourceItem:
    seed_name: str
    frame_key: str
    archive_name: str
    image_relative_path: str
    disparity_relative_path: str
    left_camview_relative_path: str
    right_camview_relative_path: str
    occ_mask_relative_path: str | None = None
    sky_mask_relative_path: str | None = None


class WMGStereoPipeline(DatasetPipeline, ABC):
    """Shared family pipeline for category-specific WMGStereo subsets."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    CATEGORY_NAME = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None

    def is_metric_scale(self) -> bool:
        return False

    def _category_name(self) -> str:
        if not self.CATEGORY_NAME:
            raise ValueError("WMGStereo concrete pipelines must define CATEGORY_NAME")
        return self.CATEGORY_NAME

    def _release_name(self) -> str:
        return str(self.dataset_config.options.get("release", "release_full"))

    def _downloads_root(self) -> Path:
        return self.paths.raw / "_downloads"

    def _archive_root(self) -> Path:
        return self._downloads_root() / self._release_name() / self._category_name()

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(repo_id=self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _available_remote_archive_paths(self) -> list[str]:
        category = self._category_name()
        release = self._release_name()
        archive_paths: list[str] = []
        for repo_path in self._remote_repo_files():
            path = Path(repo_path)
            if path.suffixes[-2:] != [".tar", ".gz"]:
                continue
            parts = path.parts
            if len(parts) >= 3 and parts[0] == release and parts[1] == category:
                archive_paths.append(repo_path)
                continue
            if len(parts) >= 2 and parts[0] == category:
                archive_paths.append(repo_path)
        return sorted(archive_paths)

    def _available_local_archive_paths(self) -> list[Path]:
        roots = [
            Path(str(self.dataset_config.options["local_archive_root"])) / self._release_name() / self._category_name(),
            Path(str(self.dataset_config.options["local_archive_root"])) / self._category_name(),
        ]
        archive_paths: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            archive_paths.extend(sorted(root.glob("*.tar.gz")))
        deduped: dict[Path, None] = {}
        for path in archive_paths:
            deduped[path] = None
        return list(deduped)

    def _configured_archive_names(self) -> list[str]:
        configured = self.dataset_config.options.get("archives", "*")
        if isinstance(configured, str):
            if configured.strip().lower() in self.ALL_SELECTOR_VALUES:
                return []
            return [configured]
        return [
            str(archive_name)
            for archive_name in configured
            if str(archive_name).strip().lower() not in self.ALL_SELECTOR_VALUES
        ]

    def _selected_archive_count(self, discovered_count: int) -> int:
        configured = self.dataset_config.options.get("archive_count")
        if configured is None:
            return discovered_count
        count = int(configured)
        if count < 1:
            raise ValueError("archive_count must be at least 1")
        return min(count, discovered_count)

    def _selected_archive_units(self) -> list[WMGStereoArchiveUnit]:
        downloaded_archives = sorted(self._archive_root().glob("*.tar.gz"))
        if downloaded_archives:
            archive_by_name = {path.name: path for path in downloaded_archives}
            configured_names = self._configured_archive_names()
            if configured_names:
                selected_names = [name for name in configured_names if name in archive_by_name]
            else:
                selected_names = sorted(archive_by_name)
            selected_names = selected_names[: self._selected_archive_count(len(selected_names))]
            return [
                WMGStereoArchiveUnit(
                    category=self._category_name(),
                    archive_name=name,
                    repo_path=str(archive_by_name[name]),
                )
                for name in selected_names
            ]

        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            available_local = self._available_local_archive_paths()
            archive_by_name = {path.name: path for path in available_local}
            configured_names = self._configured_archive_names()
            if configured_names:
                selected_names = [name for name in configured_names if name in archive_by_name]
            else:
                selected_names = sorted(archive_by_name)
            selected_names = selected_names[: self._selected_archive_count(len(selected_names))]
            return [
                WMGStereoArchiveUnit(
                    category=self._category_name(),
                    archive_name=name,
                    repo_path=str(archive_by_name[name]),
                )
                for name in selected_names
            ]

        available_remote = self._available_remote_archive_paths()
        archive_by_name = {Path(repo_path).name: repo_path for repo_path in available_remote}
        configured_names = self._configured_archive_names()
        if configured_names:
            selected_names = [name for name in configured_names if name in archive_by_name]
        else:
            selected_names = sorted(archive_by_name)
        selected_names = selected_names[: self._selected_archive_count(len(selected_names))]
        return [
            WMGStereoArchiveUnit(
                category=self._category_name(),
                archive_name=name,
                repo_path=archive_by_name[name],
            )
            for name in selected_names
        ]

    def enumerate_download_units(self) -> Iterable[WMGStereoArchiveUnit]:
        return self._selected_archive_units()

    def is_partial_download_build(self) -> bool:
        configured_names = self._configured_archive_names()
        configured_count = self.dataset_config.options.get("archive_count")
        if configured_names:
            if configured_count is None:
                return False
            return int(configured_count) < len(configured_names)
        if configured_count is not None:
            return True
        return False

    def _archive_path(self, unit: WMGStereoArchiveUnit) -> Path:
        return self._archive_root() / unit.archive_name

    def _archive_seed_name(self, archive_name: str) -> str:
        if archive_name.endswith(".tar.gz"):
            return archive_name[: -len(".tar.gz")]
        return Path(archive_name).stem

    def _download_hf_file(self, repo_path: str, target_path: Path) -> Path:
        cached_path = self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            filename=repo_path,
            repo_type="dataset",
            local_dir=self.paths.hf_cache / "downloads",
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, target_path)
        return target_path

    def download_unit(self, unit: WMGStereoArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_path = Path(unit.repo_path)
            if not source_path.exists():
                raise FileNotFoundError(f"missing local WMGStereo archive source: {source_path}")
            shutil.copy2(source_path, archive_path)
            return
        self._download_hf_file(unit.repo_path, archive_path)

    def enumerate_extraction_units(self) -> Iterable[WMGStereoArchiveUnit]:
        return self._selected_archive_units()

    def extract_unit(self, unit: WMGStereoArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        if not archive_path.exists():
            raise FileNotFoundError(f"missing archive for extraction: {archive_path}")
        self.paths.raw.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(path=self.paths.raw, filter="data")

    def enumerate_source_items(self) -> Iterable[WMGStereoSourceItem]:
        for seed_root in self._seed_roots():
            image_root = seed_root / "frames" / "Image" / "camera_0"
            disparity_root = seed_root / "frames" / "disparity" / "camera_0"
            camview0_root = seed_root / "frames" / "camview" / "camera_0"
            camview1_root = seed_root / "frames" / "camview" / "camera_1"
            if not image_root.exists() or not disparity_root.exists() or not camview0_root.exists() or not camview1_root.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=seed_root.name,
                    error_message=f"incomplete WMGStereo frame layout under {seed_root}",
                )
                continue
            disparity_by_key = self._files_by_frame_key(disparity_root)
            left_camview_by_key = self._files_by_frame_key(camview0_root)
            right_camview_by_key = self._files_by_frame_key(camview1_root)
            occ_mask_by_key = self._files_by_frame_key(seed_root / "frames" / "occ_mask" / "camera_0")
            if not occ_mask_by_key:
                occ_mask_by_key = self._files_by_frame_key(seed_root / "frames" / "disparity_masks" / "camera_0")
            sky_mask_by_key = self._files_by_frame_key(seed_root / "frames" / "sky_mask" / "camera_0")
            archive_name = self._seed_archive_name(seed_root.name)
            for image_path in sorted(self._iter_supported_files(image_root)):
                frame_key = self._frame_key_from_path(image_path)
                disparity_path = disparity_by_key.get(frame_key)
                left_camview_path = left_camview_by_key.get(frame_key)
                right_camview_path = right_camview_by_key.get(frame_key)
                if disparity_path is None or left_camview_path is None or right_camview_path is None:
                    self.record_error(
                        stage="enumeration",
                        item_id=f"{seed_root.name}/{frame_key}",
                        error_message=f"missing paired WMGStereo files for frame {frame_key}",
                    )
                    continue
                yield WMGStereoSourceItem(
                    seed_name=seed_root.name,
                    frame_key=frame_key,
                    archive_name=archive_name,
                    image_relative_path=str(image_path.relative_to(self.paths.raw)),
                    disparity_relative_path=str(disparity_path.relative_to(self.paths.raw)),
                    left_camview_relative_path=str(left_camview_path.relative_to(self.paths.raw)),
                    right_camview_relative_path=str(right_camview_path.relative_to(self.paths.raw)),
                    occ_mask_relative_path=(
                        None if occ_mask_by_key.get(frame_key) is None else str(occ_mask_by_key[frame_key].relative_to(self.paths.raw))
                    ),
                    sky_mask_relative_path=(
                        None if sky_mask_by_key.get(frame_key) is None else str(sky_mask_by_key[frame_key].relative_to(self.paths.raw))
                    ),
                )

    def _seed_roots(self) -> list[Path]:
        if not self.paths.raw.exists():
            return []
        return sorted(
            path
            for path in self.paths.raw.iterdir()
            if path.is_dir() and path.name != "_downloads" and not path.name.startswith(".")
        )

    def _seed_archive_name(self, seed_name: str) -> str:
        configured_names = self._configured_archive_names()
        if len(configured_names) == 1:
            return configured_names[0]
        local_archive_names = sorted(path.name for path in self._archive_root().glob("*.tar.gz"))
        for archive_name in local_archive_names:
            archive_stem = self._archive_seed_name(archive_name)
            if archive_stem == seed_name:
                return archive_name
            if "-" in archive_stem:
                return archive_name
        return f"{seed_name}.tar.gz"

    def _iter_supported_files(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            return ()
        return (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".npz"}
        )

    def _files_by_frame_key(self, root: Path) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for path in self._iter_supported_files(root):
            mapping[self._frame_key_from_path(path)] = path
        return mapping

    def _frame_key_from_path(self, path: Path) -> str:
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 4:
            return f"{parts[1]}_{parts[3]}"
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
        if len(parts) >= 2:
            return parts[1]
        return stem

    def load_source_item(self, item: WMGStereoSourceItem) -> dict[str, object]:
        image = self._load_image(self.paths.raw / item.image_relative_path)
        disparity = self._load_array(self.paths.raw / item.disparity_relative_path)
        left_camview = self._load_camview(self.paths.raw / item.left_camview_relative_path)
        right_camview = self._load_camview(self.paths.raw / item.right_camview_relative_path)
        sky_mask = None
        if item.sky_mask_relative_path is not None:
            sky_mask = self._load_mask(self.paths.raw / item.sky_mask_relative_path)
        return {
            "image": image,
            "disparity": disparity,
            "left_camview": left_camview,
            "right_camview": right_camview,
            "sky_mask": sky_mask,
        }

    def build_camera_model(self, item: WMGStereoSourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        left_camview = loaded_item["left_camview"]
        assert isinstance(left_camview, dict)
        k_matrix = np.asarray(left_camview["K"], dtype=np.float32)
        height, width = image.shape[:2]
        return PinholeCameraModel(
            width=width,
            height=height,
            fx=float(k_matrix[0, 0]),
            fy=float(k_matrix[1, 1]),
            cx=float(k_matrix[0, 2]),
            cy=float(k_matrix[1, 2]),
        )

    def build_sample(
        self,
        item: WMGStereoSourceItem,
        loaded_item: object,
        camera_model: PinholeCameraModel,
    ) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        disparity = np.asarray(loaded_item["disparity"], dtype=np.float32)
        left_camview = loaded_item["left_camview"]
        right_camview = loaded_item["right_camview"]
        sky_mask = loaded_item.get("sky_mask")
        assert isinstance(left_camview, dict)
        assert isinstance(right_camview, dict)
        if disparity.ndim == 3 and disparity.shape[-1] == 1:
            disparity = disparity[..., 0]
        if disparity.ndim != 2:
            raise ValueError("WMGStereo disparity must decode to a 2D array")
        if image.shape[:2] != disparity.shape:
            raise ValueError("WMGStereo image and disparity shapes must match")

        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        z_depth = self._depth_from_disparity(
            disparity=disparity,
            left_camview=left_camview,
            right_camview=right_camview,
        )
        distance = z_depth_to_distance(z_depth[..., None].astype(np.float32), ray_dir).astype(np.float32)
        distance = np.clip(distance, 0.0, self.config.project.max_dist)
        invalid_mask = ~np.isfinite(distance[..., 0]) | (distance[..., 0] <= 1e-6)
        if sky_mask is not None:
            sky_mask_array = np.asarray(sky_mask, dtype=bool)
            distance[..., 0][sky_mask_array] = self.config.project.max_dist
            invalid_mask |= sky_mask_array
        distance = self._normalize_non_metric_distance(distance, invalid_mask)
        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance={
                "category": self._category_name(),
                "seed_name": item.seed_name,
                "frame_key": item.frame_key,
                "archive_name": item.archive_name,
                "scale_semantics": "scene-relative",
                "distance_normalization": "[0, 1]",
            },
        )

    def _depth_from_disparity(
        self,
        *,
        disparity: np.ndarray,
        left_camview: dict[str, object],
        right_camview: dict[str, object],
    ) -> np.ndarray:
        left_k = np.asarray(left_camview["K"], dtype=np.float32)
        fx = float(left_k[0, 0])
        left_translation = self._translation_vector(left_camview["T"])
        right_translation = self._translation_vector(right_camview["T"])
        baseline = float(np.linalg.norm(right_translation - left_translation))
        if baseline <= 1e-9:
            raise ValueError("WMGStereo stereo baseline must be non-zero")
        denominator = disparity.astype(np.float32)
        z_depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(denominator) & (denominator > 1e-6)
        z_depth[valid] = (fx * baseline) / denominator[valid]
        return z_depth

    def _translation_vector(self, value: object) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (4, 4):
            return array[:3, 3]
        if array.shape == (3, 4):
            return array[:3, 3]
        if array.shape == (3,):
            return array
        if array.shape == (1, 3) or array.shape == (3, 1):
            return array.reshape(3)
        raise ValueError(f"unsupported WMGStereo translation shape: {array.shape}")

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
            "category": self._category_name(),
        }
        self._write_json_atomic(self.paths.metadata, metadata)

    def validate_output(self) -> None:
        if not self.paths.metadata.exists():
            raise ValueError("metadata.json was not created")

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, WMGStereoArchiveUnit)
        return f"{unit.category}/{unit.archive_name}"

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, WMGStereoSourceItem)
        return f"{item.seed_name}/{item.frame_key}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self._selected_archive_units():
            yield self._archive_path(unit)

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, WMGStereoArchiveUnit)
        archive_path = self._archive_path(unit)
        if archive_path.exists():
            archive_path.unlink()

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, WMGStereoArchiveUnit)
        return self._archive_path(unit)

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, WMGStereoArchiveUnit)
        return self.paths.raw

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, WMGStereoArchiveUnit)
        return self._archive_path(unit).exists()

    def is_extraction_unit_satisfied(self, unit: object) -> bool:
        del unit
        seed_roots = self._seed_roots()
        if not seed_roots:
            return False
        return any(any(path.is_file() for path in seed_root.rglob("*")) for seed_root in seed_roots)

    def _load_image(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def _load_mask(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32) > 0.5

    def _load_array(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        if suffix == ".npz":
            with np.load(path, allow_pickle=True) as payload:
                if "data" in payload:
                    return np.asarray(payload["data"], dtype=np.float32)
                return np.asarray(payload[payload.files[0]], dtype=np.float32)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return np.asarray(Image.open(path), dtype=np.float32)
        raise ValueError(f"unsupported WMGStereo array file suffix: {path.suffix}")

    def _load_camview(self, path: Path) -> dict[str, object]:
        with np.load(path, allow_pickle=True) as payload:
            if {"K", "T", "HW"} <= set(payload.files):
                return {key: payload[key] for key in ("K", "T", "HW")}
            if len(payload.files) == 1:
                candidate = payload[payload.files[0]]
                if candidate.shape == () and isinstance(candidate.item(), dict):
                    camview = candidate.item()
                    if {"K", "T", "HW"} <= set(camview):
                        return camview
        raise ValueError(f"unsupported WMGStereo camview payload: {path}")

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

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(len(shard_names) * self.config.project.train_val_split)
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


class WMGStereoFlyingPipeline(WMGStereoPipeline):
    CATEGORY_NAME = "flying"


class WMGStereoIndoorPipeline(WMGStereoPipeline):
    CATEGORY_NAME = "indoor"


class WMGStereoNaturePipeline(WMGStereoPipeline):
    CATEGORY_NAME = "nature"


__all__ = [
    "WMGStereoArchiveUnit",
    "WMGStereoSourceItem",
    "WMGStereoPipeline",
    "WMGStereoFlyingPipeline",
    "WMGStereoIndoorPipeline",
    "WMGStereoNaturePipeline",
]
