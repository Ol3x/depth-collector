from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import tarfile
from typing import Iterable

import numpy as np

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import PinholeCameraModel, clip_distance_to_max_dist, generate_pinhole_rays, z_depth_to_distance
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class NYUDepthV2ArchiveUnit:
    repo_path: str

    @property
    def shard_name(self) -> str:
        return Path(self.repo_path).name


@dataclass(frozen=True)
class NYUDepthV2SourceItem:
    shard_name: str
    extracted_root_name: str
    relative_path: str

    @property
    def relative_stem(self) -> str:
        return str(Path(self.relative_path).with_suffix(""))


class NYUDepthV2Pipeline(DatasetPipeline):
    """Archive-backed NYU Depth V2 pipeline over HF tar shards of HDF5 samples."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    DEFAULT_CAMERA_INTRINSICS = {
        "width": 640.0,
        "height": 480.0,
        "fx": 518.8579,
        "fy": 519.4696,
        "cx": 325.5824,
        "cy": 253.7362,
    }
    DEFAULT_IMAGE_DATASET_KEYS = ("rgb", "image", "images")
    DEFAULT_DEPTH_DATASET_KEYS = ("depth", "depths")

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None

    def _data_dir(self) -> str:
        return str(self.dataset_config.options.get("data_dir", "data")).strip().strip("/")

    def _downloads_root(self) -> Path:
        return self.paths.raw / "_downloads"

    def _extract_root(self) -> Path:
        return self.paths.raw / "_extracted"

    def _camera_intrinsics(self) -> dict[str, float]:
        configured = self.dataset_config.options.get("camera_intrinsics", {})
        assert isinstance(configured, dict)
        values = dict(self.DEFAULT_CAMERA_INTRINSICS)
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if key in configured:
                values[key] = float(configured[key])
        return values

    def _configured_splits(self) -> list[str]:
        configured = self.dataset_config.options.get("splits", ["val"])
        if isinstance(configured, str):
            normalized = configured.strip().lower()
            if normalized in self.ALL_SELECTOR_VALUES or not normalized:
                return []
            return [configured]
        splits = [
            str(split_name)
            for split_name in configured
            if str(split_name).strip().lower() not in self.ALL_SELECTOR_VALUES and str(split_name).strip()
        ]
        return splits

    def _configured_shards(self) -> list[str]:
        configured = self.dataset_config.options.get("shards")
        if isinstance(configured, str):
            normalized = configured.strip().lower()
            if normalized in self.ALL_SELECTOR_VALUES or not normalized:
                return []
            return [configured]
        if isinstance(configured, list):
            return [
                str(shard_name)
                for shard_name in configured
                if str(shard_name).strip().lower() not in self.ALL_SELECTOR_VALUES and str(shard_name).strip()
            ]
        return []

    def _repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(self.dataset_config.hf_dataset_id, repo_type="dataset"))
        return self._remote_repo_files_cache

    def _discover_local_repo_paths(self) -> list[str]:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if not local_archive_root:
            return []
        local_root = Path(str(local_archive_root)) / self._data_dir()
        if not local_root.exists():
            return []
        return [path.relative_to(Path(str(local_archive_root))).as_posix() for path in sorted(local_root.rglob("*.tar"))]

    def _discover_extracted_repo_paths(self) -> list[str]:
        return [f"{self._data_dir()}/{path.name}.tar" for path in self._discovered_local_extract_roots()]

    def _discover_remote_repo_paths(self) -> list[str]:
        data_prefix = f"{self._data_dir()}/"
        return sorted(
            path
            for path in self._repo_files()
            if path.startswith(data_prefix) and Path(path).suffix.lower() == ".tar"
        )

    def _selected_repo_paths(self) -> list[str]:
        configured_shards = self._configured_shards()
        if configured_shards:
            paths = [self._normalize_repo_path(shard_name) for shard_name in configured_shards]
        else:
            discovered_paths = (
                self._discover_local_repo_paths()
                or self._discover_extracted_repo_paths()
                or self._discover_remote_repo_paths()
            )
            configured_splits = self._configured_splits()
            if configured_splits:
                split_prefixes = tuple(f"{self._data_dir()}/{split_name}-" for split_name in configured_splits)
                paths = [path for path in discovered_paths if path.startswith(split_prefixes)]
            else:
                paths = discovered_paths
        if not paths:
            raise FileNotFoundError(
                "NYU Depth V2 requires configured `datasets.nyu_depth_v2.shards`, "
                "local tar shards, or remote tar shards under the configured data directory"
            )
        return [str(repo_path) for repo_path in self.apply_dataset_selection(paths)]

    def _normalize_repo_path(self, shard_name: str) -> str:
        normalized = shard_name.strip().strip("/")
        if normalized.startswith(f"{self._data_dir()}/"):
            return normalized
        return f"{self._data_dir()}/{normalized}"

    def _archive_path(self, unit: NYUDepthV2ArchiveUnit) -> Path:
        return self._downloads_root() / unit.repo_path

    def _unit_extract_root(self, unit: NYUDepthV2ArchiveUnit) -> Path:
        return self._extract_root() / Path(unit.shard_name).stem

    def _source_item_path(self, item: NYUDepthV2SourceItem) -> Path:
        return self._extract_root() / item.extracted_root_name / item.relative_path

    def _discovered_local_extract_roots(self) -> list[Path]:
        extract_root = self._extract_root()
        if not extract_root.exists():
            return []
        return sorted(path for path in extract_root.iterdir() if path.is_dir())

    def enumerate_download_units(self) -> Iterable[NYUDepthV2ArchiveUnit]:
        for repo_path in self._selected_repo_paths():
            yield NYUDepthV2ArchiveUnit(repo_path=repo_path)

    def download_unit(self, unit: NYUDepthV2ArchiveUnit) -> None:
        target_path = self._archive_path(unit)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_path = Path(str(local_archive_root)) / unit.repo_path
            if not source_path.exists():
                raise FileNotFoundError(f"missing local NYU Depth V2 archive source: {source_path}")
            shutil.copy2(source_path, target_path)
            return

        downloaded_path = self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            filename=unit.repo_path,
            revision=self.dataset_config.revision,
        )
        if downloaded_path.resolve() != target_path.resolve():
            shutil.copy2(downloaded_path, target_path)

    def enumerate_extraction_units(self) -> Iterable[NYUDepthV2ArchiveUnit]:
        return self.enumerate_download_units()

    def extract_unit(self, unit: NYUDepthV2ArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        if not archive_path.exists():
            raise FileNotFoundError(f"missing NYU Depth V2 shard for extraction: {archive_path}")
        extract_root = self._unit_extract_root(unit)
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(extract_root, filter="data")

    def enumerate_source_items(self) -> Iterable[NYUDepthV2SourceItem]:
        extract_roots = self._discovered_local_extract_roots()
        if not extract_roots:
            extract_roots = [self._unit_extract_root(unit) for unit in self.enumerate_extraction_units()]
        for extract_root in extract_roots:
            if not extract_root.exists():
                self.record_error("enumeration", extract_root.name, f"missing extracted NYU Depth V2 root: {extract_root}")
                continue
            for h5_path in sorted(extract_root.rglob("*.h5")):
                relative_path = h5_path.relative_to(extract_root).as_posix()
                yield NYUDepthV2SourceItem(
                    shard_name=f"{extract_root.name}.tar",
                    extracted_root_name=extract_root.name,
                    relative_path=relative_path,
                )

    def load_source_item(self, item: NYUDepthV2SourceItem) -> dict[str, object]:
        path = self._source_item_path(item)
        image_array, depth_array = self._load_hdf5_arrays(path)
        image = self._normalize_image_array(image_array)
        depth = np.asarray(depth_array, dtype=np.float32)
        return {
            "image": image,
            "depth": depth,
        }

    def build_camera_model(self, item: NYUDepthV2SourceItem, loaded_item: object) -> PinholeCameraModel:
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
        item: NYUDepthV2SourceItem,
        loaded_item: object,
        camera_model: PinholeCameraModel,
    ) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth = np.asarray(loaded_item["depth"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError("NYU Depth V2 depth must decode to a 2D array")
        if image.shape[:2] != depth.shape:
            raise ValueError("NYU Depth V2 image and depth shapes must match")

        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        depth_semantics = str(self.dataset_config.options.get("depth_semantics", "z_depth")).strip().lower()
        if depth_semantics == "distance":
            distance = depth
        elif depth_semantics == "z_depth":
            distance = z_depth_to_distance(depth[..., None], ray_dir)[..., 0]
        else:
            raise ValueError("NYU Depth V2 depth_semantics must be 'distance' or 'z_depth'")

        valid_mask = np.isfinite(distance) & (distance > 0.0)
        distance = distance[..., None]
        distance[..., 0][~valid_mask] = self.config.project.max_dist
        distance = clip_distance_to_max_dist(distance, self.config.project.max_dist)

        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance.astype(np.float32),
            ray_dir=ray_dir,
            provenance={
                "shard_name": item.shard_name,
                "relative_path": item.relative_path,
                "depth_semantics": depth_semantics,
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
        assert isinstance(unit, NYUDepthV2ArchiveUnit)
        return unit.repo_path

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, NYUDepthV2SourceItem)
        return f"{item.shard_name}:{item.relative_stem}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self.enumerate_download_units():
            yield self._archive_path(unit)

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, NYUDepthV2ArchiveUnit)
        return self._archive_path(unit)

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, NYUDepthV2ArchiveUnit)
        return self._unit_extract_root(unit)

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, NYUDepthV2ArchiveUnit)
        archive_path = self._archive_path(unit)
        if archive_path.exists():
            archive_path.unlink()

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, NYUDepthV2ArchiveUnit)
        archive_path = self._archive_path(unit)
        return archive_path.exists() and archive_path.is_file()

    def _load_hdf5_arrays(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("NYU Depth V2 processing requires h5py for HDF5 samples") from exc
        with h5py.File(path, "r") as handle:
            image_key = self._resolve_dataset_key(handle, self.DEFAULT_IMAGE_DATASET_KEYS, "image_dataset_key", path)
            depth_key = self._resolve_dataset_key(handle, self.DEFAULT_DEPTH_DATASET_KEYS, "depth_dataset_key", path)
            return np.asarray(handle[image_key]), np.asarray(handle[depth_key], dtype=np.float32)

    def _resolve_dataset_key(
        self,
        handle: object,
        candidate_keys: tuple[str, ...],
        override_option: str,
        path: Path,
    ) -> str:
        override = self.dataset_config.options.get(override_option)
        if isinstance(override, str) and override:
            if override in handle:  # type: ignore[operator]
                return override
            raise KeyError(f"NYU Depth V2 sample {path} does not contain configured dataset key {override!r}")
        for key in candidate_keys:
            if key in handle:  # type: ignore[operator]
                return key
        raise KeyError(f"NYU Depth V2 sample {path} is missing expected datasets {candidate_keys}")

    def _normalize_image_array(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError("NYU Depth V2 RGB image must decode to a 3D array")
        if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] != 3:
            raise ValueError("NYU Depth V2 RGB image must have 3 channels")
        image = np.asarray(image, dtype=np.float32)
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(math.floor(len(shard_names) * self.config.project.train_val_split))
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "NYUDepthV2ArchiveUnit",
    "NYUDepthV2Pipeline",
    "NYUDepthV2SourceItem",
]
