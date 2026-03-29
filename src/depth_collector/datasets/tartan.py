from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Callable, Iterable
import zipfile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

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
class TartanArchiveUnit:
    environment: str
    difficulty: str
    modality: str

    @property
    def filename(self) -> str:
        return f"{self.modality}.zip"


@dataclass(frozen=True)
class TartanSourceItem:
    environment: str
    difficulty: str
    image_relative_path: str
    depth_relative_path: str


class TartanPipeline(DatasetPipeline, ABC):
    """Shared family behavior for Tartan-style dataset pipelines."""

    ENUMERATION_MANIFEST_VERSION = 3
    ALL_SELECTOR_VALUES = {"*", "all"}
    DEFAULT_DIFFICULTIES = ("Easy", "Hard")
    DEFAULT_MODALITIES = ("image_left",)
    DEFAULT_ENVIRONMENTS = ()
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
    DEPTH_SUFFIXES = {".npy", ".npz"}
    REQUIRED_MODALITIES = ("image_left", "depth_left")

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._run_stats["pairing_error_count"] = 0
        self._enumeration_manifest_cache: dict[str, object] | None = None
        self._remote_repo_files_cache: tuple[str, ...] | None = None

    def _get_option_list(self, key: str, default: tuple[str, ...]) -> list[str]:
        value = self.dataset_config.options.get(key, default)
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _uses_all_selector(self, key: str, default: tuple[str, ...] = ()) -> bool:
        value = self.dataset_config.options.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in self.ALL_SELECTOR_VALUES
        if isinstance(value, list):
            return any(str(item).strip().lower() in self.ALL_SELECTOR_VALUES for item in value)
        return False

    def _selected_environments(self) -> list[str]:
        configured = self._configured_environments()
        if configured:
            return configured
        return self._discover_environments()

    def _configured_environments(self) -> list[str]:
        if self._uses_all_selector("environments", self.DEFAULT_ENVIRONMENTS):
            return []
        return self._get_option_list("environments", self.DEFAULT_ENVIRONMENTS)

    def _discover_environments(self) -> list[str]:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            environments = self._discover_environments_from_root(Path(str(local_archive_root)))
            if environments:
                return environments
        if self.paths.raw.exists():
            environments = self._discover_environments_from_root(self.paths.raw)
            if environments:
                return environments
        return self._discover_remote_environments()

    def _discover_environments_from_root(self, root: Path) -> list[str]:
        if not root.exists():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))

    def _discover_remote_environments(self) -> list[str]:
        environments: set[str] = set()
        for relative_path in self._iter_remote_relative_repo_paths():
            parts = Path(relative_path).parts
            if self._looks_like_remote_archive_path(parts):
                environments.add(parts[0])
        return sorted(environments)

    def _looks_like_remote_archive_path(self, parts: tuple[str, ...]) -> bool:
        if len(parts) < 3:
            return False
        filename = parts[-1]
        if not filename.endswith(".zip"):
            return False
        return True

    def _iter_remote_relative_repo_paths(self) -> Iterable[str]:
        prefix = str(self.dataset_config.options.get("hf_path_prefix", "")).strip("/")
        prefix_parts = tuple(part for part in Path(prefix).parts if part)
        for repo_path in self._remote_repo_files():
            repo_parts = Path(repo_path).parts
            if prefix_parts:
                if repo_parts[: len(prefix_parts)] != prefix_parts:
                    continue
                repo_parts = repo_parts[len(prefix_parts) :]
            if not repo_parts:
                continue
            yield str(Path(*repo_parts))

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self._list_hf_files(self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _list_hf_files(self, repo_id: str) -> list[str]:
        return self.hf_list_repo_files(repo_id=repo_id, repo_type="dataset")

    def _selected_environment_count(self) -> int:
        configured = self.dataset_config.options.get("environment_count")
        environments = self._selected_environments()
        if configured is None:
            return len(environments)
        count = int(configured)
        if count < 1:
            raise ValueError("environment_count must be at least 1")
        return min(count, len(environments))

    def _selected_difficulties(self) -> list[str]:
        return self._get_option_list("difficulties", self.DEFAULT_DIFFICULTIES)

    def _selected_modalities(self) -> list[str]:
        configured = self._get_option_list("modalities", self.DEFAULT_MODALITIES)
        selected: list[str] = []
        for modality in (*configured, *self.REQUIRED_MODALITIES):
            if modality not in selected:
                selected.append(modality)
        return selected

    def _selected_group_keys(self) -> list[tuple[str, str]]:
        groups: list[tuple[str, str]] = []
        for environment in self._selected_environments():
            for difficulty in self._selected_difficulties():
                groups.append((environment, difficulty))
        return self._limit_group_keys(groups)

    def _limit_group_keys(self, groups: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
        return groups[: self._selected_environment_count()]

    def enumerate_download_units(self) -> Iterable[object]:
        for group_key in self._selected_group_keys():
            yield from self._iter_group_download_units(group_key)

    def _iter_group_download_units(self, group_key: tuple[object, ...]) -> list[object]:
        environment, difficulty = group_key
        units: list[TartanArchiveUnit] = []
        for modality in self._selected_modalities():
            units.append(
                TartanArchiveUnit(
                    environment=str(environment),
                    difficulty=str(difficulty),
                    modality=modality,
                )
            )
        return units

    def download_unit(self, unit: TartanArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = self._download_archive_from_hub(unit)
        downloaded_path = Path(downloaded_path)
        if downloaded_path.resolve() != archive_path.resolve():
            shutil.copy2(downloaded_path, archive_path)

    def enumerate_extraction_units(self) -> Iterable[TartanArchiveUnit]:
        return self.enumerate_download_units()

    def extract_unit(self, unit: TartanArchiveUnit) -> None:
        archive_path = self._archive_path(unit)
        if not archive_path.exists():
            raise FileNotFoundError(f"missing archive for extraction: {archive_path}")
        extracted_dir = self._extracted_dir(unit)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted_dir)

    def enumerate_source_items(self) -> Iterable[object]:
        for group_key in self._selected_group_keys():
            image_dir = self.get_group_image_dir(group_key)
            depth_dir = self.get_group_depth_dir(group_key)
            group_id = self.get_group_id(group_key)
            if not image_dir.exists():
                self._record_pairing_error(group_id, f"missing extracted image directory: {image_dir}")
                continue
            if not depth_dir.exists():
                self._record_pairing_error(group_id, f"missing extracted depth directory: {depth_dir}")
                continue
            cached_group = self._load_cached_group_manifest(group_id, image_dir, depth_dir)
            if cached_group is None:
                items, pairing_errors = self._scan_group_source_items(
                    group_key=group_key,
                    group_id=group_id,
                    image_dir=image_dir,
                    depth_dir=depth_dir,
                )
                self._store_cached_group_manifest(group_id, image_dir, depth_dir, items, pairing_errors)
            else:
                items, pairing_errors = cached_group

            for item_id, error_message in pairing_errors:
                self._record_pairing_error(item_id, error_message)
            yield from self._iter_cached_manifest_progress(items, group_id)

    def _iter_enumeration_progress(self, image_paths: list[Path], group_id: str) -> Iterable[Path]:
        if not image_paths or not sys.stdout.isatty():
            return image_paths
        return tqdm(
            image_paths,
            desc=f"{group_id} enumerate",
            unit="frame",
            leave=False,
        )

    def _iter_cached_manifest_progress(
        self,
        items: list[TartanSourceItem],
        group_id: str,
    ) -> Iterable[TartanSourceItem]:
        if not items or not sys.stdout.isatty():
            return items
        return self._iter_logged_progress(
            items,
            label=f"{group_id} cached replay",
            unit="frame",
            every_items=5000,
            every_seconds=2.0,
        )

    def _scan_group_source_items(
        self,
        group_key: tuple[object, ...],
        group_id: str,
        image_dir: Path,
        depth_dir: Path,
    ) -> tuple[list[object], list[tuple[str, str]]]:
        environment, difficulty = group_key
        return self._scan_paired_group_source_items(
            group_id=group_id,
            image_dir=image_dir,
            depth_dir=depth_dir,
            image_key_fn=lambda relative_path: self._paired_relative_key(
                relative_path,
                environment=str(environment),
                difficulty=str(difficulty),
            ),
            depth_key_fn=lambda relative_path: self._paired_relative_key(
                relative_path,
                environment=str(environment),
                difficulty=str(difficulty),
            ),
            missing_pair_message_fn=lambda image_relative_path: (
                f"missing paired depth_left file for image_left frame: {image_relative_path}"
            ),
            item_factory=lambda image_relative_path, depth_relative_path: self.build_group_source_item(
                group_key,
                image_relative_path,
                depth_relative_path,
            ),
        )

    def _scan_paired_group_source_items(
        self,
        group_id: str,
        image_dir: Path,
        depth_dir: Path,
        image_key_fn: Callable[[str], str],
        depth_key_fn: Callable[[str], str],
        missing_pair_message_fn: Callable[[str], str],
        item_factory: Callable[[str, str], object],
    ) -> tuple[list[object], list[tuple[str, str]]]:
        depth_paths_by_key: dict[str, str] = {}
        for path in sorted(depth_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.DEPTH_SUFFIXES:
                continue
            relative_path = str(path.relative_to(depth_dir))
            depth_paths_by_key[depth_key_fn(relative_path)] = relative_path

        items: list[object] = []
        pairing_errors: list[tuple[str, str]] = []
        image_paths = [
            path
            for path in sorted(image_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        ]
        for path in self._iter_enumeration_progress(image_paths, group_id):
            image_relative_path = str(path.relative_to(image_dir))
            depth_relative_path = depth_paths_by_key.get(image_key_fn(image_relative_path))
            if depth_relative_path is None:
                pairing_errors.append(
                    (
                        f"{group_id}/{image_relative_path}",
                        missing_pair_message_fn(image_relative_path),
                    )
                )
                continue
            items.append(item_factory(image_relative_path, depth_relative_path))
        return items, pairing_errors

    def _enumeration_manifest_path(self) -> Path:
        return self.paths.state / "enumeration_manifest.json"

    def clear_enumeration_manifest_cache(self) -> None:
        self._enumeration_manifest_cache = None
        manifest_path = self._enumeration_manifest_path()
        if manifest_path.exists():
            manifest_path.unlink()

    def _load_enumeration_manifest_cache(self) -> dict[str, object]:
        if self._enumeration_manifest_cache is not None:
            return self._enumeration_manifest_cache
        manifest_path = self._enumeration_manifest_path()
        if manifest_path.exists():
            loaded = json.loads(manifest_path.read_text())
            if loaded.get("version") == self.ENUMERATION_MANIFEST_VERSION:
                self._enumeration_manifest_cache = loaded
            else:
                self._enumeration_manifest_cache = {"version": self.ENUMERATION_MANIFEST_VERSION, "groups": {}}
        else:
            self._enumeration_manifest_cache = {"version": self.ENUMERATION_MANIFEST_VERSION, "groups": {}}
        return self._enumeration_manifest_cache

    def _save_enumeration_manifest_cache(self) -> None:
        manifest_path = self._enumeration_manifest_path()
        partial_path = manifest_path.with_name(f"{manifest_path.name}.partial")
        partial_path.write_text(json.dumps(self._load_enumeration_manifest_cache(), indent=2, sort_keys=True))
        partial_path.replace(manifest_path)

    def _group_manifest_fingerprint(self, image_dir: Path, depth_dir: Path) -> dict[str, int]:
        return {
            "image_dir_mtime_ns": image_dir.stat().st_mtime_ns,
            "depth_dir_mtime_ns": depth_dir.stat().st_mtime_ns,
        }

    def _load_cached_group_manifest(
        self,
        group_id: str,
        image_dir: Path,
        depth_dir: Path,
    ) -> tuple[list[object], list[tuple[str, str]]] | None:
        manifest = self._load_enumeration_manifest_cache()
        groups = manifest.get("groups", {})
        if not isinstance(groups, dict):
            return None
        group_payload = groups.get(group_id)
        if not isinstance(group_payload, dict):
            return None
        if group_payload.get("fingerprint") != self._group_manifest_fingerprint(image_dir, depth_dir):
            return None
        items_payload = group_payload.get("items", [])
        pairing_errors_payload = group_payload.get("pairing_errors", [])
        items: list[object] = []
        for payload in [payload for payload in items_payload if isinstance(payload, dict)]:
            items.append(self._deserialize_manifest_item(payload))
        pairing_errors: list[tuple[str, str]] = []
        for payload in pairing_errors_payload:
            if not isinstance(payload, dict):
                continue
            pairing_errors.append((str(payload["item_id"]), str(payload["error_message"])))
        return items, pairing_errors

    def _store_cached_group_manifest(
        self,
        group_id: str,
        image_dir: Path,
        depth_dir: Path,
        items: list[object],
        pairing_errors: list[tuple[str, str]],
    ) -> None:
        manifest = self._load_enumeration_manifest_cache()
        groups = manifest.setdefault("groups", {})
        assert isinstance(groups, dict)
        groups[group_id] = {
            "fingerprint": self._group_manifest_fingerprint(image_dir, depth_dir),
            "items": [self._serialize_manifest_item(item) for item in items],
            "pairing_errors": [
                {
                    "item_id": item_id,
                    "error_message": error_message,
                }
                for item_id, error_message in pairing_errors
            ],
        }
        self._save_enumeration_manifest_cache()

    def load_source_item(self, item: object) -> dict[str, object]:
        return self._load_source_paths(
            self.get_source_item_image_path(item),
            self.get_source_item_depth_path(item),
        )

    def build_camera_model(self, item: TartanSourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = loaded_item["image"]
        assert isinstance(image, np.ndarray)
        height, width = image.shape[:2]
        return PinholeCameraModel(
            width=width,
            height=height,
            fx=320.0 * (width / 640.0),
            fy=320.0 * (height / 640.0),
            cx=320.0 * (width / 640.0),
            cy=320.0 * (height / 640.0),
        )

    def build_sample(self, item: TartanSourceItem, loaded_item: object, camera_model: PinholeCameraModel) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = loaded_item["image"]
        z_depth = loaded_item["z_depth"]
        assert isinstance(image, np.ndarray)
        assert isinstance(z_depth, np.ndarray)
        if z_depth.shape != image.shape[:2]:
            raise ValueError("Tartan image and depth shapes must match")
        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        sanitized_z_depth = np.nan_to_num(z_depth, nan=self.config.project.max_dist, posinf=self.config.project.max_dist)
        radial_distance = z_depth_to_distance(sanitized_z_depth[..., None], ray_dir).astype(np.float32)
        self._validate_distance_not_equal_to_source_z_depth(radial_distance, sanitized_z_depth)
        distance = clip_distance_to_max_dist(radial_distance, self.config.project.max_dist)
        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance=self._sample_provenance(item),
        )

    def _validate_distance_not_equal_to_source_z_depth(
        self,
        radial_distance: np.ndarray,
        sanitized_z_depth: np.ndarray,
    ) -> None:
        valid_mask = (sanitized_z_depth > 1e-6) & np.isfinite(sanitized_z_depth)
        if not np.any(valid_mask):
            return
        relative_difference = np.abs(radial_distance[..., 0] - sanitized_z_depth) / np.maximum(sanitized_z_depth, 1e-6)
        mean_relative_difference = float(np.mean(relative_difference[valid_mask]))
        if mean_relative_difference < 1e-3:
            raise ValueError("computed distance is suspiciously close to source z-depth")

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
            "pairing_error_count": self._run_stats["pairing_error_count"],
        }
        self._write_json_atomic(self.paths.metadata, metadata)

    def validate_output(self) -> None:
        if not self.paths.metadata.exists():
            raise ValueError("metadata.json was not created")

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, TartanArchiveUnit)
        return f"{unit.environment}/{unit.difficulty}/{unit.modality}"

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, TartanSourceItem)
        return f"{item.environment}/{item.difficulty}/{item.image_relative_path}"

    def _archive_path(self, unit: TartanArchiveUnit) -> Path:
        return self.paths.raw / unit.environment / unit.difficulty / unit.filename

    def _extracted_dir(self, unit: TartanArchiveUnit) -> Path:
        return self.paths.raw / unit.environment / unit.difficulty / unit.modality

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self.enumerate_download_units():
            yield self._archive_path(unit)

    def remove_download_artifact(self, unit: object) -> None:
        archive_path = self._archive_path(unit)
        if archive_path.exists():
            archive_path.unlink()

    def get_download_artifact_path(self, unit: object) -> Path | None:
        return self._archive_path(unit)

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        return self._extracted_dir(unit)

    def is_download_unit_satisfied(self, unit: object) -> bool:
        return self._archive_path(unit).exists()

    def is_extraction_unit_satisfied(self, unit: object) -> bool:
        extracted_dir = self._extracted_dir(unit)
        if not extracted_dir.exists():
            return False
        return any(path.is_file() for path in extracted_dir.rglob("*"))

    def _load_depth_array(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            depth = np.load(path)
        elif path.suffix.lower() == ".npz":
            with np.load(path) as payload:
                first_key = next(iter(payload.files))
                depth = payload[first_key]
        else:
            raise ValueError(f"unsupported Tartan depth format: {path}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError("Tartan depth must decode to a 2D array")
        return depth

    def _paired_relative_key(self, relative_path: str, environment: str, difficulty: str) -> str:
        path = Path(*self._normalized_relative_parts(relative_path, environment=environment, difficulty=difficulty))
        stem = path.stem
        if stem.endswith("_depth"):
            stem = stem[: -len("_depth")]
        return str(path.with_name(stem).with_suffix(""))

    def _normalized_relative_parts(self, relative_path: str, environment: str, difficulty: str) -> tuple[str, ...]:
        parts = list(Path(relative_path).parts)
        if len(parts) >= 2 and parts[0] == environment and parts[1] == difficulty:
            parts = parts[2:]
        parts = [part for part in parts if part not in self._selected_modalities()]
        return tuple(parts)

    def _load_source_paths(self, image_path: Path, depth_path: Path) -> dict[str, object]:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = self._load_depth_array(depth_path)
        return {
            "image_path": image_path,
            "depth_path": depth_path,
            "image": image,
            "z_depth": depth,
        }

    def _serialize_manifest_item(self, item: object) -> dict[str, object]:
        if not is_dataclass(item):
            raise TypeError("Tartan manifest items must be dataclass instances")
        return asdict(item)

    def _deserialize_manifest_item(self, payload: dict[str, object]) -> object:
        return TartanSourceItem(
            environment=str(payload["environment"]),
            difficulty=str(payload["difficulty"]),
            image_relative_path=str(payload["image_relative_path"]),
            depth_relative_path=str(payload["depth_relative_path"]),
        )

    def _sample_provenance(self, item: object) -> dict[str, object]:
        if not is_dataclass(item):
            raise TypeError("Tartan provenance items must be dataclass instances")
        return asdict(item)

    def get_group_id(self, group_key: tuple[object, ...]) -> str:
        environment, difficulty = group_key
        return f"{environment}/{difficulty}"

    def get_group_image_dir(self, group_key: tuple[object, ...]) -> Path:
        environment, difficulty = group_key
        return self._extracted_dir(
            TartanArchiveUnit(
                environment=str(environment),
                difficulty=str(difficulty),
                modality="image_left",
            )
        )

    def get_group_depth_dir(self, group_key: tuple[object, ...]) -> Path:
        environment, difficulty = group_key
        return self._extracted_dir(
            TartanArchiveUnit(
                environment=str(environment),
                difficulty=str(difficulty),
                modality="depth_left",
            )
        )

    def build_group_source_item(
        self,
        group_key: tuple[object, ...],
        image_relative_path: str,
        depth_relative_path: str,
    ) -> object:
        environment, difficulty = group_key
        return TartanSourceItem(
            environment=str(environment),
            difficulty=str(difficulty),
            image_relative_path=image_relative_path,
            depth_relative_path=depth_relative_path,
        )

    def get_source_item_image_path(self, item: object) -> Path:
        assert isinstance(item, TartanSourceItem)
        return self._extracted_dir(
            TartanArchiveUnit(
                environment=item.environment,
                difficulty=item.difficulty,
                modality="image_left",
            )
        ) / item.image_relative_path

    def get_source_item_depth_path(self, item: object) -> Path:
        assert isinstance(item, TartanSourceItem)
        return self._extracted_dir(
            TartanArchiveUnit(
                environment=item.environment,
                difficulty=item.difficulty,
                modality="depth_left",
            )
        ) / item.depth_relative_path

    def _download_archive_from_hub(self, unit: TartanArchiveUnit) -> Path:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            local_path = Path(str(local_archive_root)) / self._hub_repo_filename(unit)
            if not local_path.exists():
                raise FileNotFoundError(f"missing local archive source: {local_path}")
            return local_path

        return self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            filename=self._hub_repo_filename(unit),
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
        )

    def _hub_repo_filename(self, unit: TartanArchiveUnit) -> str:
        prefix = str(self.dataset_config.options.get("hf_path_prefix", "")).strip("/")
        parts = [prefix, unit.environment, unit.difficulty, unit.filename]
        return "/".join(part for part in parts if part)

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(len(shard_names) * self.config.project.train_val_split)
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]

    def _record_pairing_error(self, item_id: str, error_message: str) -> None:
        self._run_stats["pairing_error_count"] += 1
        self.record_error(stage="enumeration", item_id=item_id, error_message=error_message)
