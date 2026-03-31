from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
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
class VirtualKITTI2ArchiveUnit:
    archive_name: str


@dataclass(frozen=True)
class VirtualKITTI2SequenceUnit:
    sequence_name: str


@dataclass(frozen=True)
class VirtualKITTI2SourceItem:
    sequence_name: str
    frame_id: str
    frame_index: int
    array_index: int
    image_relative_path: str
    depth_relative_path: str


class VirtualKITTI2Pipeline(DatasetPipeline):
    """Archive-backed VKITTI2 pipeline using RGB, dense metric depth, and per-frame intrinsics."""

    ALL_SELECTOR_VALUES = {"*", "all"}

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None
        self._camera_arrays_cache: dict[str, dict[str, np.ndarray]] = {}
        self._frame_index_map_cache: dict[str, dict[str, int]] = {}
        self._scene_info_cache: dict[str, dict[str, object]] = {}

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(repo_id=self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _archive_filename(self) -> str:
        return str(self.dataset_config.options.get("archive_filename", "vkitti2_vlbm.tar.gz"))

    def _sequence_option(self) -> list[str]:
        configured = self.dataset_config.options.get("sequences", "*")
        if isinstance(configured, str):
            if configured.strip().lower() in self.ALL_SELECTOR_VALUES:
                return []
            return [configured]
        return [
            str(sequence_name)
            for sequence_name in configured
            if str(sequence_name).strip().lower() not in self.ALL_SELECTOR_VALUES
        ]

    def _hf_path_prefix(self) -> str:
        configured = str(self.dataset_config.options.get("hf_path_prefix", "vkitti2_vlbm")).strip().strip("/")
        return configured

    def _raw_dataset_root(self) -> Path:
        prefix = self._hf_path_prefix()
        if not prefix:
            return self.paths.raw
        prefixed_root = self.paths.raw / prefix
        if prefixed_root.exists():
            return prefixed_root
        return self.paths.raw

    def _expected_dataset_root(self) -> Path:
        prefix = self._hf_path_prefix()
        if not prefix:
            return self.paths.raw
        return self.paths.raw / prefix

    def _sequence_root(self, sequence_name: str) -> Path:
        return self._raw_dataset_root() / sequence_name

    def _archive_path(self) -> Path:
        return self.paths.raw / self._archive_filename()

    def _discover_sequence_names_from_root(self, root: Path) -> list[str]:
        if not root.exists():
            return []
        sequence_names: list[str] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if not (path / "rgbs").exists():
                continue
            if not (path / "depths").exists():
                continue
            if not (path / "intrinsics.npy").exists():
                continue
            if not (path / "extrinsics.npy").exists():
                continue
            sequence_names.append(path.name)
        return sequence_names

    def _selected_sequence_names(self) -> list[str]:
        configured = self._sequence_option()
        if configured:
            sequence_names = configured
        else:
            local_archive_root = self.dataset_config.options.get("local_archive_root")
            if local_archive_root:
                sequence_names = self._discover_sequence_names_from_root(Path(str(local_archive_root)))
            else:
                sequence_names = []
            if not sequence_names:
                sequence_names = self._discover_sequence_names_from_root(self._raw_dataset_root())
        return [str(sequence_name) for sequence_name in self.apply_dataset_selection(sequence_names)]

    def enumerate_download_units(self) -> Iterable[VirtualKITTI2ArchiveUnit]:
        yield VirtualKITTI2ArchiveUnit(archive_name=self._archive_filename())

    def download_unit(self, unit: VirtualKITTI2ArchiveUnit) -> None:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        archive_path = self._archive_path()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dataset_selection() == self.MINIMUM_READABLE_SELECTION:
            if local_archive_root:
                source_path = Path(str(local_archive_root)) / unit.archive_name
                if not source_path.exists():
                    raise FileNotFoundError(f"missing local VKITTI2 archive source: {source_path}")
                self._write_minimum_readable_archive_from_local_source(source_path, archive_path)
                return
            self._write_minimum_readable_archive_from_remote_source(unit.archive_name, archive_path)
            return
        if local_archive_root:
            source_path = Path(str(local_archive_root)) / unit.archive_name
            if not source_path.exists():
                raise FileNotFoundError(f"missing local VKITTI2 archive source: {source_path}")
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

    def _write_minimum_readable_archive_from_local_source(self, source_path: Path, target_path: Path) -> None:
        with source_path.open("rb") as handle:
            with tarfile.open(fileobj=handle, mode="r|*") as archive:
                self._write_minimum_readable_archive_from_stream(archive, target_path)

    def _write_minimum_readable_archive_from_remote_source(self, archive_name: str, target_path: Path) -> None:
        with self.hf_open_remote_file(
            repo_id=self.dataset_config.hf_dataset_id,
            filename=archive_name,
            repo_type="dataset",
            revision=self.dataset_config.revision,
        ) as handle:
            with tarfile.open(fileobj=handle, mode="r|*") as archive:
                self._write_minimum_readable_archive_from_stream(archive, target_path)

    def _write_minimum_readable_archive_from_stream(self, archive: tarfile.TarFile, target_path: Path) -> None:
        sequence_payloads: dict[str, dict[str, object]] = {}
        for member in archive:
            if not member.isfile():
                continue
            descriptor = self._classify_archive_member(member.name)
            if descriptor is None:
                continue
            sequence_name, kind, key = descriptor
            member_file = archive.extractfile(member)
            if member_file is None:
                continue
            payload = sequence_payloads.setdefault(
                sequence_name,
                {
                    "images": {},
                    "depths": {},
                    "intrinsics": None,
                    "extrinsics": None,
                    "scene_info": None,
                    "prefix": self._hf_path_prefix(),
                },
            )
            if kind == "image":
                payload["images"][key] = (member.name, member_file.read())
            elif kind == "depth":
                payload["depths"][key] = (member.name, member_file.read())
            else:
                payload[kind] = (member.name, member_file.read())
            subset_written = self._maybe_write_minimum_readable_archive(sequence_name, payload, target_path)
            if subset_written:
                return
        raise ValueError(
            f"{self.dataset_name} could not find a readable sample inside {target_path.name}: "
            "no archive member set contained one RGB, one depth, intrinsics, extrinsics, and scene info"
        )

    def _classify_archive_member(self, member_name: str) -> tuple[str, str, str | None] | None:
        path = Path(member_name)
        parts = path.parts
        prefix = self._hf_path_prefix()
        if prefix:
            if len(parts) < 3 or parts[0] != prefix:
                return None
            parts = parts[1:]
        if len(parts) < 2:
            return None
        sequence_name = parts[0]
        if len(parts) >= 3 and parts[1] == "rgbs" and path.name.startswith("rgb_") and path.suffix.lower() == ".jpg":
            return sequence_name, "image", path.stem.removeprefix("rgb_")
        if len(parts) >= 3 and parts[1] == "depths" and path.name.startswith("depth_") and path.suffix.lower() == ".npz":
            return sequence_name, "depth", path.stem.removeprefix("depth_")
        if len(parts) == 2 and path.name == "intrinsics.npy":
            return sequence_name, "intrinsics", None
        if len(parts) == 2 and path.name == "extrinsics.npy":
            return sequence_name, "extrinsics", None
        if len(parts) == 2 and path.name == "scene_info.json":
            return sequence_name, "scene_info", None
        return None

    def _maybe_write_minimum_readable_archive(
        self,
        sequence_name: str,
        payload: dict[str, object],
        target_path: Path,
    ) -> bool:
        images = payload["images"]
        depths = payload["depths"]
        intrinsics_payload = payload["intrinsics"]
        extrinsics_payload = payload["extrinsics"]
        scene_info_payload = payload["scene_info"]
        assert isinstance(images, dict)
        assert isinstance(depths, dict)
        if intrinsics_payload is None or extrinsics_payload is None or scene_info_payload is None:
            return False
        frame_ids = sorted(set(images) & set(depths))
        if not frame_ids:
            return False
        frame_id = frame_ids[0]
        frame_index = int(frame_id)
        intrinsics_name, intrinsics_bytes = intrinsics_payload
        extrinsics_name, extrinsics_bytes = extrinsics_payload
        scene_info_name, scene_info_bytes = scene_info_payload
        intrinsics = np.asarray(np.load(io.BytesIO(intrinsics_bytes)), dtype=np.float32)
        extrinsics = np.asarray(np.load(io.BytesIO(extrinsics_bytes)), dtype=np.float32)
        if frame_index >= intrinsics.shape[0] or frame_index >= extrinsics.shape[0]:
            return False
        subset_intrinsics = intrinsics[frame_index : frame_index + 1]
        subset_extrinsics = extrinsics[frame_index : frame_index + 1]
        image_name, image_bytes = images[frame_id]
        depth_name, depth_bytes = depths[frame_id]
        prefix = str(payload["prefix"]).strip().strip("/")
        frame_index_map = {frame_id: 0}
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target_path, "w:gz") as output_archive:
            self._write_archive_bytes(output_archive, image_name, image_bytes)
            self._write_archive_bytes(output_archive, depth_name, depth_bytes)
            self._write_archive_bytes(output_archive, intrinsics_name, self._numpy_payload(subset_intrinsics))
            self._write_archive_bytes(output_archive, extrinsics_name, self._numpy_payload(subset_extrinsics))
            self._write_archive_bytes(output_archive, scene_info_name, scene_info_bytes)
            frame_index_map_path = "/".join(part for part in (prefix, sequence_name, "frame_index_map.json") if part)
            self._write_archive_bytes(
                output_archive,
                frame_index_map_path,
                json.dumps(frame_index_map, sort_keys=True).encode("utf-8"),
            )
        return True

    def _numpy_payload(self, array: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()

    def _write_archive_bytes(self, archive: tarfile.TarFile, member_name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    def enumerate_extraction_units(self) -> Iterable[VirtualKITTI2ArchiveUnit]:
        return self.enumerate_download_units()

    def extract_unit(self, unit: VirtualKITTI2ArchiveUnit) -> None:
        archive_path = self._archive_path()
        if not archive_path.exists():
            raise FileNotFoundError(f"missing VKITTI2 archive for extraction: {archive_path}")
        self.paths.raw.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(self.paths.raw, filter="data")

    def enumerate_source_items(self) -> Iterable[VirtualKITTI2SourceItem]:
        for sequence_name in self._selected_sequence_names():
            sequence_root = self._sequence_root(sequence_name)
            images_root = sequence_root / "rgbs"
            depths_root = sequence_root / "depths"
            intrinsics_path = sequence_root / "intrinsics.npy"
            extrinsics_path = sequence_root / "extrinsics.npy"
            scene_info_path = sequence_root / "scene_info.json"
            if not images_root.exists():
                self.record_error("enumeration", sequence_name, f"missing VKITTI2 RGB directory: {images_root}")
                continue
            if not depths_root.exists():
                self.record_error("enumeration", sequence_name, f"missing VKITTI2 depth directory: {depths_root}")
                continue
            if not intrinsics_path.exists():
                self.record_error("enumeration", sequence_name, f"missing VKITTI2 intrinsics file: {intrinsics_path}")
                continue
            if not extrinsics_path.exists():
                self.record_error("enumeration", sequence_name, f"missing VKITTI2 extrinsics file: {extrinsics_path}")
                continue
            if not scene_info_path.exists():
                self.record_error("enumeration", sequence_name, f"missing VKITTI2 scene info file: {scene_info_path}")
                continue

            camera_arrays = self._load_camera_arrays(sequence_name)
            frame_index_map = self._load_frame_index_map(sequence_name)
            intrinsics = camera_arrays.get("intrinsics")
            extrinsics = camera_arrays.get("extrinsics")
            if intrinsics is None or intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
                self.record_error("enumeration", sequence_name, "missing or invalid VKITTI2 intrinsics array")
                continue
            if extrinsics is None or extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
                self.record_error("enumeration", sequence_name, "missing or invalid VKITTI2 extrinsics array")
                continue

            depth_by_frame = {
                self._frame_id_from_depth_path(path): path
                for path in sorted(depths_root.glob("depth_*.npz"))
                if path.is_file()
            }
            for image_path in sorted(images_root.glob("rgb_*.jpg")):
                if not image_path.is_file():
                    continue
                frame_id = self._frame_id_from_image_path(image_path)
                depth_path = depth_by_frame.get(frame_id)
                if depth_path is None:
                    self.record_error(
                        "enumeration",
                        f"{sequence_name}/{image_path.name}",
                        f"missing paired VKITTI2 depth file for image: {image_path.name}",
                    )
                    continue
                frame_index = int(frame_id)
                array_index = frame_index_map.get(frame_id, frame_index)
                if array_index >= intrinsics.shape[0] or array_index >= extrinsics.shape[0]:
                    self.record_error(
                        "enumeration",
                        f"{sequence_name}/{frame_id}",
                        f"frame index {array_index} exceeds VKITTI2 annotations length",
                    )
                    continue
                yield VirtualKITTI2SourceItem(
                    sequence_name=sequence_name,
                    frame_id=frame_id,
                    frame_index=frame_index,
                    array_index=array_index,
                    image_relative_path=str(image_path.relative_to(self._raw_dataset_root())),
                    depth_relative_path=str(depth_path.relative_to(self._raw_dataset_root())),
                )

    @staticmethod
    def _frame_id_from_image_path(path: Path) -> str:
        return path.stem.removeprefix("rgb_")

    @staticmethod
    def _frame_id_from_depth_path(path: Path) -> str:
        return path.stem.removeprefix("depth_")

    def _load_camera_arrays(self, sequence_name: str) -> dict[str, np.ndarray]:
        cached = self._camera_arrays_cache.get(sequence_name)
        if cached is not None:
            return cached
        sequence_root = self._sequence_root(sequence_name)
        payload = {
            "intrinsics": np.asarray(np.load(sequence_root / "intrinsics.npy"), dtype=np.float32),
            "extrinsics": np.asarray(np.load(sequence_root / "extrinsics.npy"), dtype=np.float32),
        }
        self._camera_arrays_cache[sequence_name] = payload
        return payload

    def _load_frame_index_map(self, sequence_name: str) -> dict[str, int]:
        cached = self._frame_index_map_cache.get(sequence_name)
        if cached is not None:
            return cached
        path = self._sequence_root(sequence_name) / "frame_index_map.json"
        if not path.exists():
            payload: dict[str, int] = {}
        else:
            raw_payload = json.loads(path.read_text())
            payload = {str(frame_id): int(index) for frame_id, index in raw_payload.items()}
        self._frame_index_map_cache[sequence_name] = payload
        return payload

    def _load_scene_info(self, sequence_name: str) -> dict[str, object]:
        cached = self._scene_info_cache.get(sequence_name)
        if cached is not None:
            return cached
        scene_info_path = self._sequence_root(sequence_name) / "scene_info.json"
        payload = json.loads(scene_info_path.read_text())
        self._scene_info_cache[sequence_name] = payload
        return payload

    def load_source_item(self, item: VirtualKITTI2SourceItem) -> dict[str, object]:
        dataset_root = self._raw_dataset_root()
        image = np.asarray(Image.open(dataset_root / item.image_relative_path).convert("RGB"), dtype=np.float32) / 255.0
        with np.load(dataset_root / item.depth_relative_path) as handle:
            depth = np.asarray(handle["depth"], dtype=np.float32)
        camera_arrays = self._load_camera_arrays(item.sequence_name)
        scene_info = self._load_scene_info(item.sequence_name)
        return {
            "image": image,
            "depth": depth,
            "intrinsics": np.asarray(camera_arrays["intrinsics"][item.array_index], dtype=np.float32),
            "extrinsics": np.asarray(camera_arrays["extrinsics"][item.array_index], dtype=np.float32),
            "scene_info": scene_info,
        }

    def build_camera_model(self, item: VirtualKITTI2SourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        intrinsics = np.asarray(loaded_item["intrinsics"], dtype=np.float32)
        height, width = image.shape[:2]
        if intrinsics.shape != (3, 3):
            raise ValueError("VKITTI2 intrinsics must be a 3x3 matrix")
        return PinholeCameraModel(
            width=width,
            height=height,
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
        )

    def build_sample(self, item: VirtualKITTI2SourceItem, loaded_item: object, camera_model: PinholeCameraModel) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth = np.asarray(loaded_item["depth"], dtype=np.float32)
        extrinsics = np.asarray(loaded_item["extrinsics"], dtype=np.float32)
        scene_info = loaded_item["scene_info"]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError("VKITTI2 depth must decode to a 2D array")
        if image.shape[:2] != depth.shape:
            raise ValueError("VKITTI2 image and depth shapes must match")
        if extrinsics.shape != (4, 4):
            raise ValueError("VKITTI2 extrinsics must be a 4x4 matrix")

        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        depth_semantics = str(self.dataset_config.options.get("depth_semantics", "distance")).strip().lower()
        if depth_semantics == "distance":
            distance = depth
        elif depth_semantics == "z_depth":
            distance = z_depth_to_distance(depth[..., None], ray_dir)[..., 0]
        else:
            raise ValueError("VKITTI2 depth_semantics must be 'distance' or 'z_depth'")

        invalid_mask = ~np.isfinite(distance) | (distance <= 0.0)
        distance = distance[..., None]
        distance[..., 0][invalid_mask] = self.config.project.max_dist
        distance = clip_distance_to_max_dist(distance, self.config.project.max_dist)

        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance.astype(np.float32),
            ray_dir=ray_dir,
            provenance={
                "sequence_name": item.sequence_name,
                "frame_id": item.frame_id,
                "frame_index": item.frame_index,
                "image_relative_path": item.image_relative_path,
                "depth_relative_path": item.depth_relative_path,
                "projection": "pinhole",
                "depth_unit": "meters",
                "camera_axes": {"x": "right", "y": "down", "z": "forward"},
                "extrinsics_w2c": extrinsics.tolist(),
                "scene_info": scene_info,
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
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        return unit.archive_name

    def get_extraction_unit_id(self, unit: object) -> str:
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, VirtualKITTI2SourceItem)
        return f"{item.sequence_name}/{item.frame_id}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        yield self._archive_path()

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        return self._archive_path()

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        return self._raw_dataset_root()

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        archive_path = self._archive_path()
        if archive_path.exists():
            archive_path.unlink()

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        archive_path = self._archive_path()
        return archive_path.exists() and archive_path.is_file()

    def is_extraction_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, VirtualKITTI2ArchiveUnit)
        dataset_root = self._expected_dataset_root()
        if not dataset_root.exists() or not dataset_root.is_dir():
            return False
        return bool(self._discover_sequence_names_from_root(dataset_root))

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(math.floor(len(shard_names) * self.config.project.train_val_split))
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "VirtualKITTI2ArchiveUnit",
    "VirtualKITTI2Pipeline",
    "VirtualKITTI2SequenceUnit",
    "VirtualKITTI2SourceItem",
]
