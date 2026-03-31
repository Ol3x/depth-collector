from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
from PIL import Image

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import PinholeCameraModel, clip_distance_to_max_dist, generate_pinhole_rays, z_depth_to_distance
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class UrbanSynFrameUnit:
    frame_id: str


@dataclass(frozen=True)
class UrbanSynSourceItem:
    frame_id: str
    image_relative_path: str
    depth_relative_path: str
    semantic_relative_path: str | None = None


class UrbanSynPipeline(DatasetPipeline):
    """Perspective UrbanSyn pipeline with per-frame HF acquisition."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    DEFAULT_CAMERA_INTRINSICS = {
        "width": 2048.0,
        "height": 1024.0,
        "fx": 1730.207210073563,
        "fy": 1730.207210073563,
        "cx": 1024.0,
        "cy": 512.0,
    }

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(repo_id=self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _frames_option(self) -> list[str]:
        configured = self.dataset_config.options.get("frames", "*")
        if isinstance(configured, str):
            if configured.strip().lower() in self.ALL_SELECTOR_VALUES:
                return []
            return [configured]
        return [
            str(frame_id)
            for frame_id in configured
            if str(frame_id).strip().lower() not in self.ALL_SELECTOR_VALUES
        ]

    def _discover_frame_ids_from_root(self, root: Path) -> list[str]:
        if not root.exists():
            return []
        rgb_ids = {path.stem.removeprefix("rgb_") for path in (root / "rgb").glob("rgb_*.png")}
        depth_ids = {path.stem.removeprefix("depth_") for path in (root / "depth").glob("depth_*.exr")}
        frame_ids = rgb_ids & depth_ids
        if self._use_semantic_masks():
            semantic_ids = {path.stem.removeprefix("ss_") for path in (root / "ss").glob("ss_*.png")}
            frame_ids &= semantic_ids
        return sorted(frame_ids)

    def _discover_frame_ids_from_remote(self) -> list[str]:
        rgb_ids: set[str] = set()
        depth_ids: set[str] = set()
        semantic_ids: set[str] = set()
        for repo_path in self._remote_repo_files():
            path = Path(repo_path)
            parts = path.parts
            if len(parts) != 2:
                continue
            parent, filename = parts
            stem = Path(filename).stem
            if parent == "rgb" and path.suffix.lower() == ".png" and stem.startswith("rgb_"):
                rgb_ids.add(stem.removeprefix("rgb_"))
            elif parent == "depth" and path.suffix.lower() == ".exr" and stem.startswith("depth_"):
                depth_ids.add(stem.removeprefix("depth_"))
            elif parent == "ss" and path.suffix.lower() == ".png" and stem.startswith("ss_"):
                semantic_ids.add(stem.removeprefix("ss_"))
        frame_ids = rgb_ids & depth_ids
        if self._use_semantic_masks():
            frame_ids &= semantic_ids
        return sorted(frame_ids)

    def _selected_frame_ids(self) -> list[str]:
        configured = self._frames_option()
        if configured:
            frame_ids = configured
        else:
            local_archive_root = self.dataset_config.options.get("local_archive_root")
            if local_archive_root:
                frame_ids = self._discover_frame_ids_from_root(Path(str(local_archive_root)))
            else:
                frame_ids = []
            if not frame_ids:
                frame_ids = self._discover_frame_ids_from_root(self.paths.raw)
            if not frame_ids:
                frame_ids = self._discover_frame_ids_from_remote()
        return [str(frame_id) for frame_id in self.apply_dataset_selection(frame_ids)]

    def _use_semantic_masks(self) -> bool:
        return bool(self.dataset_config.options.get("use_semantic_masks", True))

    def _rgb_relative_path(self, frame_id: str) -> str:
        return f"rgb/rgb_{frame_id}.png"

    def _depth_relative_path(self, frame_id: str) -> str:
        return f"depth/depth_{frame_id}.exr"

    def _semantic_relative_path(self, frame_id: str) -> str:
        return f"ss/ss_{frame_id}.png"

    def enumerate_download_units(self) -> Iterable[UrbanSynFrameUnit]:
        for frame_id in self._selected_frame_ids():
            yield UrbanSynFrameUnit(frame_id=frame_id)

    def download_unit(self, unit: UrbanSynFrameUnit) -> None:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_root = Path(str(local_archive_root))
            self._copy_local_file(source_root / self._rgb_relative_path(unit.frame_id), self.paths.raw / self._rgb_relative_path(unit.frame_id))
            self._copy_local_file(
                source_root / self._depth_relative_path(unit.frame_id),
                self.paths.raw / self._depth_relative_path(unit.frame_id),
            )
            if self._use_semantic_masks():
                self._copy_local_file(
                    source_root / self._semantic_relative_path(unit.frame_id),
                    self.paths.raw / self._semantic_relative_path(unit.frame_id),
                )
            return

        self._download_repo_file(self._rgb_relative_path(unit.frame_id))
        self._download_repo_file(self._depth_relative_path(unit.frame_id))
        if self._use_semantic_masks():
            self._download_repo_file(self._semantic_relative_path(unit.frame_id))

    def _copy_local_file(self, source_path: Path, target_path: Path) -> None:
        if not source_path.exists():
            raise FileNotFoundError(f"missing local UrbanSyn source file: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    def _download_repo_file(self, relative_path: str) -> Path:
        return self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            filename=relative_path,
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
        )

    def enumerate_extraction_units(self) -> Iterable[object]:
        return ()

    def extract_unit(self, unit: object) -> None:
        del unit
        raise RuntimeError("UrbanSyn does not use an extract stage")

    def enumerate_source_items(self) -> Iterable[UrbanSynSourceItem]:
        for frame_id in self._selected_frame_ids():
            image_path = self.paths.raw / self._rgb_relative_path(frame_id)
            depth_path = self.paths.raw / self._depth_relative_path(frame_id)
            semantic_relative_path = self._semantic_relative_path(frame_id) if self._use_semantic_masks() else None
            semantic_path = None if semantic_relative_path is None else self.paths.raw / semantic_relative_path
            if not image_path.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=frame_id,
                    error_message=f"missing UrbanSyn RGB file: {image_path}",
                )
                continue
            if not depth_path.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=frame_id,
                    error_message=f"missing UrbanSyn depth file: {depth_path}",
                )
                continue
            if semantic_path is not None and not semantic_path.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=frame_id,
                    error_message=f"missing UrbanSyn semantic file: {semantic_path}",
                )
                continue
            yield UrbanSynSourceItem(
                frame_id=frame_id,
                image_relative_path=self._rgb_relative_path(frame_id),
                depth_relative_path=self._depth_relative_path(frame_id),
                semantic_relative_path=semantic_relative_path,
            )

    def load_source_item(self, item: UrbanSynSourceItem) -> dict[str, object]:
        image = np.asarray(Image.open(self.paths.raw / item.image_relative_path).convert("RGB"), dtype=np.float32) / 255.0
        depth_raw, exr_metadata = self._load_exr_payload(self.paths.raw / item.depth_relative_path)
        semantic = None
        if item.semantic_relative_path is not None:
            semantic = self._load_semantic_mask(self.paths.raw / item.semantic_relative_path)
        return {
            "image": image,
            "depth_raw": depth_raw,
            "exr_metadata": exr_metadata,
            "semantic": semantic,
        }

    def _load_semantic_mask(self, path: Path) -> np.ndarray:
        semantic = np.asarray(Image.open(path), dtype=np.int32)
        if semantic.ndim == 3:
            if semantic.shape[-1] == 1:
                return semantic[..., 0]
            if semantic.shape[-1] == 3 and np.array_equal(semantic[..., 0], semantic[..., 1]) and np.array_equal(
                semantic[..., 0], semantic[..., 2]
            ):
                return semantic[..., 0]
            raise ValueError("UrbanSyn semantic mask must be single-channel or RGB with identical channels")
        if semantic.ndim != 2:
            raise ValueError("UrbanSyn semantic mask must decode to a 2D array")
        return semantic

    def _load_exr_depth(self, path: Path) -> np.ndarray:
        depth, _ = self._load_exr_payload(path)
        return depth

    def _load_exr_payload(self, path: Path) -> tuple[np.ndarray, dict[str, object]]:
        try:
            import OpenEXR
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("UrbanSyn processing requires the OpenEXR Python package") from exc

        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = int(data_window.max.x - data_window.min.x + 1)
        height = int(data_window.max.y - data_window.min.y + 1)
        channel_names = list(header["channels"].keys())
        for channel_name in ("Z", "Y", "R", "G", "B"):
            if channel_name not in channel_names:
                continue
            channel = exr_file.channel(channel_name)
            depth = np.frombuffer(channel, dtype=np.float32).reshape(height, width)
            octane_payload = header.get("octane")
            metadata: dict[str, object] = {}
            if isinstance(octane_payload, bytes):
                metadata["octane"] = json.loads(octane_payload.decode("utf-8"))
            return depth, metadata
        raise ValueError(f"UrbanSyn EXR file does not expose a supported depth channel: {path}")

    def build_camera_model(self, item: UrbanSynSourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        height, width = image.shape[:2]
        exr_metadata = loaded_item.get("exr_metadata")
        intrinsics = self._camera_intrinsics_for_image(width=width, height=height, exr_metadata=exr_metadata)
        return PinholeCameraModel(
            width=width,
            height=height,
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )

    def _camera_intrinsics_for_image(
        self,
        *,
        width: int,
        height: int,
        exr_metadata: object | None = None,
    ) -> dict[str, float]:
        derived = self._camera_intrinsics_from_exr_metadata(width=width, height=height, exr_metadata=exr_metadata)
        if derived is not None:
            return derived
        configured = self.dataset_config.options.get("camera_intrinsics")
        if configured is None:
            configured = self.DEFAULT_CAMERA_INTRINSICS
        if not isinstance(configured, dict):
            raise ValueError("UrbanSyn camera_intrinsics must be an object")
        required = ("width", "height", "fx", "fy", "cx", "cy")
        if any(key not in configured for key in required):
            raise ValueError("UrbanSyn camera_intrinsics must contain width, height, fx, fy, cx, and cy")
        source_width = float(configured["width"])
        source_height = float(configured["height"])
        if source_width <= 0.0 or source_height <= 0.0:
            raise ValueError("UrbanSyn camera_intrinsics width and height must be positive")
        scale_x = width / source_width
        scale_y = height / source_height
        return {
            "fx": float(configured["fx"]) * scale_x,
            "fy": float(configured["fy"]) * scale_y,
            "cx": float(configured["cx"]) * scale_x,
            "cy": float(configured["cy"]) * scale_y,
        }

    def _camera_intrinsics_from_exr_metadata(
        self,
        *,
        width: int,
        height: int,
        exr_metadata: object | None,
    ) -> dict[str, float] | None:
        if not isinstance(exr_metadata, dict):
            return None
        octane = exr_metadata.get("octane")
        if not isinstance(octane, dict):
            return None
        render_target = octane.get("renderTarget")
        if not isinstance(render_target, dict):
            return None
        camera = render_target.get("camera")
        if not isinstance(camera, dict):
            return None
        resolution = render_target.get("resolution")
        if not isinstance(resolution, dict):
            return None
        dimensions = resolution.get("dimensions")
        if not isinstance(dimensions, dict):
            return None

        source_width = float(dimensions.get("x", width))
        source_height = float(dimensions.get("y", height))
        if source_width <= 0.0 or source_height <= 0.0:
            return None

        fx: float | None = None
        fy: float | None = None
        focal_length = camera.get("focalLength")
        sensor_width = camera.get("sensorWidth")
        if focal_length is not None and sensor_width not in (None, 0):
            fx = source_width * float(focal_length) / float(sensor_width)
            fy = fx
        elif camera.get("fov") is not None:
            fx = (source_width / 2.0) / math.tan(math.radians(float(camera["fov"])) / 2.0)
            fy = fx
        if fx is None or fy is None:
            return None

        lens_shift = camera.get("lensShift", {})
        shift_x = float(lens_shift.get("x", 0.0)) if isinstance(lens_shift, dict) else 0.0
        shift_y = float(lens_shift.get("y", 0.0)) if isinstance(lens_shift, dict) else 0.0
        cx = source_width / 2.0 - shift_x * source_width
        cy = source_height / 2.0 - shift_y * source_height

        scale_x = width / source_width
        scale_y = height / source_height
        return {
            "fx": fx * scale_x,
            "fy": fy * scale_y,
            "cx": cx * scale_x,
            "cy": cy * scale_y,
        }

    def build_sample(self, item: UrbanSynSourceItem, loaded_item: object, camera_model: PinholeCameraModel) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth_raw = np.asarray(loaded_item["depth_raw"], dtype=np.float32)
        semantic = loaded_item["semantic"]
        if depth_raw.ndim == 3 and depth_raw.shape[-1] == 1:
            depth_raw = depth_raw[..., 0]
        if depth_raw.ndim != 2:
            raise ValueError("UrbanSyn depth must decode to a 2D array")
        if image.shape[:2] != depth_raw.shape:
            raise ValueError("UrbanSyn image and depth shapes must match")

        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        depth_meters = depth_raw * float(self.dataset_config.options.get("depth_unit_meters", 1e-5))
        if not np.isfinite(depth_meters).all():
            raise ValueError("UrbanSyn depth contains non-finite values")
        if float(np.min(depth_meters)) < 0.0:
            raise ValueError("UrbanSyn depth contains negative values")

        depth_semantics = str(self.dataset_config.options.get("depth_semantics", "z_depth"))
        if depth_semantics == "distance":
            distance = depth_meters[..., None].astype(np.float32)
        elif depth_semantics == "z_depth":
            distance = z_depth_to_distance(depth_meters[..., None].astype(np.float32), ray_dir).astype(np.float32)
        else:
            raise ValueError("UrbanSyn depth_semantics must be 'distance' or 'z_depth'")

        if semantic is not None:
            semantic_array = np.asarray(semantic, dtype=np.int32)
            if semantic_array.shape != depth_raw.shape:
                raise ValueError("UrbanSyn semantic mask shape must match image and depth")
            sky_class_id = int(self.dataset_config.options.get("sky_class_id", 10))
            distance[..., 0][semantic_array == sky_class_id] = self.config.project.max_dist

        distance = clip_distance_to_max_dist(distance, self.config.project.max_dist)

        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance={
                "frame_id": item.frame_id,
                "image_relative_path": item.image_relative_path,
                "depth_relative_path": item.depth_relative_path,
                "semantic_relative_path": item.semantic_relative_path,
                "depth_semantics": depth_semantics,
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
        assert isinstance(unit, UrbanSynFrameUnit)
        return unit.frame_id

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, UrbanSynSourceItem)
        return item.frame_id

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        yield self.paths.raw / "rgb"
        yield self.paths.raw / "depth"
        if self._use_semantic_masks():
            yield self.paths.raw / "ss"

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, UrbanSynFrameUnit)
        return self.paths.raw / self._rgb_relative_path(unit.frame_id)

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        del unit
        return self.paths.raw

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, UrbanSynFrameUnit)
        image_path = self.paths.raw / self._rgb_relative_path(unit.frame_id)
        depth_path = self.paths.raw / self._depth_relative_path(unit.frame_id)
        if not image_path.exists() or not depth_path.exists():
            return False
        if self._use_semantic_masks():
            semantic_path = self.paths.raw / self._semantic_relative_path(unit.frame_id)
            if not semantic_path.exists():
                return False
        return True

    def is_partial_download_build(self) -> bool:
        return self.dataset_selection() != self.ALL_SELECTION

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(len(shard_names) * self.config.project.train_val_split)
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "UrbanSynFrameUnit",
    "UrbanSynPipeline",
    "UrbanSynSourceItem",
]
