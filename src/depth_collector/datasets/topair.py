from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
class TopAirTrajectoryUnit:
    trajectory_name: str


@dataclass(frozen=True)
class TopAirSourceItem:
    trajectory_name: str
    frame_id: str
    image_relative_path: str
    depth_relative_path: str
    semantic_relative_path: str | None = None


class TopAirPipeline(DatasetPipeline):
    """Trajectory-folder TopAir pipeline using pinhole nadir-view geometry."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    DEFAULT_CAMERA_INTRINSICS = {
        "width": 384.0,
        "height": 384.0,
        "fx": 192.0,
        "fy": 192.0,
        "cx": 192.0,
        "cy": 192.0,
    }

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []
        self._remote_repo_files_cache: tuple[str, ...] | None = None
        self._camera_pose_cache: dict[str, dict[str, dict[str, float]]] = {}

    def _remote_repo_files(self) -> tuple[str, ...]:
        if self._remote_repo_files_cache is None:
            self._remote_repo_files_cache = tuple(self.hf_list_repo_files(repo_id=self.dataset_config.hf_dataset_id))
        return self._remote_repo_files_cache

    def _trajectory_option(self) -> list[str]:
        configured = self.dataset_config.options.get("trajectories", "*")
        if isinstance(configured, str):
            if configured.strip().lower() in self.ALL_SELECTOR_VALUES:
                return []
            return [configured]
        return [
            str(trajectory_name)
            for trajectory_name in configured
            if str(trajectory_name).strip().lower() not in self.ALL_SELECTOR_VALUES
        ]

    def _use_semantic_masks(self) -> bool:
        return bool(self.dataset_config.options.get("use_semantic_masks", True))

    def _camera_intrinsics(self) -> dict[str, float]:
        configured = self.dataset_config.options.get("camera_intrinsics", {})
        assert isinstance(configured, dict)
        values = dict(self.DEFAULT_CAMERA_INTRINSICS)
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if key in configured:
                values[key] = float(configured[key])
        return values

    def _discover_trajectory_names_from_root(self, root: Path) -> list[str]:
        if not root.exists():
            return []
        trajectory_names: list[str] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if not (path / "images").exists():
                continue
            if not (path / "depth").exists():
                continue
            if self._use_semantic_masks() and not (path / "seg_id").exists():
                continue
            trajectory_names.append(path.name)
        return trajectory_names

    def _discover_trajectory_names_from_remote(self) -> list[str]:
        trajectory_names: set[str] = set()
        for repo_path in self._remote_repo_files():
            parts = Path(repo_path).parts
            if len(parts) >= 2:
                trajectory_names.add(parts[0])
        return sorted(trajectory_names)

    def _selected_trajectory_names(self) -> list[str]:
        configured = self._trajectory_option()
        if configured:
            trajectory_names = configured
        else:
            local_archive_root = self.dataset_config.options.get("local_archive_root")
            if local_archive_root:
                trajectory_names = self._discover_trajectory_names_from_root(Path(str(local_archive_root)))
            else:
                trajectory_names = []
            if not trajectory_names:
                trajectory_names = self._discover_trajectory_names_from_root(self.paths.raw)
            if not trajectory_names:
                trajectory_names = self._discover_trajectory_names_from_remote()
        count = self.dataset_config.options.get("trajectory_count")
        if count is None:
            return trajectory_names
        trajectory_count = int(count)
        if trajectory_count < 1:
            raise ValueError("trajectory_count must be at least 1")
        return trajectory_names[: min(trajectory_count, len(trajectory_names))]

    def enumerate_download_units(self) -> Iterable[TopAirTrajectoryUnit]:
        for trajectory_name in self._selected_trajectory_names():
            yield TopAirTrajectoryUnit(trajectory_name=trajectory_name)

    def download_unit(self, unit: TopAirTrajectoryUnit) -> None:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_root = Path(str(local_archive_root)) / unit.trajectory_name
            if not source_root.exists():
                raise FileNotFoundError(f"missing local TopAir trajectory source: {source_root}")
            shutil.copytree(source_root, self.paths.raw / unit.trajectory_name, dirs_exist_ok=True)
            return

        self.hf_snapshot_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
            allow_patterns=[f"{unit.trajectory_name}/**"],
        )

    def enumerate_extraction_units(self) -> Iterable[object]:
        return ()

    def extract_unit(self, unit: object) -> None:
        del unit
        raise RuntimeError("TopAir does not use an extract stage")

    def enumerate_source_items(self) -> Iterable[TopAirSourceItem]:
        for trajectory_name in self._selected_trajectory_names():
            trajectory_root = self.paths.raw / trajectory_name
            images_root = trajectory_root / "images"
            depth_root = trajectory_root / "depth"
            semantic_root = trajectory_root / "seg_id"
            if not images_root.exists():
                self.record_error("enumeration", trajectory_name, f"missing TopAir images directory: {images_root}")
                continue
            if not depth_root.exists():
                self.record_error("enumeration", trajectory_name, f"missing TopAir depth directory: {depth_root}")
                continue
            if self._use_semantic_masks() and not semantic_root.exists():
                self.record_error("enumeration", trajectory_name, f"missing TopAir semantic directory: {semantic_root}")
                continue

            depth_by_frame = {
                path.stem: path
                for path in sorted(depth_root.rglob("*"))
                if path.is_file() and path.suffix.lower() == ".png"
            }
            semantic_by_frame = {
                path.stem: path
                for path in sorted(semantic_root.rglob("*"))
                if path.is_file() and path.suffix.lower() == ".png"
            }
            for image_path in sorted(images_root.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                frame_id = image_path.stem
                depth_path = depth_by_frame.get(frame_id)
                if depth_path is None:
                    self.record_error(
                        "enumeration",
                        f"{trajectory_name}/{image_path.name}",
                        f"missing paired TopAir depth file for image: {image_path.name}",
                    )
                    continue
                semantic_relative_path = None
                if self._use_semantic_masks():
                    semantic_path = semantic_by_frame.get(frame_id)
                    if semantic_path is None:
                        self.record_error(
                            "enumeration",
                            f"{trajectory_name}/{image_path.name}",
                            f"missing paired TopAir semantic file for image: {image_path.name}",
                        )
                        continue
                    semantic_relative_path = str(semantic_path.relative_to(self.paths.raw))
                yield TopAirSourceItem(
                    trajectory_name=trajectory_name,
                    frame_id=frame_id,
                    image_relative_path=str(image_path.relative_to(self.paths.raw)),
                    depth_relative_path=str(depth_path.relative_to(self.paths.raw)),
                    semantic_relative_path=semantic_relative_path,
                )

    def load_source_item(self, item: TopAirSourceItem) -> dict[str, object]:
        image = np.asarray(Image.open(self.paths.raw / item.image_relative_path).convert("RGB"), dtype=np.float32) / 255.0
        depth_raw = np.asarray(Image.open(self.paths.raw / item.depth_relative_path), dtype=np.uint8)
        semantic = None
        if item.semantic_relative_path is not None:
            semantic = np.asarray(Image.open(self.paths.raw / item.semantic_relative_path))
        return {
            "image": image,
            "depth_raw": depth_raw,
            "semantic": semantic,
            "camera_pose": self._camera_pose_for_item(item),
        }

    def _camera_pose_for_item(self, item: TopAirSourceItem) -> dict[str, float] | None:
        pose_map = self._load_camera_pose_map(item.trajectory_name)
        return pose_map.get(item.frame_id)

    def _load_camera_pose_map(self, trajectory_name: str) -> dict[str, dict[str, float]]:
        cached = self._camera_pose_cache.get(trajectory_name)
        if cached is not None:
            return cached
        camera_loc_path = self.paths.raw / trajectory_name / "camera_loc.txt"
        if not camera_loc_path.exists():
            self._camera_pose_cache[trajectory_name] = {}
            return {}
        lines = [line.strip() for line in camera_loc_path.read_text().splitlines() if line.strip()]
        pose_map: dict[str, dict[str, float]] = {}
        for line in lines:
            parts = [part for part in line.replace(",", " ").split() if part]
            if len(parts) < 7:
                continue
            frame_id = parts[0]
            try:
                values = [float(part) for part in parts[1:7]]
            except ValueError:
                continue
            pose_map[frame_id] = {
                "x": values[0],
                "y": values[1],
                "z": values[2],
                "roll": values[3],
                "pitch": values[4],
                "yaw": values[5],
            }
        self._camera_pose_cache[trajectory_name] = pose_map
        return pose_map

    def build_camera_model(self, item: TopAirSourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        height, width = image.shape[:2]
        intrinsics = self._camera_intrinsics()
        expected_width = int(round(intrinsics["width"]))
        expected_height = int(round(intrinsics["height"]))
        if width != expected_width or height != expected_height:
            scale_x = width / intrinsics["width"]
            scale_y = height / intrinsics["height"]
            fx = intrinsics["fx"] * scale_x
            fy = intrinsics["fy"] * scale_y
            cx = intrinsics["cx"] * scale_x
            cy = intrinsics["cy"] * scale_y
        else:
            fx = intrinsics["fx"]
            fy = intrinsics["fy"]
            cx = intrinsics["cx"]
            cy = intrinsics["cy"]
        return PinholeCameraModel(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    def build_sample(self, item: TopAirSourceItem, loaded_item: object, camera_model: PinholeCameraModel) -> SampleRecord:
        assert isinstance(loaded_item, dict)
        image = np.asarray(loaded_item["image"], dtype=np.float32)
        depth_raw = np.asarray(loaded_item["depth_raw"])
        semantic = loaded_item["semantic"]
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]
        if depth_raw.ndim != 2:
            raise ValueError("TopAir depth must decode to a 2D array")
        if image.shape[:2] != depth_raw.shape:
            raise ValueError("TopAir image and depth shapes must match")

        depth_unit_meters = float(self.dataset_config.options.get("depth_unit_meters", 100.0 / 255.0))
        if depth_unit_meters <= 0.0:
            raise ValueError("TopAir depth_unit_meters must be positive")
        ray_dir = generate_pinhole_rays(camera_model).astype(np.float32)
        depth = depth_raw.astype(np.float32) * depth_unit_meters
        depth_semantics = str(self.dataset_config.options.get("depth_semantics", "distance")).strip().lower()
        if depth_semantics == "distance":
            distance = depth
        elif depth_semantics == "z_depth":
            distance = z_depth_to_distance(depth[..., None], ray_dir)[..., 0]
        else:
            raise ValueError("TopAir depth_semantics must be 'distance' or 'z_depth'")
        invalid_mask = ~np.isfinite(distance) | (distance <= 0.0)
        if semantic is not None:
            semantic = self._normalize_semantic_ids(np.asarray(semantic))
            sky_class_id = int(self.dataset_config.options.get("sky_class_id", 0))
            invalid_mask |= semantic == sky_class_id
        distance = distance[..., None]
        distance[..., 0][invalid_mask] = self.config.project.max_dist
        distance = clip_distance_to_max_dist(distance, self.config.project.max_dist)

        provenance = {
            "trajectory_name": item.trajectory_name,
            "frame_id": item.frame_id,
            "image_relative_path": item.image_relative_path,
            "depth_relative_path": item.depth_relative_path,
            "semantic_relative_path": item.semantic_relative_path,
            "projection": "pinhole",
        }
        camera_pose = loaded_item.get("camera_pose")
        if camera_pose is not None:
            provenance["camera_pose"] = camera_pose
        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance.astype(np.float32),
            ray_dir=ray_dir,
            provenance=provenance,
        )

    def _normalize_semantic_ids(self, semantic: np.ndarray) -> np.ndarray:
        if semantic.ndim == 2:
            return semantic.astype(np.int32)
        if semantic.ndim == 3 and semantic.shape[-1] >= 1:
            first_channel = semantic[..., 0]
            if np.all(semantic == semantic[..., :1]):
                return first_channel.astype(np.int32)
        raise ValueError("TopAir semantic masks must decode to a single ID map")

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
        assert isinstance(unit, TopAirTrajectoryUnit)
        return unit.trajectory_name

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, TopAirSourceItem)
        return f"{item.trajectory_name}/{item.frame_id}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self.enumerate_download_units():
            yield self.paths.raw / unit.trajectory_name

    def get_download_artifact_path(self, unit: object) -> Path | None:
        assert isinstance(unit, TopAirTrajectoryUnit)
        return self.paths.raw / unit.trajectory_name

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, TopAirTrajectoryUnit)
        return self.paths.raw / unit.trajectory_name

    def remove_download_artifact(self, unit: object) -> None:
        assert isinstance(unit, TopAirTrajectoryUnit)
        trajectory_root = self.paths.raw / unit.trajectory_name
        if trajectory_root.exists():
            shutil.rmtree(trajectory_root)

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, TopAirTrajectoryUnit)
        trajectory_root = self.paths.raw / unit.trajectory_name
        images_root = trajectory_root / "images"
        depth_root = trajectory_root / "depth"
        if not images_root.exists() or not depth_root.exists():
            return False
        if self._use_semantic_masks() and not (trajectory_root / "seg_id").exists():
            return False
        return any(path.is_file() for path in images_root.rglob("*")) and any(path.is_file() for path in depth_root.rglob("*"))

    def _suggest_shard_splits(self, shard_names: list[str]) -> tuple[list[str], list[str]]:
        if not shard_names:
            return [], []
        if len(shard_names) == 1:
            return shard_names[:], shard_names[:]
        split_index = int(math.floor(len(shard_names) * self.config.project.train_val_split))
        split_index = min(max(split_index, 1), len(shard_names) - 1)
        return shard_names[:split_index], shard_names[split_index:]


__all__ = [
    "TopAirPipeline",
    "TopAirSourceItem",
    "TopAirTrajectoryUnit",
]
