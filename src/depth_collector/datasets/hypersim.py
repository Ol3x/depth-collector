from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Iterable
import zipfile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from depth_collector.core.pipeline import DatasetPipeline
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.geometry import PinholeCameraModel, clip_distance_to_max_dist, normalize_rays
from depth_collector.io import ShardWriter


@dataclass(frozen=True)
class HypersimSceneUnit:
    scene_name: str

    @property
    def filename(self) -> str:
        return f"{self.scene_name}.zip"


@dataclass(frozen=True)
class HypersimSourceItem:
    scene_name: str
    camera_name: str
    frame_id: str
    color_relative_path: str
    depth_relative_path: str
    depth_plane_relative_path: str


class HypersimPipeline(DatasetPipeline):
    """Scene-based Hypersim pipeline using HDF5 geometry and camera metadata."""

    ALL_SELECTOR_VALUES = {"*", "all"}
    FRAME_PATTERN = re.compile(r"frame\.(\d{4})\.")
    COLOR_SUFFIX = ".tonemap.jpg"
    DEPTH_SUFFIX = ".depth_meters.hdf5"
    DEPTH_PLANE_SUFFIX = ".depth_meters_plane.npz"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._written_shards: list[dict[str, object]] = []

    def _selected_scenes(self) -> list[str]:
        configured = self.dataset_config.options.get("scenes")
        if isinstance(configured, str) and configured.strip().lower() not in self.ALL_SELECTOR_VALUES:
            scenes = [configured]
        elif isinstance(configured, list):
            scenes = [
                str(scene)
                for scene in configured
                if str(scene).strip().lower() not in self.ALL_SELECTOR_VALUES
            ]
            if not scenes:
                scenes = self._discover_all_scenes()
        else:
            scenes = self._discover_all_scenes()
        bad_scenes_config = self.dataset_config.options.get("bad_scenes", [])
        if isinstance(bad_scenes_config, str):
            bad_scenes = {bad_scenes_config}
        else:
            bad_scenes = {str(scene) for scene in bad_scenes_config}
        if bad_scenes:
            scenes = [scene for scene in scenes if scene not in bad_scenes]
        count = self.dataset_config.options.get("scene_count")
        if count is None:
            return scenes
        scene_count = int(count)
        if scene_count < 1:
            raise ValueError("scene_count must be at least 1")
        return scenes[: min(scene_count, len(scenes))]

    def _discover_all_scenes(self) -> list[str]:
        archive_scenes = sorted(path.stem for path in self.paths.raw.glob("ai_*_*.zip"))
        if archive_scenes:
            return archive_scenes
        extracted_scenes = sorted(path.name for path in self.paths.raw.glob("ai_*_*") if path.is_dir())
        if extracted_scenes:
            return extracted_scenes
        return self._list_remote_scene_names()

    def _list_remote_scene_names(self) -> list[str]:
        scene_names: set[str] = set()
        for repo_path in self.hf_list_repo_files(repo_id=self.dataset_config.hf_dataset_id, repo_type="dataset"):
            parts = Path(repo_path).parts
            if not parts:
                continue
            first = parts[0]
            if len(parts) == 1 and first.startswith("ai_") and first.endswith(".zip"):
                scene_names.add(Path(first).stem)
                continue
            if first.startswith("ai_"):
                scene_names.add(first)
        return sorted(scene_names)

    def _selected_camera_names(self, scene_root: Path) -> list[str]:
        configured = self.dataset_config.options.get("camera_trajectories")
        if isinstance(configured, str):
            return [configured]
        if isinstance(configured, list):
            return [str(name) for name in configured]
        detail_root = scene_root / "_detail"
        return sorted(path.name for path in detail_root.glob("cam_*") if path.is_dir())

    def enumerate_download_units(self) -> Iterable[HypersimSceneUnit]:
        for scene_name in self._selected_scenes():
            yield HypersimSceneUnit(scene_name=scene_name)

    def download_unit(self, unit: HypersimSceneUnit) -> None:
        if self._download_mode() == "directory":
            self._download_scene_directory_from_hub(unit)
            return
        archive_path = self._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = self._download_archive_from_hub(unit)
        downloaded_path = Path(downloaded_path)
        if downloaded_path.resolve() != archive_path.resolve():
            shutil.copy2(downloaded_path, archive_path)

    def enumerate_extraction_units(self) -> Iterable[HypersimSceneUnit]:
        if self._download_mode() == "directory":
            return ()
        return self.enumerate_download_units()

    def extract_unit(self, unit: HypersimSceneUnit) -> None:
        if self._download_mode() == "directory":
            return
        archive_path = self._archive_path(unit)
        if not archive_path.exists():
            raise FileNotFoundError(f"missing archive for extraction: {archive_path}")
        self.paths.raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(self.paths.raw)

    def enumerate_source_items(self) -> Iterable[HypersimSourceItem]:
        for scene_name in self._selected_scenes():
            scene_root = self._scene_root(scene_name)
            if not scene_root.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=scene_name,
                    error_message=f"missing extracted scene directory: {scene_root}",
                )
                continue
            images_root = scene_root / "images"
            if not images_root.exists():
                self.record_error(
                    stage="enumeration",
                    item_id=scene_name,
                    error_message=f"missing images directory: {images_root}",
                )
                continue
            for camera_name in self._selected_camera_names(scene_root):
                geometry_root = images_root / f"scene_{camera_name}_geometry_hdf5"
                final_root = images_root / f"scene_{camera_name}_final_preview"
                if not geometry_root.exists():
                    self.record_error(
                        stage="enumeration",
                        item_id=f"{scene_name}/{camera_name}",
                        error_message=f"missing geometry directory: {geometry_root}",
                    )
                    continue
                if not final_root.exists():
                    self.record_error(
                        stage="enumeration",
                        item_id=f"{scene_name}/{camera_name}",
                        error_message=f"missing final image directory: {final_root}",
                    )
                    continue

                depth_by_frame: dict[str, str] = {}
                depth_plane_by_frame: dict[str, str] = {}
                for path in sorted(geometry_root.glob(f"*{self.DEPTH_SUFFIX}")):
                    frame_id = self._frame_id_from_name(path.name)
                    depth_by_frame[frame_id] = str(path.relative_to(scene_root))
                for path in sorted(geometry_root.glob(f"*{self.DEPTH_PLANE_SUFFIX}")):
                    frame_id = self._frame_id_from_name(path.name)
                    depth_plane_by_frame[frame_id] = str(path.relative_to(scene_root))

                for color_path in sorted(final_root.glob(f"*{self.COLOR_SUFFIX}")):
                    frame_id = self._frame_id_from_name(color_path.name)
                    depth_relative_path = depth_by_frame.get(frame_id)
                    depth_plane_relative_path = depth_plane_by_frame.get(frame_id)
                    if depth_relative_path is None or depth_plane_relative_path is None:
                        self.record_error(
                            stage="enumeration",
                            item_id=f"{scene_name}/{camera_name}/{frame_id}",
                            error_message=(
                                f"missing paired geometry files for Hypersim frame: "
                                f"{color_path.name}"
                            ),
                        )
                        continue
                    yield HypersimSourceItem(
                        scene_name=scene_name,
                        camera_name=camera_name,
                        frame_id=frame_id,
                        color_relative_path=str(color_path.relative_to(scene_root)),
                        depth_relative_path=depth_relative_path,
                        depth_plane_relative_path=depth_plane_relative_path,
                    )

    def load_source_item(self, item: HypersimSourceItem) -> dict[str, object]:
        scene_root = self._scene_root(item.scene_name)
        color = self._load_image_array(scene_root / item.color_relative_path)
        depth_meters = self._load_hdf5_array(scene_root / item.depth_relative_path)
        depth_plane_meters = self._load_npz_array(scene_root / item.depth_plane_relative_path)
        if depth_meters.ndim == 3 and depth_meters.shape[-1] == 1:
            depth_meters = depth_meters[..., 0]
        if depth_plane_meters.ndim == 3 and depth_plane_meters.shape[-1] == 1:
            depth_plane_meters = depth_plane_meters[..., 0]
        if depth_meters.ndim != 2:
            raise ValueError("Hypersim depth_meters must decode to a 2D array")
        if depth_plane_meters.ndim != 2:
            raise ValueError("Hypersim depth_meters_plane must decode to a 2D array")

        orientation_path = scene_root / "_detail" / item.camera_name / "camera_keyframe_orientations.hdf5"
        position_path = scene_root / "_detail" / item.camera_name / "camera_keyframe_positions.hdf5"
        orientations = self._load_hdf5_array(orientation_path)
        camera_positions_asset = self._load_hdf5_array(position_path)
        frame_index = int(item.frame_id.split(".")[1])
        orientation_world_from_camera = np.asarray(orientations[frame_index], dtype=np.float32)
        camera_position_asset = np.asarray(camera_positions_asset[frame_index], dtype=np.float32)
        camera_parameters = self._load_camera_parameters(item.scene_name)

        return {
            "image": color,
            "depth_meters": np.asarray(depth_meters, dtype=np.float32),
            "depth_plane_meters": np.asarray(depth_plane_meters, dtype=np.float32),
            "orientation_world_from_camera": orientation_world_from_camera,
            "camera_position_asset": camera_position_asset,
            "meters_per_asset_unit": camera_parameters["meters_per_asset_unit"],
            "m_cam_from_uv": camera_parameters["m_cam_from_uv"],
        }

    def build_camera_model(self, item: HypersimSourceItem, loaded_item: object) -> PinholeCameraModel:
        del item
        assert isinstance(loaded_item, dict)
        image = loaded_item["image"]
        assert isinstance(image, np.ndarray)
        height, width = image.shape[:2]
        m_cam_from_uv = np.asarray(loaded_item["m_cam_from_uv"], dtype=np.float32)
        fx = width / (2.0 * float(m_cam_from_uv[0, 0]))
        fy = height / (2.0 * float(m_cam_from_uv[1, 1]))
        return PinholeCameraModel(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=width / 2.0,
            cy=height / 2.0,
        )

    def build_sample(
        self,
        item: HypersimSourceItem,
        loaded_item: object,
        camera_model: PinholeCameraModel,
    ) -> SampleRecord:
        del camera_model
        assert isinstance(loaded_item, dict)
        image = loaded_item["image"]
        depth_meters = loaded_item["depth_meters"]
        depth_plane_meters = loaded_item["depth_plane_meters"]
        orientation_world_from_camera = loaded_item["orientation_world_from_camera"]
        m_cam_from_uv = loaded_item["m_cam_from_uv"]
        assert isinstance(image, np.ndarray)
        assert isinstance(depth_meters, np.ndarray)
        assert isinstance(depth_plane_meters, np.ndarray)
        assert isinstance(orientation_world_from_camera, np.ndarray)
        assert isinstance(m_cam_from_uv, np.ndarray)

        if image.shape[:2] != depth_meters.shape or image.shape[:2] != depth_plane_meters.shape:
            raise ValueError("Hypersim image and depth shapes must match")

        points_camera_hypersim = self._reconstruct_camera_points(depth_plane_meters, np.asarray(m_cam_from_uv, dtype=np.float32))
        ray_dir = normalize_rays(-points_camera_hypersim).astype(np.float32)
        distance_from_points = np.linalg.norm(points_camera_hypersim, axis=-1, keepdims=True).astype(np.float32)
        self._validate_hypersim_depth_semantics(
            distance_from_points[..., 0],
            depth_meters,
            depth_plane_meters,
        )
        distance = clip_distance_to_max_dist(distance_from_points, self.config.project.max_dist)

        return SampleRecord(
            sample_id=self.get_source_item_id(item),
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance={
                "scene_name": item.scene_name,
                "camera_name": item.camera_name,
                "frame_id": item.frame_id,
                "color_relative_path": item.color_relative_path,
                "depth_relative_path": item.depth_relative_path,
                "depth_plane_relative_path": item.depth_plane_relative_path,
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
        if not self.paths.metadata.exists():
            raise ValueError("metadata.json was not created")

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, HypersimSceneUnit)
        return unit.scene_name

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, HypersimSourceItem)
        return f"{item.scene_name}/{item.camera_name}/{item.frame_id}"

    def iter_download_artifact_paths(self) -> Iterable[Path]:
        for unit in self.enumerate_download_units():
            if self._download_mode() == "directory":
                yield self._scene_root(unit.scene_name)
            else:
                yield self._archive_path(unit)

    def remove_download_artifact(self, unit: object) -> None:
        if self._download_mode() == "directory":
            assert isinstance(unit, HypersimSceneUnit)
            scene_root = self._scene_root(unit.scene_name)
            if scene_root.exists():
                shutil.rmtree(scene_root)
            return
        archive_path = self._archive_path(unit)
        if archive_path.exists():
            archive_path.unlink()

    def get_download_artifact_path(self, unit: object) -> Path | None:
        if self._download_mode() == "directory":
            assert isinstance(unit, HypersimSceneUnit)
            return self._scene_root(unit.scene_name)
        return self._archive_path(unit)

    def get_extracted_artifact_root(self, unit: object) -> Path | None:
        assert isinstance(unit, HypersimSceneUnit)
        return self._scene_root(unit.scene_name)

    def is_download_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, HypersimSceneUnit)
        if self._download_mode() == "directory":
            scene_root = self._scene_root(unit.scene_name)
            return scene_root.exists() and any(path.is_file() for path in scene_root.rglob("*"))
        return self._archive_path(unit).exists()

    def is_extraction_unit_satisfied(self, unit: object) -> bool:
        assert isinstance(unit, HypersimSceneUnit)
        if self._download_mode() == "directory":
            return self.is_download_unit_satisfied(unit)
        scene_root = self._scene_root(unit.scene_name)
        if not scene_root.exists():
            return False
        return any(path.is_file() for path in scene_root.rglob("*"))

    def _archive_path(self, unit: object) -> Path:
        assert isinstance(unit, HypersimSceneUnit)
        return self.paths.raw / unit.filename

    def _scene_root(self, scene_name: str) -> Path:
        return self.paths.raw / scene_name

    def _download_archive_from_hub(self, unit: HypersimSceneUnit) -> Path:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            local_path = Path(str(local_archive_root)) / unit.filename
            if not local_path.exists():
                raise FileNotFoundError(f"missing local archive source: {local_path}")
            return local_path

        prefix = str(self.dataset_config.options.get("hf_path_prefix", "")).strip("/")
        filename = "/".join(part for part in (prefix, unit.filename) if part)
        return self.hf_hub_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            filename=filename,
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
        )

    def _download_scene_directory_from_hub(self, unit: HypersimSceneUnit) -> None:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_root = Path(str(local_archive_root)) / unit.scene_name
            if not source_root.exists():
                raise FileNotFoundError(f"missing local scene source: {source_root}")
            shutil.copytree(source_root, self._scene_root(unit.scene_name), dirs_exist_ok=True)
            return

        self.hf_snapshot_download(
            repo_id=self.dataset_config.hf_dataset_id,
            repo_type="dataset",
            revision=self.dataset_config.revision,
            local_dir=self.paths.raw,
            allow_patterns=[
                f"{unit.scene_name}/**",
                "metadata_camera_parameters.csv",
                "metadata_images_split_scene_v1.csv",
            ],
            tqdm_class=_SilentTqdm,
        )

    def get_download_progress_plan(self, unit: object) -> dict[str, object] | None:
        if self._download_mode() != "directory":
            return None
        assert isinstance(unit, HypersimSceneUnit)
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            source_root = Path(str(local_archive_root)) / unit.scene_name
            if not source_root.exists():
                return None
            total_files = sum(1 for path in source_root.rglob("*") if path.is_file())
            return {
                "label": unit.scene_name,
                "root": self._scene_root(unit.scene_name),
                "total_files": total_files,
            }
        return None

    def _download_mode(self) -> str:
        mode = str(self.dataset_config.options.get("download_mode", "directory"))
        if mode not in {"directory", "archive"}:
            raise ValueError(f"unsupported Hypersim download_mode: {mode}")
        return mode

    def _load_hdf5_array(self, path: Path) -> np.ndarray:
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Hypersim processing requires h5py") from exc
        with h5py.File(path, "r") as handle:
            if len(handle.keys()) == 1:
                dataset = handle[next(iter(handle.keys()))]
                return np.asarray(dataset)
            if "dataset" in handle:
                return np.asarray(handle["dataset"])
            raise ValueError(f"could not determine dataset key in HDF5 file: {path}")

    def _load_npz_array(self, path: Path) -> np.ndarray:
        arrays = np.load(path)
        if "data" in arrays:
            return np.asarray(arrays["data"])
        if arrays.files:
            return np.asarray(arrays[arrays.files[0]])
        raise ValueError(f"could not determine array key in NPZ file: {path}")

    def _load_image_array(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0

    def _load_camera_parameters(self, scene_name: str) -> dict[str, object]:
        metadata_path = self.paths.raw / "metadata_camera_parameters.csv"
        with metadata_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("scene_name") != scene_name:
                    continue
                m_cam_from_uv = np.array(
                    [
                        [float(row["M_cam_from_uv_00"]), float(row["M_cam_from_uv_01"]), float(row["M_cam_from_uv_02"])],
                        [float(row["M_cam_from_uv_10"]), float(row["M_cam_from_uv_11"]), float(row["M_cam_from_uv_12"])],
                        [float(row["M_cam_from_uv_20"]), float(row["M_cam_from_uv_21"]), float(row["M_cam_from_uv_22"])],
                    ],
                    dtype=np.float32,
                )
                return {
                    "meters_per_asset_unit": float(row["settings_units_info_meters_scale"]),
                    "m_cam_from_uv": m_cam_from_uv,
                }
        raise ValueError(f"could not load Hypersim camera parameters for scene {scene_name}")

    def _load_meters_per_asset_unit(self, path: Path) -> float:
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            for row in rows:
                if "parameter_name" in row and row.get("parameter_name") == "meters_per_asset_unit":
                    return float(row["parameter_value"])
                if "meters_per_asset_unit" in row and row.get("meters_per_asset_unit"):
                    return float(row["meters_per_asset_unit"])
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                for index, cell in enumerate(row):
                    if cell == "meters_per_asset_unit" and index + 1 < len(row):
                        return float(row[index + 1])
        raise ValueError(f"could not load meters_per_asset_unit from {path}")

    def _frame_id_from_name(self, filename: str) -> str:
        match = self.FRAME_PATTERN.search(filename)
        if match is None:
            raise ValueError(f"could not parse Hypersim frame id from {filename}")
        return f"frame.{match.group(1)}"

    def _reconstruct_camera_points(self, depth_plane_meters: np.ndarray, m_cam_from_uv: np.ndarray) -> np.ndarray:
        height, width = depth_plane_meters.shape
        u = ((np.arange(width, dtype=np.float32) + 0.5) / width) * 2.0 - 1.0
        v = ((np.arange(height, dtype=np.float32) + 0.5) / height) * 2.0 - 1.0
        grid_u, grid_v = np.meshgrid(u, v)
        uv1 = np.stack([grid_u, grid_v, np.ones_like(grid_u)], axis=-1)
        rays_camera_hypersim = np.einsum("...j,ij->...i", uv1, m_cam_from_uv)
        points_camera_hypersim = rays_camera_hypersim * depth_plane_meters[..., None]
        return points_camera_hypersim.astype(np.float32)

    def _validate_hypersim_depth_semantics(
        self,
        distance_from_points_meters: np.ndarray,
        depth_meters: np.ndarray,
        depth_plane_meters: np.ndarray,
    ) -> None:
        valid_mask = (
            np.isfinite(distance_from_points_meters)
            & np.isfinite(depth_meters)
            & np.isfinite(depth_plane_meters)
            & (depth_meters > 1e-6)
            & (depth_plane_meters > 1e-6)
        )
        if not np.any(valid_mask):
            raise ValueError("Hypersim sample has no valid distance pixels")
        radial_relative_difference = np.abs(distance_from_points_meters - depth_meters) / np.maximum(depth_meters, 1e-6)
        mean_radial_relative_difference = float(np.mean(radial_relative_difference[valid_mask]))
        if mean_radial_relative_difference > 1e-2:
            raise ValueError(
                f"Hypersim point-derived distance disagrees with depth_meters "
                f"(mean relative difference={mean_radial_relative_difference:.6f})"
            )
        plane_relative_difference = (
            np.abs(distance_from_points_meters - depth_plane_meters) / np.maximum(depth_plane_meters, 1e-6)
        )
        mean_plane_relative_difference = float(np.mean(plane_relative_difference[valid_mask]))
        if mean_plane_relative_difference < 1e-2:
            raise ValueError(
                f"Hypersim point-derived distance is suspiciously close to depth_meters_plane "
                f"(mean relative difference={mean_plane_relative_difference:.6f})"
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
    "HypersimPipeline",
    "HypersimSceneUnit",
    "HypersimSourceItem",
]


class _SilentTqdm(tqdm):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
