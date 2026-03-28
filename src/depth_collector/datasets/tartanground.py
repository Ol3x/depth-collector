from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from .tartan import TartanPipeline


@dataclass(frozen=True)
class TartanGroundArchiveUnit:
    environment: str
    version: str
    trajectory: str
    modality: str
    camera_name: str

    @property
    def filename(self) -> str:
        return f"{self.modality}_{self.camera_name}.zip"


@dataclass(frozen=True)
class TartanGroundSourceItem:
    environment: str
    version: str
    trajectory: str
    camera_name: str
    image_relative_path: str
    depth_relative_path: str


class TartanGroundPipeline(TartanPipeline):
    """Concrete TartanGround pipeline built on the shared Tartan family behavior."""

    DEFAULT_MODALITIES = ("image",)
    REQUIRED_MODALITIES = ("image", "depth")
    DEFAULT_VERSIONS = ("omni",)
    DEFAULT_TRAJECTORIES = ("P0000",)
    DEFAULT_CAMERA_NAMES = ("lcam_front",)
    DEPTH_SUFFIXES = {".png"}

    def _selected_versions(self) -> list[str]:
        return self._get_option_list("versions", self.DEFAULT_VERSIONS)

    def _selected_trajectories(self) -> list[str]:
        return self._get_option_list("trajectories", self.DEFAULT_TRAJECTORIES)

    def _selected_camera_names(self) -> list[str]:
        return self._get_option_list("camera_names", self.DEFAULT_CAMERA_NAMES)

    def _selected_group_keys(self) -> list[tuple[str, str, str, str]]:
        groups: list[tuple[str, str, str, str]] = []
        for environment in self._selected_environments():
            for version in self._selected_versions():
                for trajectory in self._selected_trajectories():
                    for camera_name in self._selected_camera_names():
                        groups.append((environment, version, trajectory, camera_name))
        return groups

    def _iter_group_download_units(self, group_key: tuple[object, ...]) -> list[object]:
        environment, version, trajectory, camera_name = group_key
        units: list[TartanGroundArchiveUnit] = []
        for modality in self._selected_modalities():
            units.append(
                TartanGroundArchiveUnit(
                    environment=str(environment),
                    version=str(version),
                    trajectory=str(trajectory),
                    modality=modality,
                    camera_name=str(camera_name),
                )
            )
        return units

    def get_download_unit_id(self, unit: object) -> str:
        assert isinstance(unit, TartanGroundArchiveUnit)
        return f"{unit.environment}/{unit.version}/{unit.trajectory}/{unit.modality}/{unit.camera_name}"

    def get_extraction_unit_id(self, unit: object) -> str:
        return self.get_download_unit_id(unit)

    def get_source_item_id(self, item: object) -> str:
        assert isinstance(item, TartanGroundSourceItem)
        return f"{item.environment}/{item.version}/{item.trajectory}/{item.camera_name}/{item.image_relative_path}"

    def get_group_id(self, group_key: tuple[object, ...]) -> str:
        environment, version, trajectory, camera_name = group_key
        return f"{environment}/{version}/{trajectory}/{camera_name}"

    def get_group_image_dir(self, group_key: tuple[object, ...]) -> Path:
        environment, version, trajectory, camera_name = group_key
        return self._extracted_dir(
            TartanGroundArchiveUnit(
                environment=str(environment),
                version=str(version),
                trajectory=str(trajectory),
                modality="image",
                camera_name=str(camera_name),
            )
        )

    def get_group_depth_dir(self, group_key: tuple[object, ...]) -> Path:
        environment, version, trajectory, camera_name = group_key
        return self._extracted_dir(
            TartanGroundArchiveUnit(
                environment=str(environment),
                version=str(version),
                trajectory=str(trajectory),
                modality="depth",
                camera_name=str(camera_name),
            )
        )

    def _scan_group_source_items(
        self,
        group_key: tuple[object, ...],
        group_id: str,
        image_dir: Path,
        depth_dir: Path,
    ) -> tuple[list[object], list[tuple[str, str]]]:
        environment, version, trajectory, camera_name = group_key
        return self._scan_paired_group_source_items(
            group_id=group_id,
            image_dir=image_dir,
            depth_dir=depth_dir,
            image_key_fn=self._paired_ground_key,
            depth_key_fn=self._paired_ground_key,
            missing_pair_message_fn=lambda image_relative_path: (
                f"missing paired depth file for image frame: {image_relative_path}"
            ),
            item_factory=lambda image_relative_path, depth_relative_path: TartanGroundSourceItem(
                environment=str(environment),
                version=str(version),
                trajectory=str(trajectory),
                camera_name=str(camera_name),
                image_relative_path=image_relative_path,
                depth_relative_path=depth_relative_path,
            ),
        )

    def get_source_item_image_path(self, item: object) -> Path:
        assert isinstance(item, TartanGroundSourceItem)
        return self._extracted_dir(
            TartanGroundArchiveUnit(
                environment=item.environment,
                version=item.version,
                trajectory=item.trajectory,
                modality="image",
                camera_name=item.camera_name,
            )
        ) / item.image_relative_path

    def get_source_item_depth_path(self, item: object) -> Path:
        assert isinstance(item, TartanGroundSourceItem)
        return self._extracted_dir(
            TartanGroundArchiveUnit(
                environment=item.environment,
                version=item.version,
                trajectory=item.trajectory,
                modality="depth",
                camera_name=item.camera_name,
            )
        ) / item.depth_relative_path

    def _archive_path(self, unit: TartanGroundArchiveUnit) -> Path:
        return (
            self.paths.raw
            / unit.environment
            / self._version_directory(unit.version)
            / unit.trajectory
            / unit.filename
        )

    def _extracted_dir(self, unit: TartanGroundArchiveUnit) -> Path:
        return (
            self.paths.raw
            / unit.environment
            / self._version_directory(unit.version)
            / unit.trajectory
            / f"{unit.modality}_{unit.camera_name}"
        )

    def _hub_repo_filename(self, unit: TartanGroundArchiveUnit) -> str:
        prefix = str(self.dataset_config.options.get("hf_path_prefix", "")).strip("/")
        parts = [prefix, unit.environment, self._version_directory(unit.version), unit.trajectory, unit.filename]
        return "/".join(part for part in parts if part)

    def _deserialize_manifest_item(self, payload: dict[str, object]) -> object:
        return TartanGroundSourceItem(
            environment=str(payload["environment"]),
            version=str(payload["version"]),
            trajectory=str(payload["trajectory"]),
            camera_name=str(payload["camera_name"]),
            image_relative_path=str(payload["image_relative_path"]),
            depth_relative_path=str(payload["depth_relative_path"]),
        )

    def _version_directory(self, version: str) -> str:
        mapping = {
            "omni": "Data_omni",
            "diff": "Data_diff",
            "anymal": "Data_anymal",
        }
        if version not in mapping:
            raise ValueError(f"unsupported TartanGround version: {version}")
        return mapping[version]

    def _paired_ground_key(self, relative_path: str) -> str:
        parts = list(Path(relative_path).parts)
        if parts and (parts[0].startswith("image_") or parts[0].startswith("depth_")):
            parts = parts[1:]
        path = Path(*parts)
        stem = path.stem
        if stem.endswith("_depth"):
            stem = stem[: -len("_depth")]
        return str(path.with_name(stem).with_suffix(""))

    def _load_depth_array(self, path: Path) -> np.ndarray:
        if path.suffix.lower() != ".png":
            return super()._load_depth_array(path)
        image = Image.open(path)
        rgba = np.asarray(image, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError("TartanGround depth PNG must decode to RGBA")
        depth = np.frombuffer(rgba.tobytes(), dtype="<f4").reshape(rgba.shape[:2])
        return depth.astype(np.float32, copy=False)


__all__ = [
    "TartanGroundArchiveUnit",
    "TartanGroundPipeline",
    "TartanGroundSourceItem",
]
