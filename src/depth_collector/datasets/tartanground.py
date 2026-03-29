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
    DEFAULT_VERSIONS = ()
    DEFAULT_TRAJECTORIES = ()
    DEFAULT_CAMERA_NAMES = ()
    DEPTH_SUFFIXES = {".png"}

    def _selected_versions(self) -> list[str]:
        configured = self._configured_versions()
        if configured:
            return configured
        return sorted({unit.version for unit in self._discover_archive_units()})

    def _selected_trajectories(self) -> list[str]:
        configured = self._configured_trajectories()
        if configured:
            return configured
        return sorted({unit.trajectory for unit in self._discover_archive_units()})

    def _selected_camera_names(self) -> list[str]:
        configured = self._configured_camera_names()
        if configured:
            return configured
        return sorted({unit.camera_name for unit in self._discover_archive_units()})

    def _configured_versions(self) -> list[str]:
        if self._uses_all_selector("versions", self.DEFAULT_VERSIONS):
            return []
        return self._get_option_list("versions", self.DEFAULT_VERSIONS)

    def _configured_trajectories(self) -> list[str]:
        if self._uses_all_selector("trajectories", self.DEFAULT_TRAJECTORIES):
            return []
        return self._get_option_list("trajectories", self.DEFAULT_TRAJECTORIES)

    def _configured_camera_names(self) -> list[str]:
        if self._uses_all_selector("camera_names", self.DEFAULT_CAMERA_NAMES):
            return []
        return self._get_option_list("camera_names", self.DEFAULT_CAMERA_NAMES)

    def _selected_group_keys(self) -> list[tuple[str, str, str, str]]:
        selected_environments = self._selected_environments()
        selected_environment_set = set(selected_environments)
        configured_versions = self._configured_versions()
        configured_trajectories = self._configured_trajectories()
        configured_camera_names = self._configured_camera_names()

        if not (configured_versions and configured_trajectories and configured_camera_names):
            discovered_groups = sorted(
                {
                    (unit.environment, unit.version, unit.trajectory, unit.camera_name)
                    for unit in self._discover_archive_units()
                    if unit.environment in selected_environment_set
                    and (not configured_versions or unit.version in configured_versions)
                    and (not configured_trajectories or unit.trajectory in configured_trajectories)
                    and (not configured_camera_names or unit.camera_name in configured_camera_names)
                }
            )
            if discovered_groups:
                return [
                    (str(environment), str(version), str(trajectory), str(camera_name))
                    for environment, version, trajectory, camera_name in self._limit_group_keys(discovered_groups)
                ]

        groups: list[tuple[str, str, str, str]] = []
        for environment in selected_environments:
            for version in self._selected_versions():
                for trajectory in self._selected_trajectories():
                    for camera_name in self._selected_camera_names():
                        groups.append((environment, version, trajectory, camera_name))
        return [
            (str(environment), str(version), str(trajectory), str(camera_name))
            for environment, version, trajectory, camera_name in self._limit_group_keys(groups)
        ]

    def _discover_archive_units(self) -> list[TartanGroundArchiveUnit]:
        local_archive_root = self.dataset_config.options.get("local_archive_root")
        if local_archive_root:
            units = self._discover_archive_units_from_root(Path(str(local_archive_root)))
            if units:
                return units
        units = self._discover_archive_units_from_root(self.paths.raw)
        if units:
            return units
        return self._discover_archive_units_from_remote()

    def _discover_archive_units_from_root(self, root: Path) -> list[TartanGroundArchiveUnit]:
        if not root.exists():
            return []
        units: set[TartanGroundArchiveUnit] = set()
        for archive_path in root.rglob("*.zip"):
            try:
                relative_path = archive_path.relative_to(root)
            except ValueError:
                continue
            unit = self._parse_archive_unit_from_relative_path(relative_path)
            if unit is not None:
                units.add(unit)
        return sorted(units, key=lambda unit: (unit.environment, unit.version, unit.trajectory, unit.camera_name, unit.modality))

    def _discover_archive_units_from_remote(self) -> list[TartanGroundArchiveUnit]:
        units: set[TartanGroundArchiveUnit] = set()
        for relative_path in self._iter_remote_relative_repo_paths():
            unit = self._parse_archive_unit_from_relative_path(Path(relative_path))
            if unit is not None:
                units.add(unit)
        return sorted(units, key=lambda unit: (unit.environment, unit.version, unit.trajectory, unit.camera_name, unit.modality))

    def _parse_archive_unit_from_relative_path(self, relative_path: Path) -> TartanGroundArchiveUnit | None:
        parts = relative_path.parts
        if len(parts) != 4:
            return None
        environment, version_dir, trajectory, filename = parts
        if Path(filename).suffix != ".zip" or "_" not in Path(filename).stem:
            return None
        modality, camera_name = Path(filename).stem.split("_", 1)
        version = self._version_name_from_directory(version_dir)
        if version is None:
            return None
        return TartanGroundArchiveUnit(
            environment=environment,
            version=version,
            trajectory=trajectory,
            modality=modality,
            camera_name=camera_name,
        )

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

    def _version_name_from_directory(self, version_dir: str) -> str | None:
        mapping = {
            "Data_omni": "omni",
            "Data_diff": "diff",
            "Data_anymal": "anymal",
        }
        return mapping.get(version_dir)

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
