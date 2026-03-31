import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import TartanGroundPipeline
from depth_collector.datasets.tartanground import TartanGroundArchiveUnit, TartanGroundSourceItem


class TartanGroundPipelineTest(unittest.TestCase):
    def _make_config(
        self,
        root_data_dir: str,
        selection: object = "minimum_readable",
        process_ratio: float = 1.0,
    ) -> dict[str, object]:
        return {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 2,
                "process_ratio": process_ratio,
                "shuffle_seed": 0,
                "resume": True,
                "skip_known_errors": True,
                "write_error_traces": True,
                "target_shard_size_gb": 1.0,
            },
            "output": {
                "root_data_dir": root_data_dir,
                "raw_subdir_name": "raw",
                "processed_subdir_name": "processed",
                "state_subdir_name": "state",
                "metadata_filename": "metadata.json",
            },
            "datasets": {
                "tartanground": {
                    "enabled": True,
                    "hf_dataset_id": "theairlabcmu/TartanGround",
                    "selection": selection,
                    "environments": ["AbandonedCable"],
                    "versions": ["omni"],
                    "trajectories": ["P0000"],
                    "camera_names": ["lcam_front"],
                    "modalities": ["image"],
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str) -> TartanGroundPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir)))
        return TartanGroundPipeline(load_config(config_path), "tartanground")

    def _write_test_archive(self, pipeline: TartanGroundPipeline, unit: TartanGroundArchiveUnit) -> Path:
        archive_path = pipeline._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        image_relative_path = Path("image_lcam_front") / "000000_lcam_front.png"
        depth_relative_path = Path("depth_lcam_front") / "000000_lcam_front_depth.png"
        with tempfile.TemporaryDirectory() as build_dir:
            build_root = Path(build_dir)
            if unit.modality == "image":
                image = np.zeros((4, 6, 3), dtype=np.uint8)
                image[..., 0] = 32
                image[..., 1] = 128
                image[..., 2] = 224
                source_image_path = build_root / image_relative_path
                source_image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(source_image_path)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.write(source_image_path, arcname=str(image_relative_path))
            elif unit.modality == "depth":
                depth = np.full((4, 6), 2.0, dtype=np.float32)
                source_depth_path = build_root / depth_relative_path
                source_depth_path.parent.mkdir(parents=True, exist_ok=True)
                rgba = np.frombuffer(depth.astype("<f4").tobytes(), dtype=np.uint8).reshape(4, 6, 4)
                Image.fromarray(rgba, mode="RGBA").save(source_depth_path)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.write(source_depth_path, arcname=str(depth_relative_path))
            else:
                raise AssertionError(f"unsupported test modality: {unit.modality}")

        return archive_path

    def test_archive_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            units = list(pipeline.enumerate_download_units())
            self.assertEqual(
                units,
                [
                    TartanGroundArchiveUnit(
                        environment="AbandonedCable",
                        version="omni",
                        trajectory="P0000",
                        modality="image",
                        camera_name="lcam_front",
                    ),
                    TartanGroundArchiveUnit(
                        environment="AbandonedCable",
                        version="omni",
                        trajectory="P0000",
                        modality="depth",
                        camera_name="lcam_front",
                    ),
                ],
            )

    def test_all_selectors_discover_remote_archive_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.dataset_config.options["environments"] = "*"
            pipeline.dataset_config.options["versions"] = "*"
            pipeline.dataset_config.options["trajectories"] = "*"
            pipeline.dataset_config.options["camera_names"] = "*"
            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    "EnvB/Data_diff/P0001/image_lcam_right.zip",
                    "EnvA/Data_omni/P0000/image_lcam_front.zip",
                    "EnvA/Data_omni/P0000/depth_lcam_front.zip",
                ],
            ):
                units = list(pipeline.enumerate_download_units())
            self.assertEqual(
                units,
                [
                    TartanGroundArchiveUnit(
                        environment="EnvA",
                        version="omni",
                        trajectory="P0000",
                        modality="image",
                        camera_name="lcam_front",
                    ),
                    TartanGroundArchiveUnit(
                        environment="EnvA",
                        version="omni",
                        trajectory="P0000",
                        modality="depth",
                        camera_name="lcam_front",
                    ),
                ],
            )

    def test_minimum_readable_selection_applies_to_complete_group_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.dataset_config.options["versions"] = ["omni", "diff"]
            selected_units = list(pipeline.enumerate_download_units())
            self.assertEqual(
                selected_units,
                [
                    TartanGroundArchiveUnit(
                        environment="AbandonedCable",
                        version="omni",
                        trajectory="P0000",
                        modality="image",
                        camera_name="lcam_front",
                    ),
                    TartanGroundArchiveUnit(
                        environment="AbandonedCable",
                        version="omni",
                        trajectory="P0000",
                        modality="depth",
                        camera_name="lcam_front",
                    ),
                ],
            )

    def test_hub_repo_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="image",
                camera_name="lcam_front",
            )
            self.assertEqual(
                pipeline._hub_repo_filename(unit),
                "AbandonedCable/Data_omni/P0000/image_lcam_front.zip",
            )

    def test_minimum_readable_download_writes_single_member_archives_from_local_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.dataset_config.options["local_archive_root"] = source_dir
            source_root = Path(source_dir) / "AbandonedCable" / "Data_omni" / "P0000"
            source_root.mkdir(parents=True, exist_ok=True)

            image_archive = source_root / "image_lcam_front.zip"
            depth_archive = source_root / "depth_lcam_front.zip"
            with tempfile.TemporaryDirectory() as build_dir:
                build_root = Path(build_dir)
                for frame_id in ("000000_lcam_front", "000001_lcam_front"):
                    image = np.zeros((4, 6, 3), dtype=np.uint8)
                    image[..., 1] = 128
                    image[..., 2] = 224
                    image_path = build_root / "image_lcam_front" / f"{frame_id}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(image_path)
                    depth = np.full((4, 6), 2.0, dtype=np.float32)
                    depth_path = build_root / "depth_lcam_front" / f"{frame_id}_depth.png"
                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                    rgba = np.frombuffer(depth.astype("<f4").tobytes(), dtype=np.uint8).reshape(4, 6, 4)
                    Image.fromarray(rgba, mode="RGBA").save(depth_path)
                with zipfile.ZipFile(image_archive, "w") as archive:
                    archive.write(
                        build_root / "image_lcam_front" / "000000_lcam_front.png",
                        arcname="image_lcam_front/000000_lcam_front.png",
                    )
                    archive.write(
                        build_root / "image_lcam_front" / "000001_lcam_front.png",
                        arcname="image_lcam_front/000001_lcam_front.png",
                    )
                with zipfile.ZipFile(depth_archive, "w") as archive:
                    archive.write(
                        build_root / "depth_lcam_front" / "000000_lcam_front_depth.png",
                        arcname="depth_lcam_front/000000_lcam_front_depth.png",
                    )
                    archive.write(
                        build_root / "depth_lcam_front" / "000001_lcam_front_depth.png",
                        arcname="depth_lcam_front/000001_lcam_front_depth.png",
                    )

            image_unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="image",
                camera_name="lcam_front",
            )
            depth_unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="depth",
                camera_name="lcam_front",
            )
            pipeline.download_unit(image_unit)
            pipeline.download_unit(depth_unit)

            with zipfile.ZipFile(pipeline._archive_path(image_unit)) as archive:
                self.assertEqual(archive.namelist(), ["image_lcam_front/000000_lcam_front.png"])
            with zipfile.ZipFile(pipeline._archive_path(depth_unit)) as archive:
                self.assertEqual(archive.namelist(), ["depth_lcam_front/000000_lcam_front_depth.png"])

    def test_extract_and_enumerate_real_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="image",
                camera_name="lcam_front",
            )
            depth_unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="depth",
                camera_name="lcam_front",
            )
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    TartanGroundSourceItem(
                        environment="AbandonedCable",
                        version="omni",
                        trajectory="P0000",
                        camera_name="lcam_front",
                        image_relative_path="image_lcam_front/000000_lcam_front.png",
                        depth_relative_path="depth_lcam_front/000000_lcam_front_depth.png",
                    )
                ],
            )

    def test_pairing_key_normalizes_embedded_modality_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_key = pipeline._paired_ground_key("image_lcam_front/000000_lcam_front.png")
            depth_key = pipeline._paired_ground_key("depth_lcam_front/000000_lcam_front_depth.png")
            self.assertEqual(image_key, "000000_lcam_front")
            self.assertEqual(depth_key, "000000_lcam_front")

    def test_build_sample_uses_shared_tartan_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="image",
                camera_name="lcam_front",
            )
            depth_unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="depth",
                camera_name="lcam_front",
            )
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)
            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (4, 6, 3))
            self.assertEqual(sample.distance.shape, (4, 6, 1))
            self.assertEqual(sample.ray_dir.shape, (4, 6, 3))
            self.assertEqual(sample.provenance["version"], "omni")
            self.assertEqual(sample.provenance["trajectory"], "P0000")
            self.assertEqual(sample.provenance["camera_name"], "lcam_front")


if __name__ == "__main__":
    unittest.main()
