import json
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import TartanAirPipeline
import depth_collector.datasets.tartanair as tartanair_module
import depth_collector.datasets.tartan as tartan_module
from depth_collector.datasets.tartanair import TartanAirArchiveUnit, TartanAirSourceItem
from depth_collector.geometry import PinholeCameraModel


class TartanAirPipelineTest(unittest.TestCase):
    def _make_config(
        self,
        root_data_dir: str,
        environment_count: int = 1,
        process_ratio: float = 0.01,
        shuffle_seed: int = 0,
    ) -> dict[str, object]:
        return {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "process_ratio": process_ratio,
                "shuffle_seed": shuffle_seed,
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
                "tartanair": {
                    "enabled": True,
                    "hf_dataset_id": "theairlabcmu/tartanair",
                    "environments": ["neighborhood"],
                    "environment_count": environment_count,
                    "difficulties": ["Easy"],
                    "modalities": ["image_left"],
                }
            },
        }

    def _make_pipeline(
        self,
        tmp_dir: str,
        environment_count: int = 1,
        process_ratio: float = 1.0,
        shuffle_seed: int = 0,
    ) -> TartanAirPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(
            json.dumps(
                self._make_config(
                    tmp_dir,
                    environment_count=environment_count,
                    process_ratio=process_ratio,
                    shuffle_seed=shuffle_seed,
                )
            )
        )
        return TartanAirPipeline(load_config(config_path), "tartanair")

    def _write_test_archive(self, pipeline: TartanAirPipeline, unit: TartanAirArchiveUnit) -> Path:
        archive_path = pipeline._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        image_relative_path = Path("P000") / "000000_left.png"
        depth_relative_path = Path("P000") / "000000_left_depth.npy"
        non_image_path = Path("P000") / "notes.txt"
        with tempfile.TemporaryDirectory() as build_dir:
            build_root = Path(build_dir)
            if unit.modality == "image_left":
                image = np.zeros((4, 6, 3), dtype=np.uint8)
                image[..., 0] = 64
                image[..., 1] = 128
                image[..., 2] = 255
                source_image_path = build_root / image_relative_path
                source_image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(source_image_path)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.write(source_image_path, arcname=str(image_relative_path))
                    archive.writestr(str(non_image_path), "ignore me")
            elif unit.modality == "depth_left":
                depth = np.full((4, 6), 2.0, dtype=np.float32)
                source_depth_path = build_root / depth_relative_path
                source_depth_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(source_depth_path, depth)
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
                    TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left"),
                ],
            )

    def test_environment_count_limits_selected_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, environment_count=2)
            pipeline.dataset_config.options["environments"] = ["env_a", "env_b", "env_c"]
            selected_units = pipeline._iter_selected_download_units()
            self.assertEqual(len(selected_units), 4)
            self.assertEqual(
                selected_units,
                [
                    TartanAirArchiveUnit(environment="env_a", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="env_a", difficulty="Easy", modality="depth_left"),
                    TartanAirArchiveUnit(environment="env_b", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="env_b", difficulty="Easy", modality="depth_left"),
                ],
            )

    def test_all_environment_selector_discovers_remote_environments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, environment_count=2)
            pipeline.dataset_config.options["environments"] = "*"
            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    "env_b/Easy/image_left.zip",
                    "env_a/Easy/image_left.zip",
                    "env_c/Hard/depth_left.zip",
                ],
            ):
                selected_units = pipeline._iter_selected_download_units()
            self.assertEqual(
                selected_units,
                [
                    TartanAirArchiveUnit(environment="env_a", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="env_a", difficulty="Easy", modality="depth_left"),
                    TartanAirArchiveUnit(environment="env_b", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="env_b", difficulty="Easy", modality="depth_left"),
                ],
            )

    def test_all_environment_selector_ignores_non_archive_repo_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, environment_count=1)
            pipeline.dataset_config.options["environments"] = "*"
            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    ".gitattributes",
                    "README.md",
                    "neighborhood/Easy/image_left.zip",
                ],
            ):
                selected_units = pipeline._iter_selected_download_units()
            self.assertEqual(
                selected_units,
                [
                    TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left"),
                ],
            )

    def test_environment_count_applies_to_environment_difficulty_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, environment_count=1)
            pipeline.dataset_config.options["difficulties"] = ["Easy", "Hard"]
            selected_units = pipeline._iter_selected_download_units()
            self.assertEqual(
                selected_units,
                [
                    TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left"),
                    TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left"),
                ],
            )

    def test_local_environment_discovery_ignores_hidden_cache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, environment_count=1)
            pipeline.dataset_config.options["environments"] = "*"
            (pipeline.paths.raw / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
            (pipeline.paths.raw / "abandonedfactory").mkdir(parents=True, exist_ok=True)
            self.assertEqual(pipeline._selected_environments(), ["abandonedfactory"])

    def test_source_item_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            item = TartanAirSourceItem(
                environment="neighborhood",
                difficulty="Easy",
                image_relative_path="P000/image.png",
                depth_relative_path="P000/image_depth.npy",
            )
            self.assertEqual(
                pipeline.get_source_item_id(item),
                "neighborhood/Easy/P000/image.png",
            )

    def test_pairing_key_normalizes_embedded_modality_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_key = pipeline._paired_relative_key(
                "neighborhood/Easy/P000/image_left/000000_left.png",
                environment="neighborhood",
                difficulty="Easy",
            )
            depth_key = pipeline._paired_relative_key(
                "neighborhood/Easy/P000/depth_left/000000_left_depth.npy",
                environment="neighborhood",
                difficulty="Easy",
            )
            self.assertEqual(image_key, "P000/000000_left")
            self.assertEqual(depth_key, "P000/000000_left")

    def test_download_unit_fetches_archive_to_expected_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")

            with tempfile.TemporaryDirectory() as download_dir:
                downloaded_path = Path(download_dir) / "image_left.zip"
                downloaded_path.write_bytes(b"archive-bytes")
                pipeline._download_archive_from_hub = lambda _unit: downloaded_path  # type: ignore[method-assign]

                pipeline.download_unit(unit)

            archive_path = pipeline._archive_path(unit)
            self.assertEqual(archive_path.read_bytes(), b"archive-bytes")

    def test_download_stage_repairs_missing_archive_even_if_state_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            unit_id = pipeline.get_download_unit_id(unit)
            pipeline.prepare_directories()
            pipeline.download_state.mark_complete(unit_id)

            with tempfile.TemporaryDirectory() as download_dir:
                downloaded_path = Path(download_dir) / "image_left.zip"
                downloaded_path.write_bytes(b"archive-bytes")
                pipeline._download_archive_from_hub = lambda _unit: downloaded_path  # type: ignore[method-assign]
                pipeline.run_download_stage()

            self.assertTrue(pipeline._archive_path(unit).exists())
            self.assertEqual(pipeline._archive_path(unit).read_bytes(), b"archive-bytes")

    def test_camera_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            model = pipeline.build_camera_model(
                TartanAirSourceItem("neighborhood", "Easy", "P000/image.png", "P000/image_depth.npy"),
                {"image": np.zeros((640, 640, 3), dtype=np.float32)},
            )
            self.assertIsInstance(model, PinholeCameraModel)
            self.assertEqual((model.width, model.height), (640, 640))
            self.assertEqual((model.fx, model.cx), (320.0, 320.0))

    def test_extract_and_enumerate_real_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    TartanAirSourceItem(
                        environment="neighborhood",
                        difficulty="Easy",
                        image_relative_path="P000/000000_left.png",
                        depth_relative_path="P000/000000_left_depth.npy",
                    )
                ],
            )

    def test_enumeration_uses_cached_manifest_on_second_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            first_items = list(pipeline.enumerate_source_items())

            def _fail_scan(**_: object) -> tuple[list[TartanAirSourceItem], list[tuple[str, str]]]:
                raise AssertionError("expected cached enumeration manifest to be reused")

            pipeline._scan_group_source_items = _fail_scan  # type: ignore[method-assign]
            second_items = list(pipeline.enumerate_source_items())

            self.assertEqual(second_items, first_items)

    def test_cached_manifest_replay_still_yields_items_in_compact_tty_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            first_items = list(pipeline.enumerate_source_items())

            tartan_module.sys.stdout.isatty = lambda: True  # type: ignore[method-assign]
            try:
                pipeline.verbose = False
                second_items = list(pipeline.enumerate_source_items())
            finally:
                tartan_module.sys.stdout.isatty = sys.stdout.isatty  # type: ignore[method-assign]

            self.assertEqual(second_items, first_items)

    def test_extraction_stage_repairs_missing_extracted_files_even_if_state_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            self._write_test_archive(pipeline, image_unit)
            pipeline.prepare_directories()
            pipeline.extraction_state.mark_complete(pipeline.get_extraction_unit_id(image_unit))

            pipeline.run_extraction_cleanup_stage()

            extracted_image = pipeline._extracted_dir(image_unit) / "P000" / "000000_left.png"
            self.assertTrue(extracted_image.exists())
            self.assertFalse(pipeline._archive_path(image_unit).exists())

    def test_load_source_item_and_build_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)
            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            item = next(iter(pipeline.enumerate_source_items()))

            loaded = pipeline.load_source_item(item)
            sample = pipeline.build_sample(item, loaded, pipeline.build_camera_model(item, loaded))

            self.assertEqual(sample.image.shape, (4, 6, 3))
            self.assertEqual(sample.distance.shape, (4, 6, 1))
            self.assertEqual(sample.ray_dir.shape, (4, 6, 3))
            self.assertEqual(sample.image.dtype, np.float32)
            self.assertTrue(np.all(sample.image >= 0.0))
            self.assertTrue(np.all(sample.image <= 1.0))
            self.assertTrue(np.all(sample.distance >= 2.0))
            self.assertTrue(np.allclose(np.linalg.norm(sample.ray_dir, axis=-1), 1.0, atol=1e-6))
            expected_distance = 2.0 / sample.ray_dir[..., 2:3]
            self.assertTrue(np.allclose(sample.distance, expected_distance, atol=1e-6))

    def test_validator_rejects_image_values_outside_zero_one_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)
            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            item = next(iter(pipeline.enumerate_source_items()))

            loaded = pipeline.load_source_item(item)
            sample = pipeline.build_sample(item, loaded, pipeline.build_camera_model(item, loaded))
            sample.image[0, 0, 0] = 1.5
            report = pipeline.validator.validate(sample)

            self.assertFalse(report.valid)
            self.assertTrue(any(issue.code == "image_range_high" for issue in report.issues))

    def test_build_sample_rejects_distance_equal_to_source_z_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)
            pipeline.extract_unit(image_unit)
            pipeline.extract_unit(depth_unit)
            item = next(iter(pipeline.enumerate_source_items()))
            loaded = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded)

            original_converter = tartan_module.z_depth_to_distance
            tartan_module.z_depth_to_distance = lambda depth, _ray_dir: depth.astype(np.float32)
            try:
                with self.assertRaises(ValueError):
                    pipeline.build_sample(item, loaded, camera_model)
            finally:
                tartan_module.z_depth_to_distance = original_converter

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.prepare_directories()
            pipeline.run_extraction_stage()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()
            pipeline.validate_output()

            shard_paths = sorted(pipeline.paths.processed_files.glob("*.tar"))
            self.assertEqual(len(shard_paths), 1)

            with tarfile.open(shard_paths[0], "r") as archive:
                members = archive.getmembers()
                member_names = sorted(member.name for member in members)
                self.assertEqual(
                    member_names,
                    [
                        "neighborhood__Easy__P000__000000_left.png.distance.pt",
                        "neighborhood__Easy__P000__000000_left.png.image.pt",
                        "neighborhood__Easy__P000__000000_left.png.meta.json",
                        "neighborhood__Easy__P000__000000_left.png.ray_dir.pt",
                    ],
                )

                image_payload = archive.extractfile("neighborhood__Easy__P000__000000_left.png.image.pt")
                distance_payload = archive.extractfile("neighborhood__Easy__P000__000000_left.png.distance.pt")
                ray_dir_payload = archive.extractfile("neighborhood__Easy__P000__000000_left.png.ray_dir.pt")
                meta_payload = archive.extractfile("neighborhood__Easy__P000__000000_left.png.meta.json")
                assert image_payload is not None
                assert distance_payload is not None
                assert ray_dir_payload is not None
                assert meta_payload is not None
                image_tensor = torch.load(io.BytesIO(image_payload.read()), weights_only=False)
                distance_tensor = torch.load(io.BytesIO(distance_payload.read()), weights_only=False)
                ray_dir_tensor = torch.load(io.BytesIO(ray_dir_payload.read()), weights_only=False)
                meta = json.loads(meta_payload.read())

            self.assertEqual(tuple(image_tensor.shape), (4, 6, 3))
            self.assertEqual(tuple(distance_tensor.shape), (4, 6, 1))
            self.assertEqual(tuple(ray_dir_tensor.shape), (4, 6, 3))
            self.assertEqual(meta["sample_id"], "neighborhood/Easy/P000/000000_left.png")

            metadata = json.loads(pipeline.paths.metadata.read_text())
            self.assertIn("created_at", metadata)
            self.assertEqual(metadata["available_source_item_count"], 1)
            self.assertEqual(metadata["selected_source_item_count"], 1)
            self.assertEqual(metadata["skipped_by_process_ratio_count"], 0)
            self.assertEqual(metadata["shard_count"], 1)
            self.assertEqual(metadata["valid_sample_count"], 1)
            self.assertEqual(metadata["invalid_sample_count"], 0)
            self.assertEqual(metadata["processing_error_count"], 0)
            self.assertEqual(metadata["pairing_error_count"], 0)
            self.assertEqual(metadata["samples_per_shard"], {shard_paths[0].name: 1})
            self.assertEqual(metadata["suggested_train_shards"], [shard_paths[0].name])
            self.assertEqual(metadata["suggested_val_shards"], [shard_paths[0].name])
            run_report = json.loads(pipeline.paths.run_report.read_text())
            self.assertEqual(run_report["dataset"], "tartanair")
            self.assertEqual(run_report["shard_count"], 1)
            self.assertEqual(run_report["valid_sample_count"], 1)
            self.assertEqual(run_report["error_stage_counts"], {})
            self.assertEqual(run_report["run_stats"]["valid_sample_count"], 1)

    def test_validate_output_rejects_mismatched_valid_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.prepare_directories()
            pipeline.run_extraction_stage()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()

            metadata = json.loads(pipeline.paths.metadata.read_text())
            metadata["valid_sample_count"] = 99
            pipeline.paths.metadata.write_text(json.dumps(metadata))

            with self.assertRaisesRegex(ValueError, "valid_sample_count"):
                pipeline.validate_output()

    def test_validate_output_rejects_missing_referenced_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.prepare_directories()
            pipeline.run_extraction_stage()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()

            shard_path = next(pipeline.paths.processed_files.glob("*.tar"))
            shard_path.unlink()

            with self.assertRaisesRegex(ValueError, "missing shard file"):
                pipeline.validate_output()

    def test_process_ratio_still_processes_one_item_for_tiny_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, process_ratio=1e-12, shuffle_seed=0)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            self._write_test_archive(pipeline, depth_unit)

            pipeline.prepare_directories()
            pipeline.run_extraction_stage()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()

            shard_paths = sorted(pipeline.paths.processed_files.glob("*.tar"))
            self.assertEqual(len(shard_paths), 2)
            self.assertEqual([path.name for path in shard_paths], ["shard-000000.tar", "shard-000001.tar"])

            metadata = json.loads(pipeline.paths.metadata.read_text())
            self.assertEqual(metadata["available_source_item_count"], 1)
            self.assertEqual(metadata["selected_source_item_count"], 1)
            self.assertEqual(metadata["skipped_by_process_ratio_count"], 0)
            self.assertEqual(metadata["valid_sample_count"], 1)
            self.assertEqual(metadata["shard_count"], 2)
            self.assertEqual(metadata["suggested_train_shards"], [shard_paths[0].name])
            self.assertEqual(metadata["suggested_val_shards"], [shard_paths[1].name])

    def test_process_ratio_is_deterministic_for_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline_a = self._make_pipeline(tmp_dir, process_ratio=0.5, shuffle_seed=7)
            pipeline_b = self._make_pipeline(tmp_dir, process_ratio=0.5, shuffle_seed=7)
            item_id = "neighborhood/Easy/P000/000000_left.png"

            self.assertEqual(
                pipeline_a._should_process_item(item_id),
                pipeline_b._should_process_item(item_id),
            )

    def test_missing_pair_records_enumeration_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            self._write_test_archive(pipeline, image_unit)

            pipeline.prepare_directories()
            pipeline.extract_unit(image_unit)
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()

            metadata = json.loads(pipeline.paths.metadata.read_text())
            self.assertEqual(metadata["available_source_item_count"], 0)
            self.assertEqual(metadata["pairing_error_count"], 1)

            error_lines = (pipeline.paths.state / "errors.jsonl").read_text().splitlines()
            self.assertEqual(len(error_lines), 1)
            payload = json.loads(error_lines[0])
            self.assertEqual(payload["stage"], "enumeration")
            self.assertIn("missing extracted depth directory", payload["error_message"])
            run_report = json.loads(pipeline.paths.run_report.read_text())
            self.assertEqual(run_report["error_stage_counts"], {"enumeration": 1})
            self.assertEqual(len(run_report["recent_errors"]), 1)

    def test_missing_pair_error_is_not_duplicated_across_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            self._write_test_archive(pipeline, image_unit)

            pipeline.prepare_directories()
            pipeline.extract_unit(image_unit)
            list(pipeline.iter_valid_samples())
            list(pipeline.iter_valid_samples())

            error_lines = (pipeline.paths.state / "errors.jsonl").read_text().splitlines()
            self.assertEqual(len(error_lines), 1)
            payload = json.loads(error_lines[0])
            self.assertEqual(payload["stage"], "enumeration")

    def test_decode_failure_records_processing_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, depth_unit)

            archive_path = pipeline._archive_path(image_unit)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("P000/000000_left.png", b"not-a-real-image")

            pipeline.prepare_directories()
            pipeline.run_extraction_stage()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()

            metadata = json.loads(pipeline.paths.metadata.read_text())
            self.assertEqual(metadata["available_source_item_count"], 1)
            self.assertEqual(metadata["selected_source_item_count"], 1)
            self.assertEqual(metadata["valid_sample_count"], 0)
            self.assertEqual(metadata["processing_error_count"], 1)
            self.assertEqual(metadata["shard_count"], 0)

            error_lines = (pipeline.paths.state / "errors.jsonl").read_text().splitlines()
            self.assertEqual(len(error_lines), 1)
            payload = json.loads(error_lines[0])
            self.assertEqual(payload["stage"], "processing")
            self.assertIn("cannot identify image file", payload["error_message"])
            self.assertIsNotNone(payload["traceback_text"])
            run_report = json.loads(pipeline.paths.run_report.read_text())
            self.assertEqual(run_report["error_stage_counts"], {"processing": 1})

    def test_download_failure_records_error_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            with tempfile.TemporaryDirectory() as source_dir:
                source_pipeline = self._make_pipeline(source_dir)
                image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
                source_archive_path = self._write_test_archive(source_pipeline, image_unit)

                def fake_download(unit: TartanAirArchiveUnit) -> Path:
                    if unit.modality == "depth_left":
                        raise FileNotFoundError("missing remote depth archive")
                    return source_archive_path

                pipeline._download_archive_from_hub = fake_download  # type: ignore[method-assign]

                pipeline.prepare_directories()
                pipeline.run_download_stage()
                pipeline.build_metadata()
                pipeline.build_run_report()

                metadata = json.loads(pipeline.paths.metadata.read_text())
                self.assertEqual(metadata["selected_download_unit_count"], 2)
                self.assertEqual(metadata["download_error_count"], 1)
                self.assertTrue((pipeline._archive_path(image_unit)).exists())
                self.assertFalse((pipeline._archive_path(TartanAirArchiveUnit("neighborhood", "Easy", "depth_left"))).exists())

                error_lines = (pipeline.paths.state / "errors.jsonl").read_text().splitlines()
                self.assertEqual(len(error_lines), 1)
                payload = json.loads(error_lines[0])
                self.assertEqual(payload["stage"], "download")
                self.assertIn("missing remote depth archive", payload["error_message"])
                run_report = json.loads(pipeline.paths.run_report.read_text())
                self.assertEqual(run_report["error_stage_counts"], {"download": 1})

    def test_extraction_failure_records_error_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            depth_unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="depth_left")
            self._write_test_archive(pipeline, image_unit)
            broken_depth_archive = pipeline._archive_path(depth_unit)
            broken_depth_archive.parent.mkdir(parents=True, exist_ok=True)
            broken_depth_archive.write_bytes(b"not-a-zip")

            pipeline.prepare_directories()
            pipeline.run_extraction_stage()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()

            metadata = json.loads(pipeline.paths.metadata.read_text())
            self.assertEqual(metadata["selected_extraction_unit_count"], 2)
            self.assertEqual(metadata["extraction_error_count"], 1)
            self.assertEqual(metadata["available_source_item_count"], 0)
            self.assertEqual(metadata["pairing_error_count"], 1)

            error_lines = (pipeline.paths.state / "errors.jsonl").read_text().splitlines()
            self.assertEqual(len(error_lines), 2)
            payloads = [json.loads(line) for line in error_lines]
            self.assertEqual(payloads[0]["stage"], "extraction")
            self.assertIn("zip", payloads[0]["error_message"].lower())
            self.assertEqual(payloads[1]["stage"], "enumeration")
            self.assertIn("missing paired depth_left file", payloads[1]["error_message"])
            run_report = json.loads(pipeline.paths.run_report.read_text())
            self.assertEqual(run_report["error_stage_counts"], {"enumeration": 1, "extraction": 1})


if __name__ == "__main__":
    unittest.main()
