import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import VirtualKITTI2Pipeline
from depth_collector.datasets.virtual_kitti_2 import VirtualKITTI2ArchiveUnit, VirtualKITTI2SourceItem


class VirtualKITTI2PipelineTest(unittest.TestCase):
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
                "virtual_kitti_2": {
                    "enabled": True,
                    "hf_dataset_id": "ZhengGuangze/VKITTI2_vlbm",
                    "selection": selection,
                    "archive_filename": "vkitti2_vlbm.tar.gz",
                    "sequences": "*",
                    "depth_semantics": "distance",
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str, *, selection: object = "minimum_readable") -> VirtualKITTI2Pipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir, selection=selection)))
        return VirtualKITTI2Pipeline(load_config(config_path), "virtual_kitti_2")

    def _write_sequence_tree(
        self,
        root: Path,
        *,
        sequence_name: str = "Scene06_fog",
        frame_id: str = "00000",
    ) -> None:
        dataset_root = root / "vkitti2_vlbm"
        rgb_root = dataset_root / sequence_name / "rgbs"
        depth_root = dataset_root / sequence_name / "depths"
        rgb_root.mkdir(parents=True, exist_ok=True)
        depth_root.mkdir(parents=True, exist_ok=True)

        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[..., 1] = 128
        image[..., 2] = 255
        Image.fromarray(image).save(rgb_root / f"rgb_{frame_id}.jpg")

        depth = np.full((6, 8), 12.0, dtype=np.float32)
        depth[0, 0] = 0.0
        np.savez(depth_root / f"depth_{frame_id}.npz", depth=depth)

        intrinsics = np.zeros((1, 3, 3), dtype=np.float32)
        intrinsics[0, 0, 0] = 100.0
        intrinsics[0, 1, 1] = 110.0
        intrinsics[0, 0, 2] = 4.0
        intrinsics[0, 1, 2] = 3.0
        intrinsics[0, 2, 2] = 1.0
        extrinsics = np.eye(4, dtype=np.float32)[None, ...]
        np.save(dataset_root / sequence_name / "intrinsics.npy", intrinsics)
        np.save(dataset_root / sequence_name / "extrinsics.npy", extrinsics)
        (dataset_root / sequence_name / "scene_info.json").write_text(json.dumps({"weather": "fog", "scene": "Scene06"}))

    def test_all_sequence_selector_discovers_remote_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")])

    def test_remote_download_requests_archive_file_for_all_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, selection="all")
            unit = VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")
            downloaded_archive = pipeline.paths.raw / "cache" / "vkitti2_vlbm.tar.gz"
            downloaded_archive.parent.mkdir(parents=True, exist_ok=True)
            downloaded_archive.write_bytes(b"test")
            with patch.object(pipeline, "hf_hub_download") as download_mock:
                download_mock.return_value = str(downloaded_archive)
                pipeline.download_unit(unit)
            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["filename"], "vkitti2_vlbm.tar.gz")

    def test_minimum_readable_download_materializes_single_sample_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            extracted_root = Path(source_dir) / "extracted"
            self._write_sequence_tree(extracted_root, frame_id="00000")
            sequence_root = extracted_root / "vkitti2_vlbm" / "Scene06_fog"
            image = np.zeros((6, 8, 3), dtype=np.uint8)
            image[..., 0] = 255
            Image.fromarray(image).save(sequence_root / "rgbs" / "rgb_00001.jpg")
            depth = np.full((6, 8), 13.0, dtype=np.float32)
            np.savez(sequence_root / "depths" / "depth_00001.npz", depth=depth)
            intrinsics = np.zeros((2, 3, 3), dtype=np.float32)
            intrinsics[0, 0, 0] = 100.0
            intrinsics[0, 1, 1] = 110.0
            intrinsics[0, 0, 2] = 4.0
            intrinsics[0, 1, 2] = 3.0
            intrinsics[0, 2, 2] = 1.0
            intrinsics[1] = intrinsics[0]
            extrinsics = np.eye(4, dtype=np.float32)[None, ...].repeat(2, axis=0)
            np.save(sequence_root / "intrinsics.npy", intrinsics)
            np.save(sequence_root / "extrinsics.npy", extrinsics)

            archive_path = Path(source_dir) / "vkitti2_vlbm.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(extracted_root / "vkitti2_vlbm", arcname="vkitti2_vlbm")

            pipeline.dataset_config.options["local_archive_root"] = source_dir
            unit = VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")
            pipeline.download_unit(unit)

            with tarfile.open(pipeline._archive_path(), "r:gz") as archive:
                member_names = sorted(member.name for member in archive if member.isfile())
            self.assertEqual(
                member_names,
                [
                    "vkitti2_vlbm/Scene06_fog/depths/depth_00000.npz",
                    "vkitti2_vlbm/Scene06_fog/extrinsics.npy",
                    "vkitti2_vlbm/Scene06_fog/frame_index_map.json",
                    "vkitti2_vlbm/Scene06_fog/intrinsics.npy",
                    "vkitti2_vlbm/Scene06_fog/rgbs/rgb_00000.jpg",
                    "vkitti2_vlbm/Scene06_fog/scene_info.json",
                ],
            )

    def test_minimum_readable_remote_download_uses_streaming_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            extracted_root = Path(source_dir) / "extracted"
            self._write_sequence_tree(extracted_root)
            archive_path = Path(source_dir) / "vkitti2_vlbm.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(extracted_root / "vkitti2_vlbm", arcname="vkitti2_vlbm")
            archive_bytes = archive_path.read_bytes()

            class _RemoteBytes(io.BytesIO):
                def __enter__(self) -> "_RemoteBytes":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    self.close()

            unit = VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")
            with patch.object(pipeline, "hf_open_remote_file", return_value=_RemoteBytes(archive_bytes)) as mocked_open:
                pipeline.download_unit(unit)

            mocked_open.assert_called_once()
            self.assertTrue(pipeline._archive_path().exists())

    def test_extract_archive_materializes_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            extracted_root = Path(source_dir) / "extracted"
            self._write_sequence_tree(extracted_root)

            archive_path = Path(source_dir) / "vkitti2_vlbm.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(extracted_root / "vkitti2_vlbm", arcname="vkitti2_vlbm")
            pipeline.dataset_config.options["local_archive_root"] = source_dir

            unit = VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")
            pipeline.download_unit(unit)
            pipeline.extract_unit(unit)

            self.assertTrue((pipeline.paths.raw / "vkitti2_vlbm" / "Scene06_fog" / "rgbs" / "rgb_00000.jpg").exists())

    def test_extraction_not_marked_satisfied_by_archive_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.paths.raw.mkdir(parents=True, exist_ok=True)
            (pipeline.paths.raw / "vkitti2_vlbm.tar.gz").write_bytes(b"placeholder")
            unit = VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")

            self.assertFalse(pipeline.is_extraction_unit_satisfied(unit))

    def test_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_sequence_tree(pipeline.paths.raw)

            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    VirtualKITTI2SourceItem(
                        sequence_name="Scene06_fog",
                        frame_id="00000",
                        frame_index=0,
                        array_index=0,
                        image_relative_path="Scene06_fog/rgbs/rgb_00000.jpg",
                        depth_relative_path="Scene06_fog/depths/depth_00000.npz",
                    )
                ],
            )

    def test_build_sample_uses_metric_distance_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_sequence_tree(pipeline.paths.raw)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (6, 8, 3))
            self.assertEqual(sample.distance.shape, (6, 8, 1))
            self.assertEqual(sample.ray_dir.shape, (6, 8, 3))
            self.assertAlmostEqual(float(sample.distance[1, 1, 0]), 12.0, places=5)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)
            self.assertEqual(sample.provenance["sequence_name"], "Scene06_fog")
            self.assertEqual(sample.provenance["projection"], "pinhole")
            self.assertEqual(sample.provenance["camera_axes"]["z"], "forward")

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_sequence_tree(pipeline.paths.raw)

            pipeline.prepare_directories()
            pipeline.write_samples(pipeline.iter_valid_samples())
            pipeline.build_metrics_summary()
            pipeline.build_metadata()
            pipeline.build_run_report()
            pipeline.validate_output()

            shard_paths = sorted(pipeline.paths.processed_files.glob("*.tar"))
            self.assertEqual(len(shard_paths), 1)
            metadata = json.loads(pipeline.paths.metadata.read_text())
            self.assertEqual(metadata["valid_sample_count"], 1)
            self.assertEqual(metadata["shard_count"], 1)


if __name__ == "__main__":
    unittest.main()
