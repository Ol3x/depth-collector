import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import NYUDepthV2Pipeline
from depth_collector.datasets.nyu_depth_v2 import NYUDepthV2ArchiveUnit, NYUDepthV2SourceItem


class NYUDepthV2PipelineTest(unittest.TestCase):
    def _make_config(self, root_data_dir: str, *, selection: object = "minimum_readable") -> dict[str, object]:
        return {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 1,
                "process_ratio": 1.0,
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
                "nyu_depth_v2": {
                    "enabled": True,
                    "hf_dataset_id": "sayakpaul/nyu_depth_v2",
                    "selection": selection,
                    "splits": ["val"],
                    "depth_semantics": "z_depth",
                    "camera_intrinsics": {
                        "width": 640.0,
                        "height": 480.0,
                        "fx": 518.8579,
                        "fy": 519.4696,
                        "cx": 325.5824,
                        "cy": 253.7362,
                    },
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str, *, selection: object = "minimum_readable") -> NYUDepthV2Pipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir, selection=selection)))
        return NYUDepthV2Pipeline(load_config(config_path), "nyu_depth_v2")

    def _write_h5(self, path: Path, image: np.ndarray, depth: np.ndarray) -> None:
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("rgb", data=image)
            handle.create_dataset("depth", data=depth)

    def _write_source_shard(self, archive_path: Path, *, member_name: str, image: np.ndarray, depth: np.ndarray) -> None:
        with tempfile.TemporaryDirectory() as scratch_dir:
            h5_path = Path(scratch_dir) / member_name
            self._write_h5(h5_path, image=image, depth=depth)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "w") as archive:
                archive.add(h5_path, arcname=member_name)

    def test_minimum_readable_download_uses_single_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            local_root = Path(source_dir) / "data"
            first_image = np.zeros((3, 4, 6), dtype=np.uint8)
            first_depth = np.full((4, 6), 2.0, dtype=np.float32)
            second_image = np.ones((3, 4, 6), dtype=np.uint8)
            second_depth = np.full((4, 6), 3.0, dtype=np.float32)
            self._write_source_shard(
                local_root / "val-000001.tar",
                member_name="sample_0001.h5",
                image=first_image,
                depth=first_depth,
            )
            self._write_source_shard(
                local_root / "val-000002.tar",
                member_name="sample_0002.h5",
                image=second_image,
                depth=second_depth,
            )
            pipeline.dataset_config.options["local_archive_root"] = source_dir

            units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [NYUDepthV2ArchiveUnit(repo_path="data/val-000001.tar")])
            pipeline.download_unit(units[0])

            self.assertTrue((pipeline.paths.raw / "_downloads" / "data" / "val-000001.tar").exists())
            self.assertFalse((pipeline.paths.raw / "_downloads" / "data" / "val-000002.tar").exists())

    def test_extract_and_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6)
            depth = np.full((4, 6), 2.0, dtype=np.float32)
            archive_path = Path(source_dir) / "data" / "val-000001.tar"
            self._write_source_shard(archive_path, member_name="fold/sample_0001.h5", image=image, depth=depth)
            pipeline.dataset_config.options["local_archive_root"] = source_dir

            unit = NYUDepthV2ArchiveUnit(repo_path="data/val-000001.tar")
            pipeline.download_unit(unit)
            pipeline.extract_unit(unit)

            items = list(pipeline.enumerate_source_items())
            self.assertEqual(
                items,
                [
                    NYUDepthV2SourceItem(
                        shard_name="val-000001.tar",
                        extracted_root_name="val-000001",
                        relative_path="fold/sample_0001.h5",
                    )
                ],
            )

    def test_build_sample_converts_z_depth_and_sanitizes_invalid_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image = np.zeros((3, 4, 6), dtype=np.uint8)
            image[1, :, :] = 255
            depth = np.full((4, 6), 2.0, dtype=np.float32)
            depth[0, 0] = 0.0

            sample_path = pipeline.paths.raw / "_extracted" / "val-000001" / "sample_0001.h5"
            self._write_h5(sample_path, image=image, depth=depth)

            item = NYUDepthV2SourceItem(
                shard_name="val-000001.tar",
                extracted_root_name="val-000001",
                relative_path="sample_0001.h5",
            )
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (4, 6, 3))
            self.assertEqual(sample.distance.shape, (4, 6, 1))
            self.assertEqual(sample.ray_dir.shape, (4, 6, 3))
            self.assertAlmostEqual(float(sample.distance[2, 3, 0]), 2.0, places=2)
            self.assertGreater(float(sample.distance[0, 1, 0]), 2.0)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)
            self.assertEqual(sample.provenance["projection"], "pinhole")

    def test_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, selection="all")
            pipeline.dataset_config.options["shards"] = ["val-000001.tar"]
            cached = pipeline.paths.raw / "cache" / "val-000001.tar"
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"test")
            with patch.object(pipeline, "hf_hub_download", return_value=cached) as download_mock:
                unit = NYUDepthV2ArchiveUnit(repo_path="data/val-000001.tar")
                pipeline.download_unit(unit)
            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["filename"], "data/val-000001.tar")

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            image = np.zeros((3, 4, 6), dtype=np.uint8)
            image[0, :, :] = 128
            depth = np.full((4, 6), 2.0, dtype=np.float32)
            sample_path = pipeline.paths.raw / "_extracted" / "val-000001" / "sample_0001.h5"
            self._write_h5(sample_path, image=image, depth=depth)

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
