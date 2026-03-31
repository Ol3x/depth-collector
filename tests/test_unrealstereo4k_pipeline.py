import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import UnrealStereo4KPipeline
from depth_collector.datasets.unrealstereo4k import UnrealStereo4KArchiveUnit, UnrealStereo4KSourceItem


class UnrealStereo4KPipelineTest(unittest.TestCase):
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
                "unrealstereo4k": {
                    "enabled": True,
                    "hf_dataset_id": "fabiotosi92/UnrealStereo4K",
                    "selection": selection,
                    "archives": ["00008.zip"],
                    "image_dir": "frames_cleanpass/left",
                    "disparity_dir": "disparity/left",
                    "camera_intrinsics": {
                        "width": 960.0,
                        "height": 540.0,
                        "fx": 960.0,
                        "fy": 960.0,
                        "cx": 480.0,
                        "cy": 270.0,
                    },
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str, *, selection: object = "minimum_readable") -> UnrealStereo4KPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir, selection=selection)))
        return UnrealStereo4KPipeline(load_config(config_path), "unrealstereo4k")

    def _write_archive(self, archive_path: Path, *, scene_name: str = "00008") -> None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as build_dir:
            build_root = Path(build_dir) / scene_name
            image_root = build_root / "frames_cleanpass" / "left"
            disparity_root = build_root / "disparity" / "left"
            right_root = build_root / "frames_cleanpass" / "right"
            image_root.mkdir(parents=True, exist_ok=True)
            disparity_root.mkdir(parents=True, exist_ok=True)
            right_root.mkdir(parents=True, exist_ok=True)

            first_image = np.zeros((4, 6, 3), dtype=np.uint8)
            first_image[..., 0] = 64
            Image.fromarray(first_image).save(image_root / "0001.png")
            np.save(disparity_root / "0001.npy", np.full((4, 6), 2.0, dtype=np.float32))
            Image.fromarray(first_image).save(right_root / "0001.png")

            second_image = np.zeros((4, 6, 3), dtype=np.uint8)
            second_image[..., 1] = 255
            Image.fromarray(second_image).save(image_root / "0002.png")
            np.save(disparity_root / "0002.npy", np.full((4, 6), 4.0, dtype=np.float32))
            Image.fromarray(second_image).save(right_root / "0002.png")

            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file_path in sorted(build_root.rglob("*")):
                    if file_path.is_file():
                        archive.write(file_path, arcname=file_path.relative_to(build_root.parent).as_posix())

    def test_minimum_readable_download_materializes_single_pair_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_root = Path(tmp_dir) / "source_archives"
            pipeline.dataset_config.options["local_archive_root"] = str(source_root)
            source_archive = source_root / "00008.zip"
            self._write_archive(source_archive)

            unit = UnrealStereo4KArchiveUnit(archive_name="00008.zip", repo_path=str(source_archive))
            pipeline.download_unit(unit)

            with zipfile.ZipFile(pipeline._archive_path(unit)) as archive:
                names = sorted(archive.namelist())
            self.assertEqual(
                names,
                [
                    "00008/disparity/left/0001.npy",
                    "00008/frames_cleanpass/left/0001.png",
                ],
            )

    def test_minimum_readable_remote_download_uses_zip_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as build_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_archive = Path(build_dir) / "00008.zip"
            self._write_archive(source_archive)

            class _RemoteZip(zipfile.ZipFile):
                def __enter__(self) -> "_RemoteZip":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    self.close()

            with patch.object(pipeline, "hf_open_remote_zip", return_value=_RemoteZip(source_archive)) as open_mock:
                unit = UnrealStereo4KArchiveUnit(archive_name="00008.zip", repo_path="00008.zip")
                pipeline.download_unit(unit)
            open_mock.assert_called_once()

    def test_extract_enumerate_and_build_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = UnrealStereo4KArchiveUnit(archive_name="00008.zip", repo_path="00008.zip")
            self._write_archive(pipeline._archive_path(unit))

            pipeline.extract_unit(unit)
            items = list(pipeline.enumerate_source_items())
            self.assertEqual(
                items,
                [
                    UnrealStereo4KSourceItem(
                        scene_name="00008",
                        frame_key="0001",
                        archive_name="00008.zip",
                        image_relative_path="00008/frames_cleanpass/left/0001.png",
                        disparity_relative_path="00008/disparity/left/0001.npy",
                    ),
                    UnrealStereo4KSourceItem(
                        scene_name="00008",
                        frame_key="0002",
                        archive_name="00008.zip",
                        image_relative_path="00008/frames_cleanpass/left/0002.png",
                        disparity_relative_path="00008/disparity/left/0002.npy",
                    ),
                ],
            )

            loaded_item = pipeline.load_source_item(items[0])
            camera_model = pipeline.build_camera_model(items[0], loaded_item)
            sample = pipeline.build_sample(items[0], loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (4, 6, 3))
            self.assertEqual(sample.distance.shape, (4, 6, 1))
            self.assertEqual(sample.ray_dir.shape, (4, 6, 3))
            self.assertGreaterEqual(float(np.min(sample.distance)), 0.0)
            self.assertLessEqual(float(np.max(sample.distance)), 1.0)
            self.assertAlmostEqual(float(np.max(sample.distance)), 1.0, places=5)
            self.assertGreater(float(sample.distance[2, 3, 0]), 0.0)
            self.assertEqual(sample.provenance["depth_semantics"], "inverse_disparity_relative")

    def test_remote_download_uses_hf_helper_for_all_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, selection="all")
            cached = pipeline.paths.raw / "cache" / "00008.zip"
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"test")
            with patch.object(pipeline, "hf_hub_download", return_value=cached) as download_mock:
                unit = UnrealStereo4KArchiveUnit(archive_name="00008.zip", repo_path="00008.zip")
                pipeline.download_unit(unit)
            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["filename"], "00008.zip")

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            scene_root = pipeline.paths.raw / "00008"
            (scene_root / "frames_cleanpass" / "left").mkdir(parents=True, exist_ok=True)
            (scene_root / "disparity" / "left").mkdir(parents=True, exist_ok=True)

            image = np.zeros((4, 6, 3), dtype=np.uint8)
            image[..., 2] = 255
            Image.fromarray(image).save(scene_root / "frames_cleanpass" / "left" / "0001.png")
            disparity = np.linspace(1.0, 6.0, num=24, dtype=np.float32).reshape(4, 6)
            np.save(scene_root / "disparity" / "left" / "0001.npy", disparity)

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
