import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import DIODEPipeline
from depth_collector.datasets.diode import DIODEArchiveUnit, DIODESourceItem


class DIODEPipelineTest(unittest.TestCase):
    def _make_config(self, root_data_dir: str) -> dict[str, object]:
        return {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 2,
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
                "diode_subset_train": {
                    "enabled": True,
                    "hf_dataset_id": "sayakpaul/diode-subset-train",
                    "archive_filename": "train_subset.tar.gz",
                    "splits": ["train"],
                    "scene_types": "*",
                    "depth_semantics": "distance",
                    "camera_intrinsics": {
                        "width": 1024,
                        "height": 768,
                        "fx": 512.0,
                        "fy": 512.0,
                        "cx": 512.0,
                        "cy": 384.0,
                    },
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str) -> DIODEPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir)))
        return DIODEPipeline(load_config(config_path), "diode_subset_train")

    def _write_extracted_layout(self, root: Path) -> None:
        image_path = root / "indoors" / "scene_0001" / "scan_0001" / "frame_0001.png"
        depth_path = root / "indoors" / "scene_0001" / "scan_0001" / "frame_0001_depth.npy"
        mask_path = root / "indoors" / "scene_0001" / "scan_0001" / "frame_0001_depth_mask.npy"

        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[..., 0] = 64
        image[..., 2] = 255
        Image.fromarray(image).save(image_path)

        depth = np.full((6, 8), 3.5, dtype=np.float32)
        depth[0, 0] = 0.0
        np.save(depth_path, depth)

        mask = np.ones((6, 8), dtype=bool)
        mask[0, 0] = False
        np.save(mask_path, mask)

    def test_download_uses_local_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_archive = Path(source_dir) / "train_subset.tar.gz"
            source_archive.write_bytes(b"test")
            pipeline.dataset_config.options["local_archive_root"] = source_dir

            unit = DIODEArchiveUnit(archive_name="train_subset.tar.gz")
            pipeline.download_unit(unit)

            self.assertTrue((pipeline.paths.raw / "train_subset.tar.gz").exists())

    def test_extract_and_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            extracted_root = Path(source_dir) / "extracted"
            self._write_extracted_layout(extracted_root)

            archive_path = Path(source_dir) / "train_subset.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(extracted_root / "indoors", arcname="train_subset/indoors")
            pipeline.dataset_config.options["local_archive_root"] = source_dir

            unit = DIODEArchiveUnit(archive_name="train_subset.tar.gz")
            pipeline.download_unit(unit)
            pipeline.extract_unit(unit)

            items = list(pipeline.enumerate_source_items())
            self.assertEqual(
                items,
                [
                    DIODESourceItem(
                        split_name="train",
                        scene_type="indoors",
                        relative_stem="indoors/scene_0001/scan_0001/frame_0001",
                        image_relative_path="indoors/scene_0001/scan_0001/frame_0001.png",
                        depth_relative_path="indoors/scene_0001/scan_0001/frame_0001_depth.npy",
                        depth_mask_relative_path="indoors/scene_0001/scan_0001/frame_0001_depth_mask.npy",
                    )
                ],
            )

    def test_build_sample_uses_masked_metric_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_extracted_layout(pipeline.paths.raw / "train_subset")

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (6, 8, 3))
            self.assertEqual(sample.distance.shape, (6, 8, 1))
            self.assertEqual(sample.ray_dir.shape, (6, 8, 3))
            self.assertAlmostEqual(float(sample.distance[1, 1, 0]), 3.5, places=5)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)
            self.assertEqual(sample.provenance["scene_type"], "indoors")
            self.assertEqual(sample.provenance["projection"], "pinhole")

    def test_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = DIODEArchiveUnit(archive_name="train_subset.tar.gz")
            downloaded_archive = pipeline.paths.raw / "cache" / "train_subset.tar.gz"
            downloaded_archive.parent.mkdir(parents=True, exist_ok=True)
            downloaded_archive.write_bytes(b"test")
            with patch.object(pipeline, "hf_hub_download") as download_mock:
                download_mock.return_value = str(downloaded_archive)
                pipeline.download_unit(unit)
            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["filename"], "train_subset.tar.gz")

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_extracted_layout(pipeline.paths.raw / "train_subset")

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
