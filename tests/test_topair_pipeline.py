import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import TopAirPipeline
from depth_collector.datasets.topair import TopAirSourceItem, TopAirTrajectoryUnit


class TopAirPipelineTest(unittest.TestCase):
    def _make_config(
        self,
        root_data_dir: str,
        trajectory_count: int = 1,
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
                "topair": {
                    "enabled": True,
                    "hf_dataset_id": "yaraalaa0/TopAir",
                    "trajectories": "*",
                    "trajectory_count": trajectory_count,
                    "use_semantic_masks": True,
                    "sky_class_id": 0,
                    "depth_semantics": "distance",
                    "depth_unit_meters": 100.0 / 255.0,
                    "camera_intrinsics": {
                        "width": 384,
                        "height": 384,
                        "fx": 192.0,
                        "fy": 192.0,
                        "cx": 192.0,
                        "cy": 192.0,
                    },
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str) -> TopAirPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir)))
        return TopAirPipeline(load_config(config_path), "topair")

    def _write_trajectory_tree(
        self,
        root: Path,
        *,
        trajectory_name: str = "AssetsvilleTown_2",
        frame_id: str = "0001",
        semantic_rgb: bool = False,
    ) -> None:
        image_root = root / trajectory_name / "images"
        depth_root = root / trajectory_name / "depth"
        semantic_root = root / trajectory_name / "seg_id"
        image_root.mkdir(parents=True, exist_ok=True)
        depth_root.mkdir(parents=True, exist_ok=True)
        semantic_root.mkdir(parents=True, exist_ok=True)

        image = np.zeros((384, 384, 3), dtype=np.uint8)
        image[..., 1] = 128
        image[..., 2] = 255
        Image.fromarray(image).save(image_root / f"{frame_id}.png")

        depth = np.full((384, 384), 10, dtype=np.uint8)
        Image.fromarray(depth).save(depth_root / f"{frame_id}.png")

        semantic = np.full((384, 384), 1, dtype=np.uint8)
        semantic[0, 0] = 0
        if semantic_rgb:
            semantic = np.stack([semantic, semantic, semantic], axis=-1)
        Image.fromarray(semantic).save(semantic_root / f"{frame_id}.png")

        (root / trajectory_name / "camera_loc.txt").write_text(f"{frame_id} 1 2 3 0 0 -90\n")

    def test_all_trajectory_selector_discovers_remote_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "AssetsvilleTown_2/images/0001.png",
                    "AssetsvilleTown_2/depth/0001.png",
                    "AssetsvilleTown_2/seg_id/0001.png",
                    "AbandonedFactory_1/images/0001.png",
                    "AbandonedFactory_1/depth/0001.png",
                    "AbandonedFactory_1/seg_id/0001.png",
                ],
            ):
                units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [TopAirTrajectoryUnit(trajectory_name="AbandonedFactory_1")])

    def test_remote_download_requests_whole_trajectory_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = TopAirTrajectoryUnit(trajectory_name="AssetsvilleTown_2")
            with patch.object(pipeline, "hf_snapshot_download") as download_mock:
                pipeline.download_unit(unit)
            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["allow_patterns"], ["AssetsvilleTown_2/**"])

    def test_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_trajectory_tree(pipeline.paths.raw)

            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    TopAirSourceItem(
                        trajectory_name="AssetsvilleTown_2",
                        frame_id="0001",
                        image_relative_path="AssetsvilleTown_2/images/0001.png",
                        depth_relative_path="AssetsvilleTown_2/depth/0001.png",
                        semantic_relative_path="AssetsvilleTown_2/seg_id/0001.png",
                    )
                ],
            )

    def test_build_sample_uses_metric_distance_and_masks_sky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_trajectory_tree(pipeline.paths.raw)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (384, 384, 3))
            self.assertEqual(sample.distance.shape, (384, 384, 1))
            self.assertEqual(sample.ray_dir.shape, (384, 384, 3))
            self.assertAlmostEqual(float(sample.distance[1, 1, 0]), 10.0 * (100.0 / 255.0), places=5)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)
            self.assertEqual(sample.provenance["trajectory_name"], "AssetsvilleTown_2")
            self.assertEqual(sample.provenance["projection"], "pinhole")
            self.assertIn("camera_pose", sample.provenance)

    def test_build_sample_accepts_rgb_semantic_mask_with_equal_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_trajectory_tree(pipeline.paths.raw, semantic_rgb=True)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_trajectory_tree(pipeline.paths.raw)

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
