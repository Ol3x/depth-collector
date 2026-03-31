import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import ToF360Pipeline
from depth_collector.datasets.tof_360 import ToF360SceneUnit, ToF360SourceItem


class ToF360PipelineTest(unittest.TestCase):
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
                "tof_360": {
                    "enabled": True,
                    "hf_dataset_id": "COLE-Ricoh/ToF-360",
                    "selection": selection,
                    "scenes": "*",
                    "rgb_dir": "rgb",
                    "depth_dir": "depth",
                    "depth_scale_divisor": 512.0,
                    "missing_depth_policy": "max_dist",
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str) -> ToF360Pipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir)))
        return ToF360Pipeline(load_config(config_path), "tof_360")

    def _write_scene_tree(self, root: Path, scene_name: str = "scene_0001", frame_id: str = "frame_0001") -> None:
        rgb_root = root / scene_name / "rgb"
        depth_root = root / scene_name / "depth"
        rgb_root.mkdir(parents=True, exist_ok=True)
        depth_root.mkdir(parents=True, exist_ok=True)

        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[..., 0] = 64
        image[..., 1] = 128
        image[..., 2] = 255
        Image.fromarray(image).save(rgb_root / f"{frame_id}.png")

        depth = np.full((4, 8), 512, dtype=np.uint16)
        depth[0, 0] = 0
        Image.fromarray(depth).save(depth_root / f"{frame_id}.png")

    def _write_scene_tree_with_rgb_dir(
        self,
        root: Path,
        *,
        scene_name: str = "scene_0001",
        frame_id: str = "frame_0001",
        rgb_dir_name: str = "manhattan",
    ) -> None:
        rgb_root = root / scene_name / rgb_dir_name
        depth_root = root / scene_name / "depth"
        rgb_root.mkdir(parents=True, exist_ok=True)
        depth_root.mkdir(parents=True, exist_ok=True)

        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[..., 1] = 255
        Image.fromarray(image).save(rgb_root / f"{frame_id}.png")

        depth = np.full((4, 8), 512, dtype=np.uint16)
        Image.fromarray(depth).save(depth_root / f"{frame_id}.png")

    def test_all_scene_selector_discovers_remote_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "scene_0001/rgb/frame_0001.png",
                    "scene_0001/depth/frame_0001.png",
                    "scene_0002/rgb/frame_0001.png",
                    "scene_0002/depth/frame_0001.png",
                ],
            ):
                units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [ToF360SceneUnit(scene_name="scene_0001")])

    def test_download_uses_local_scene_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            with tempfile.TemporaryDirectory() as source_dir:
                source_root = Path(source_dir)
                self._write_scene_tree(source_root)
                pipeline.dataset_config.options["local_archive_root"] = str(source_root)

                unit = ToF360SceneUnit(scene_name="scene_0001")
                pipeline.download_unit(unit)

            self.assertTrue((pipeline.paths.raw / "scene_0001" / "rgb" / "frame_0001.png").exists())
            self.assertTrue((pipeline.paths.raw / "scene_0001" / "depth" / "frame_0001.png").exists())

    def test_remote_download_requests_whole_scene_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = ToF360SceneUnit(scene_name="scene_0001")

            with patch.object(pipeline, "hf_snapshot_download") as download_mock:
                pipeline.download_unit(unit)

            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["allow_patterns"], ["scene_0001/**"])

    def test_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_scene_tree(pipeline.paths.raw)

            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    ToF360SourceItem(
                        scene_name="scene_0001",
                        frame_id="frame_0001",
                        image_relative_path="scene_0001/rgb/frame_0001.png",
                        depth_relative_path="scene_0001/depth/frame_0001.png",
                    )
                ],
            )

    def test_enumerate_source_items_resolves_alternate_rgb_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_scene_tree_with_rgb_dir(pipeline.paths.raw, rgb_dir_name="manhattan")

            items = list(pipeline.enumerate_source_items())

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image_relative_path, "scene_0001/manhattan/frame_0001.png")

    def test_enumerate_source_items_matches_depth_and_rgb_suffix_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            scene_root = pipeline.paths.raw / "scene_0001"
            rgb_root = scene_root / "manhattan"
            depth_root = scene_root / "depth"
            rgb_root.mkdir(parents=True, exist_ok=True)
            depth_root.mkdir(parents=True, exist_ok=True)

            image = np.zeros((4, 8, 3), dtype=np.uint8)
            image[..., 0] = 255
            Image.fromarray(image).save(rgb_root / "000_Hospital_equi_manhattan.png")

            depth = np.full((4, 8), 512, dtype=np.uint16)
            Image.fromarray(depth).save(depth_root / "000_Hospital_equi_depth.png")

            items = list(pipeline.enumerate_source_items())

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].frame_id, "000_Hospital_equi")
            self.assertEqual(items[0].image_relative_path, "scene_0001/manhattan/000_Hospital_equi_manhattan.png")
            self.assertEqual(items[0].depth_relative_path, "scene_0001/depth/000_Hospital_equi_depth.png")

    def test_build_sample_uses_metric_distance_and_equirectangular_rays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_scene_tree(pipeline.paths.raw)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (4, 8, 3))
            self.assertEqual(sample.distance.shape, (4, 8, 1))
            self.assertEqual(sample.ray_dir.shape, (4, 8, 3))
            self.assertAlmostEqual(float(sample.distance[1, 1, 0]), 1.0, places=5)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)
            self.assertEqual(sample.provenance["scene_name"], "scene_0001")
            self.assertEqual(sample.provenance["projection"], "equirectangular")

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_scene_tree(pipeline.paths.raw)

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
