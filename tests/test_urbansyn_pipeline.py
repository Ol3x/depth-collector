import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import UrbanSynPipeline
from depth_collector.datasets.urbansyn import UrbanSynFrameUnit, UrbanSynSourceItem


class UrbanSynPipelineTest(unittest.TestCase):
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
                "urbansyn": {
                    "enabled": True,
                    "hf_dataset_id": "UrbanSyn/UrbanSyn",
                    "selection": selection,
                    "frames": "*",
                    "use_semantic_masks": True,
                    "sky_class_id": 10,
                    "depth_semantics": "z_depth",
                    "depth_unit_meters": 1.0,
                    "camera_intrinsics": {
                        "width": 4,
                        "height": 2,
                        "fx": 2.0,
                        "fy": 2.0,
                        "cx": 2.0,
                        "cy": 1.0,
                    },
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str) -> UrbanSynPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir)))
        return UrbanSynPipeline(load_config(config_path), "urbansyn")

    def _write_source_tree(self, root: Path, frame_id: str = "0001") -> None:
        (root / "rgb").mkdir(parents=True, exist_ok=True)
        (root / "depth").mkdir(parents=True, exist_ok=True)
        (root / "ss").mkdir(parents=True, exist_ok=True)

        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[..., 0] = 64
        image[..., 1] = 128
        image[..., 2] = 255
        Image.fromarray(image).save(root / "rgb" / f"rgb_{frame_id}.png")

        (root / "depth" / f"depth_{frame_id}.exr").write_bytes(b"placeholder")

        semantics = np.zeros((2, 4), dtype=np.uint8)
        semantics[0, 0] = 10
        Image.fromarray(semantics).save(root / "ss" / f"ss_{frame_id}.png")

    def test_all_frame_selector_discovers_remote_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "rgb/rgb_0001.png",
                    "depth/depth_0001.exr",
                    "ss/ss_0001.png",
                    "rgb/rgb_0002.png",
                    "depth/depth_0002.exr",
                    "ss/ss_0002.png",
                ],
            ):
                units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [UrbanSynFrameUnit(frame_id="0001")])

    def test_download_uses_local_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            with tempfile.TemporaryDirectory() as source_dir:
                source_root = Path(source_dir)
                self._write_source_tree(source_root)
                pipeline.dataset_config.options["local_archive_root"] = str(source_root)

                unit = UrbanSynFrameUnit(frame_id="0001")
                pipeline.download_unit(unit)

            self.assertTrue((pipeline.paths.raw / "rgb" / "rgb_0001.png").exists())
            self.assertTrue((pipeline.paths.raw / "depth" / "depth_0001.exr").exists())
            self.assertTrue((pipeline.paths.raw / "ss" / "ss_0001.png").exists())

    def test_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_source_tree(pipeline.paths.raw)

            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    UrbanSynSourceItem(
                        frame_id="0001",
                        image_relative_path="rgb/rgb_0001.png",
                        depth_relative_path="depth/depth_0001.exr",
                        semantic_relative_path="ss/ss_0001.png",
                    )
                ],
            )

    def test_build_sample_uses_distance_semantics_and_sky_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_source_tree(pipeline.paths.raw)
            pipeline.dataset_config.options["depth_semantics"] = "distance"

            item = next(iter(pipeline.enumerate_source_items()))
            with patch.object(
                pipeline, "_load_exr_payload",
                return_value=(
                    np.array(
                        [
                            [200.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                        ],
                        dtype=np.float32,
                    ),
                    {},
                ),
            ):
                loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (2, 4, 3))
            self.assertEqual(sample.distance.shape, (2, 4, 1))
            self.assertEqual(sample.ray_dir.shape, (2, 4, 3))
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 100.0, places=5)
            self.assertAlmostEqual(float(sample.distance[0, 1, 0]), 2.0, places=5)
            self.assertEqual(sample.provenance["frame_id"], "0001")

    def test_load_source_item_accepts_rgb_semantic_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_source_tree(pipeline.paths.raw)

            item = next(iter(pipeline.enumerate_source_items()))
            with patch.object(
                pipeline, "_load_exr_payload",
                return_value=(np.full((2, 4), 2.0, dtype=np.float32), {}),
            ):
                loaded_item = pipeline.load_source_item(item)

            semantic = loaded_item["semantic"]
            assert isinstance(semantic, np.ndarray)
            self.assertEqual(semantic.shape, (2, 4))

    def test_run_writes_real_shard_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_source_tree(pipeline.paths.raw)

            pipeline.prepare_directories()
            with patch.object(
                pipeline, "_load_exr_payload",
                return_value=(np.full((2, 4), 2.0, dtype=np.float32), {}),
            ):
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

    def test_build_camera_model_prefers_exr_octane_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            intrinsics = pipeline._camera_intrinsics_for_image(
                width=2048,
                height=1024,
                exr_metadata={
                    "octane": {
                        "renderTarget": {
                            "camera": {
                                "focalLength": 4.900001049041748,
                                "sensorWidth": 5.8000001907348633,
                                "lensShift": {"x": 0, "y": 0},
                            },
                            "resolution": {
                                "dimensions": {"x": 2048, "y": 1024},
                            },
                        }
                    }
                },
            )
            self.assertAlmostEqual(intrinsics["fx"], 1730.207210073563, places=3)
            self.assertAlmostEqual(intrinsics["fy"], 1730.207210073563, places=3)
            self.assertAlmostEqual(intrinsics["cx"], 1024.0, places=3)
            self.assertAlmostEqual(intrinsics["cy"], 512.0, places=3)


if __name__ == "__main__":
    unittest.main()
