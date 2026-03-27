import json
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import TartanAirPipeline
from depth_collector.datasets.tartanair import TartanAirArchiveUnit, TartanAirSourceItem
from depth_collector.geometry import PinholeCameraModel


class TartanAirPipelineTest(unittest.TestCase):
    def _make_config(self, root_data_dir: str) -> dict[str, object]:
        return {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "processing_fraction": 0.01,
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
                "tartanair": {
                    "enabled": True,
                    "hf_dataset_id": "theairlabcmu/tartanair",
                    "environments": ["neighborhood"],
                    "difficulties": ["Easy"],
                    "modalities": ["image_left"],
                }
            },
        }

    def test_archive_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(self._make_config(tmp_dir)))
            pipeline = TartanAirPipeline(load_config(config_path), "tartanair")
            units = list(pipeline.enumerate_download_units())
            self.assertEqual(
                units,
                [TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")],
            )

    def test_source_item_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(self._make_config(tmp_dir)))
            pipeline = TartanAirPipeline(load_config(config_path), "tartanair")
            item = TartanAirSourceItem(
                environment="neighborhood",
                difficulty="Easy",
                modality="image_left",
                relative_path="P000/image.png",
            )
            self.assertEqual(
                pipeline.get_source_item_id(item),
                "neighborhood/Easy/image_left/P000/image.png",
            )

    def test_camera_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(self._make_config(tmp_dir)))
            pipeline = TartanAirPipeline(load_config(config_path), "tartanair")
            model = pipeline.build_camera_model(
                TartanAirSourceItem("neighborhood", "Easy", "image_left", "P000/image.png"),
                {},
            )
            self.assertIsInstance(model, PinholeCameraModel)
            self.assertEqual((model.width, model.height), (640, 640))
            self.assertEqual((model.fx, model.cx), (320.0, 320.0))


if __name__ == "__main__":
    unittest.main()
