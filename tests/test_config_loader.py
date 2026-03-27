import json
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_load_config(self) -> None:
        payload = {
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
                "root_data_dir": "data",
                "raw_subdir_name": "raw",
                "processed_subdir_name": "processed",
                "state_subdir_name": "state",
                "metadata_filename": "metadata.json",
            },
            "datasets": {
                "nyu_depth_v2": {
                    "enabled": True,
                    "hf_dataset_id": "sayakpaul/nyu_depth_v2",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps(payload))
            config = load_config(path)
        self.assertEqual(config.project.max_dist, 100.0)
        self.assertTrue(config.datasets["nyu_depth_v2"].enabled)

    def test_invalid_fraction_fails(self) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "processing_fraction": 2.0,
                "shuffle_seed": 0,
                "resume": True,
                "skip_known_errors": True,
                "write_error_traces": True,
                "target_shard_size_gb": 1.0,
            },
            "output": {
                "root_data_dir": "data",
                "raw_subdir_name": "raw",
                "processed_subdir_name": "processed",
                "state_subdir_name": "state",
                "metadata_filename": "metadata.json",
            },
            "datasets": {
                "nyu_depth_v2": {
                    "enabled": True,
                    "hf_dataset_id": "sayakpaul/nyu_depth_v2",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
