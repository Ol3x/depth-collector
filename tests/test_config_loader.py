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
                "download_workers": 2,
                "process_ratio": 0.01,
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
        self.assertEqual(config.runtime.download_workers, 2)
        self.assertEqual(config.runtime.max_relative_far_distance_fraction, 0.98)
        self.assertEqual(config.runtime.min_metric_distance_std_m, 0.1)
        self.assertEqual(config.runtime.max_relative_distance_std, 0.3)
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
                "download_workers": 2,
                "process_ratio": 2.0,
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

    def test_invalid_process_ratio_fails(self) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 2,
                "process_ratio": 0.0,
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

    def test_processing_fraction_alias_still_loads(self) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 2,
                "processing_fraction": 0.25,
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
        self.assertEqual(config.runtime.process_ratio, 0.25)

    def test_download_workers_defaults_and_validates(self) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "process_ratio": 0.25,
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
        self.assertEqual(config.runtime.download_workers, 2)

        payload["runtime"]["download_workers"] = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_validation_thresholds_validate(self) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "process_ratio": 0.25,
                "shuffle_seed": 0,
                "resume": True,
                "skip_known_errors": True,
                "write_error_traces": True,
                "target_shard_size_gb": 1.0,
                "max_relative_far_distance_fraction": 1.1,
                "min_metric_distance_std_m": 0.1,
                "max_relative_distance_std": 0.3,
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
