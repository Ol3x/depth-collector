import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401


class MainScriptsSmokeTest(unittest.TestCase):
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _python(self) -> str:
        return "/home/olx2024/micromamba/envs/depth-collector/bin/python"

    def _dc(self) -> str:
        return "/home/olx2024/micromamba/envs/depth-collector/bin/dc"

    def _write_archive_sources(self, source_root: Path) -> None:
        image_archive = source_root / "neighborhood" / "Easy" / "image_left.zip"
        depth_archive = source_root / "neighborhood" / "Easy" / "depth_left.zip"
        image_archive.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as build_dir:
            build_root = Path(build_dir)
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            image[..., 0] = 64
            image[..., 1] = 128
            image[..., 2] = 255
            image_path = build_root / "P000" / "000000_left.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(image_path)

            depth = np.full((4, 6), 2.0, dtype=np.float32)
            depth_path = build_root / "P000" / "000000_left_depth.npy"
            np.save(depth_path, depth)

            with zipfile.ZipFile(image_archive, "w") as archive:
                archive.write(image_path, arcname="P000/000000_left.png")
            with zipfile.ZipFile(depth_archive, "w") as archive:
                archive.write(depth_path, arcname="P000/000000_left_depth.npy")

    def _write_config(self, config_path: Path, data_root: Path, source_root: Path) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "script smoke test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 2,
                "process_ratio": 1e-12,
                "shuffle_seed": 0,
                "resume": True,
                "skip_known_errors": True,
                "write_error_traces": True,
                "target_shard_size_gb": 1.0,
            },
            "output": {
                "root_data_dir": str(data_root),
                "raw_subdir_name": "raw",
                "processed_subdir_name": "processed",
                "state_subdir_name": "state",
                "metadata_filename": "metadata.json",
            },
            "datasets": {
                "tartanair": {
                    "enabled": True,
                    "hf_dataset_id": "theairlabcmu/tartanair",
                    "download_workers": 1,
                    "environments": ["neighborhood"],
                    "environment_count": 1,
                    "difficulties": ["Easy"],
                    "modalities": ["image_left"],
                    "local_archive_root": str(source_root),
                }
            },
        }
        config_path.write_text(json.dumps(payload))

    def _write_megadepth_config(self, config_path: Path, data_root: Path) -> None:
        payload = {
            "project": {
                "name": "default",
                "description": "megadepth config failure smoke test",
                "max_dist": 100.0,
                "train_val_split": 0.95,
            },
            "runtime": {
                "download_workers": 1,
                "process_ratio": 0.05,
                "shuffle_seed": 0,
                "resume": True,
                "skip_known_errors": True,
                "write_error_traces": True,
                "target_shard_size_gb": 1.0,
            },
            "output": {
                "root_data_dir": str(data_root),
                "raw_subdir_name": "raw",
                "processed_subdir_name": "processed",
                "state_subdir_name": "state",
                "metadata_filename": "metadata.json",
            },
            "datasets": {
                "megadepth": {
                    "enabled": True,
                    "hf_dataset_id": "MegaDepth",
                    "bundles": ["megadepth_bundle"],
                    "bundle_count": 1,
                    "scene_info_dir": "scene_info",
                }
            },
        }
        config_path.write_text(json.dumps(payload))

    def test_dc_cli_smoke_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / "source_archives"
            data_root = root / "data"
            configs_root = root / "configs"
            configs_root.mkdir(parents=True, exist_ok=True)
            config_path = configs_root / "default.json"
            self._write_archive_sources(source_root)
            self._write_config(config_path, data_root, source_root)

            repo_root = self._repo_root()
            download_result = subprocess.run(
                [self._dc(), "download", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("[tartanair] download workers: 1", download_result.stdout)

            raw_root = data_root / "default" / "metric" / "tartanair" / "raw" / "neighborhood" / "Easy"
            hf_cache_root = data_root / "default" / "metric" / "tartanair" / ".hf_cache"
            self.assertTrue((raw_root / "image_left.zip").exists())
            self.assertTrue((raw_root / "depth_left.zip").exists())
            (hf_cache_root / "hub").mkdir(parents=True, exist_ok=True)
            (hf_cache_root / "hub" / "placeholder.txt").write_text("cache")

            subprocess.run(
                [self._dc(), "extract", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
            )

            self.assertTrue((raw_root / "image_left" / "P000" / "000000_left.png").exists())
            self.assertTrue((raw_root / "depth_left" / "P000" / "000000_left_depth.npy").exists())
            self.assertFalse((raw_root / "image_left.zip").exists())
            self.assertFalse((raw_root / "depth_left.zip").exists())
            self.assertFalse(hf_cache_root.exists())

            (hf_cache_root / "hub").mkdir(parents=True, exist_ok=True)
            (hf_cache_root / "hub" / "placeholder.txt").write_text("cache")
            extract_keep_cache_result = subprocess.run(
                [self._dc(), "extract", "default", "--config", str(config_path), "--keep-cache"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("skip extract", extract_keep_cache_result.stdout)
            self.assertTrue(hf_cache_root.exists())

            status_result = subprocess.run(
                [self._dc(), "status", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("project: default", status_result.stdout)
            self.assertIn("[tartanair] archives_present=0/2", status_result.stdout)

            subprocess.run(
                [self._dc(), "process", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
            )

            visualize_result = subprocess.run(
                [self._dc(), "visualize", "default", "--config", str(config_path), "--max-samples", "1"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("visualization summary", visualize_result.stdout)
            visualization_paths = sorted(
                (data_root / "default" / "metric" / "tartanair" / "visualizations").rglob("*.png")
            )
            self.assertEqual(len(visualization_paths), 1)
            self.assertEqual(visualization_paths[0].parent.name, "neighborhood__Easy")

            visualize_all_result = subprocess.run(
                [self._dc(), "visualize", "default", "--config", str(config_path), "--all", "--samples-per-image", "1"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("visualization summary", visualize_all_result.stdout)

            conflict_result = subprocess.run(
                [self._dc(), "visualize", "default", "--config", str(config_path), "--all", "--max-samples", "1"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(conflict_result.returncode, 0)
            self.assertIn("either --all or --max-samples", conflict_result.stderr)

            shard_paths = sorted((data_root / "default" / "metric" / "tartanair" / "processed" / "files").glob("*.tar"))
            self.assertEqual(len(shard_paths), 2)
            self.assertEqual([path.name for path in shard_paths], ["shard-000000.tar", "shard-000001.tar"])
            with tarfile.open(shard_paths[0], "r") as archive:
                member_names = sorted(member.name for member in archive.getmembers())
            self.assertEqual(
                member_names,
                [
                    "neighborhood__Easy__P000__000000_left.png.distance.pt",
                    "neighborhood__Easy__P000__000000_left.png.image.pt",
                    "neighborhood__Easy__P000__000000_left.png.meta.json",
                    "neighborhood__Easy__P000__000000_left.png.ray_dir.pt",
                ],
            )

            metadata = json.loads((data_root / "default" / "metric" / "tartanair" / "processed" / "metadata.json").read_text())
            run_report = json.loads((data_root / "default" / "metric" / "tartanair" / "processed" / "run_report.json").read_text())
            self.assertEqual(metadata["valid_sample_count"], 1)
            self.assertEqual(metadata["shard_count"], 2)
            self.assertEqual(metadata["suggested_train_shards"], [shard_paths[0].name])
            self.assertEqual(metadata["suggested_val_shards"], [shard_paths[1].name])
            self.assertEqual(run_report["dataset"], "tartanair")
            self.assertEqual(run_report["shard_count"], 2)
            self.assertEqual(run_report["error_stage_counts"], {})
            no_op_process_result = subprocess.run(
                [self._dc(), "process", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("all selected source items were already processed", no_op_process_result.stdout)
            metadata_after_no_op = json.loads(
                (data_root / "default" / "metric" / "tartanair" / "processed" / "metadata.json").read_text()
            )
            self.assertEqual(metadata_after_no_op["shard_count"], 2)

            clean_process_result = subprocess.run(
                [self._dc(), "clean_process", "default", "--config", str(config_path), "--yes"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("removed process artifacts", clean_process_result.stdout)
            self.assertTrue((raw_root / "image_left" / "P000" / "000000_left.png").exists())
            self.assertFalse((data_root / "default" / "metric" / "tartanair" / "processed").exists())
            self.assertFalse((data_root / "default" / "metric" / "tartanair" / "state" / "processed.jsonl").exists())
            self.assertFalse((data_root / "default" / "metric" / "tartanair" / "state" / "enumeration_manifest.json").exists())

            rebuilt_process_result = subprocess.run(
                [self._dc(), "process", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            rebuilt_shard_paths = sorted((data_root / "default" / "metric" / "tartanair" / "processed" / "files").glob("*.tar"))
            self.assertEqual(len(rebuilt_shard_paths), 2)

            manifest_path = data_root / "default" / "metric" / "tartanair" / "state" / "enumeration_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text())
            group_payload = manifest_payload["groups"]["neighborhood/Easy"]
            group_payload["items"] = []
            manifest_path.write_text(json.dumps(manifest_payload))
            shutil.rmtree(data_root / "default" / "metric" / "tartanair" / "processed")
            recovered_process_result = subprocess.run(
                [self._dc(), "process", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("retrying source enumeration because extracted files exist", recovered_process_result.stdout)
            recovered_shard_paths = sorted((data_root / "default" / "metric" / "tartanair" / "processed" / "files").glob("*.tar"))
            self.assertEqual(len(recovered_shard_paths), 2)

            self.assertEqual(
                sorted(path.name for path in (data_root / "default" / "metric" / "tartanair" / "processed").glob("*.partial")),
                [],
            )
            self.assertEqual(
                sorted(path.name for path in (data_root / "default" / "metric" / "tartanair" / "state").glob("*.partial")),
                [],
            )

    def test_download_reports_megadepth_selection_failure_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_root = root / "data"
            configs_root = root / "configs"
            configs_root.mkdir(parents=True, exist_ok=True)
            config_path = configs_root / "default.json"
            self._write_megadepth_config(config_path, data_root)

            repo_root = self._repo_root()
            result = subprocess.run(
                [self._dc(), "download", "default", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("[megadepth] download summary: downloaded=0 skipped=0 failed=1", result.stdout)
            self.assertNotIn("Traceback", result.stdout)
            self.assertFalse((data_root / "default" / "relative" / "megadepth" / "raw" / "prep_scene_info").exists())


if __name__ == "__main__":
    unittest.main()
