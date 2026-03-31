import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import MegaDepthPipeline
from depth_collector.datasets.megadepth import MegaDepthDownloadUnit, MegaDepthSourceItem


class MegaDepthPipelineTest(unittest.TestCase):
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
                "megadepth": {
                    "enabled": True,
                    "hf_dataset_id": "ssbai/MegaDepth_v1",
                    "selection": "minimum_readable",
                    "bundles": ["megadepth_bundle"],
                    "scene_info_dir": "prep_scene_info",
                    "scenes": ["0015"],
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str) -> MegaDepthPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir)))
        return MegaDepthPipeline(load_config(config_path), "megadepth")

    def _write_hdf5(self, path: Path, array: np.ndarray) -> None:
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("dataset", data=array)

    def _write_scene_layout(self, root: Path) -> None:
        image_path = root / "Undistorted_SfM" / "0015" / "images" / "img0000.jpg"
        depth_path = root / "phoenix" / "0015" / "depths" / "img0000.h5"
        scene_info_path = root / "prep_scene_info" / "0015.npz"

        image_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        scene_info_path.parent.mkdir(parents=True, exist_ok=True)

        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[..., 0] = 64
        image[..., 1] = 128
        image[..., 2] = 255
        Image.fromarray(image).save(image_path)

        depth = np.full((4, 6), 2.0, dtype=np.float32)
        self._write_hdf5(depth_path, depth)

        intrinsics = np.array(
            [
                [
                    [4.0, 0.0, 3.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        np.savez(
            scene_info_path,
            image_paths=np.array(["Undistorted_SfM/0015/images/img0000.jpg"], dtype=object),
            depth_paths=np.array(["phoenix/0015/depths/img0000.h5"], dtype=object),
            intrinsics=intrinsics,
        )

    def test_enumerate_and_build_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            self._write_scene_layout(pipeline.paths.raw)

            units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [MegaDepthDownloadUnit(unit_name="0015")])

            items = list(pipeline.enumerate_source_items())
            self.assertEqual(
                items,
                [
                    MegaDepthSourceItem(
                        scene_name="0015",
                        image_index=0,
                        image_relative_path="Undistorted_SfM/0015/images/img0000.jpg",
                        depth_relative_path="phoenix/0015/depths/img0000.h5",
                        intrinsics=(4.0, 4.0, 3.0, 2.0),
                    )
                ],
            )

            item = items[0]
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (4, 6, 3))
            self.assertEqual(sample.distance.shape, (4, 6, 1))
            self.assertEqual(sample.ray_dir.shape, (4, 6, 3))
            self.assertTrue(np.all(sample.image >= 0.0))
            self.assertTrue(np.all(sample.image <= 1.0))
            self.assertGreaterEqual(float(np.min(sample.distance)), 0.0)
            self.assertLessEqual(float(np.max(sample.distance)), 1.0)
            self.assertAlmostEqual(float(np.max(sample.distance)), 1.0, places=5)
            center_distance = float(sample.distance[2, 3, 0])
            self.assertGreater(center_distance, 0.0)
            self.assertLess(center_distance, 1.0)
            self.assertEqual(sample.provenance["scene_name"], "0015")
            self.assertEqual(sample.provenance["scale_semantics"], "scene-relative")
            self.assertEqual(sample.provenance["distance_normalization"], "[0, 1]")

    def test_directory_mode_download_from_local_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_root = Path(source_dir)
            self._write_scene_layout(source_root)
            pipeline.dataset_config.options["local_archive_root"] = str(source_root)

            unit = MegaDepthDownloadUnit(unit_name="0015")
            pipeline.download_unit(unit)

            self.assertTrue((pipeline.paths.raw / "prep_scene_info" / "0015.npz").exists())
            self.assertTrue((pipeline.paths.raw / "Undistorted_SfM" / "0015" / "images" / "img0000.jpg").exists())
            self.assertTrue((pipeline.paths.raw / "phoenix" / "0015" / "depths" / "img0000.h5").exists())

    def test_hf_multipart_download_and_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_root = Path(source_dir)
            extracted_root = source_root / "extracted"
            self._write_scene_layout(extracted_root)

            bundle_tar = source_root / "MegaDepth_v1.tar.gz"
            with tarfile.open(bundle_tar, "w:gz") as archive:
                archive.add(
                    extracted_root / "Undistorted_SfM",
                    arcname="Undistorted_SfM",
                )
                archive.add(
                    extracted_root / "phoenix",
                    arcname="phoenix",
                )
            bundle_bytes = bundle_tar.read_bytes()
            midpoint = len(bundle_bytes) // 2
            part0 = source_root / "MegaDepth_v1.tar.gz_part00"
            part1 = source_root / "MegaDepth_v1.tar.gz_part01"
            part0.write_bytes(bundle_bytes[:midpoint])
            part1.write_bytes(bundle_bytes[midpoint:])

            repo_files = [
                "MegaDepth_v1.tar.gz_part00",
                "MegaDepth_v1.tar.gz_part01",
                "prep_scene_info/0015.npz",
            ]

            def fake_download_hf_file(repo_id: str, repo_path: str, target_path: Path) -> Path:
                del repo_id
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source_path = extracted_root / repo_path if repo_path.startswith("prep_scene_info/") else source_root / Path(repo_path).name
                shutil.copy2(source_path, target_path)
                return target_path

            with patch.object(pipeline, "_list_hf_files", return_value=repo_files):
                units = list(pipeline.enumerate_download_units())
                self.assertEqual(
                    units,
                    [
                        MegaDepthDownloadUnit(
                            unit_name="megadepth_bundle",
                            unit_type="bundle",
                        ),
                    ],
                )
                with patch.object(pipeline, "_download_hf_file", side_effect=fake_download_hf_file):
                    pipeline.download_unit(units[0])

            self.assertTrue((pipeline.paths.raw / "_downloads" / "megadepth_bundle" / "MegaDepth_v1.tar.gz_part00").exists())
            self.assertTrue((pipeline.paths.raw / "prep_scene_info" / "0015.npz").exists())

            pipeline.extract_unit(MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle_extract"))
            self.assertTrue((pipeline.paths.raw / "prep_scene_info" / "0015.npz").exists())
            self.assertTrue((pipeline.paths.raw / "Undistorted_SfM" / "0015" / "images" / "img0000.jpg").exists())
            self.assertTrue((pipeline.paths.raw / "phoenix" / "0015" / "depths" / "img0000.h5").exists())

    def test_hf_scene_file_download_uses_scene_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_root = Path(source_dir)
            self._write_scene_layout(source_root)

            repo_files = [
                "prep_scene_info/0015.npz",
                "Undistorted_SfM/0015/images/img0000.jpg",
                "phoenix/0015/depths/img0000.h5",
            ]

            def fake_download_hf_file(repo_id: str, repo_path: str, target_path: Path) -> Path:
                del repo_id
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / repo_path, target_path)
                return target_path

            with patch.object(pipeline, "_list_hf_files", return_value=repo_files):
                units = list(pipeline.enumerate_download_units())
                self.assertEqual(units, [MegaDepthDownloadUnit(unit_name="0015")])
                with patch.object(pipeline, "_download_hf_file", side_effect=fake_download_hf_file):
                    pipeline.download_unit(units[0])

            self.assertTrue((pipeline.paths.raw / "prep_scene_info" / "0015.npz").exists())
            self.assertTrue((pipeline.paths.raw / "Undistorted_SfM" / "0015" / "images" / "img0000.jpg").exists())
            self.assertTrue((pipeline.paths.raw / "phoenix" / "0015" / "depths" / "img0000.h5").exists())
            self.assertEqual(tuple(pipeline.enumerate_extraction_units()), ())

    def test_minimum_readable_selection_still_selects_complete_bundle_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = self._make_config(tmp_dir)
            payload["datasets"]["megadepth"]["bundles"] = ["megadepth_bundle", "megadepth_bundle_2"]
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")

            repo_files = [
                "MegaDepth_v1.tar.gz_part00",
                "MegaDepth_v1.tar.gz_part01",
                "MegaDepth_v1.tar.gz_part02",
            ]
            with patch.object(pipeline, "_list_hf_files", return_value=repo_files):
                self.assertEqual(
                    pipeline.get_selected_download_units(),
                    [
                        MegaDepthDownloadUnit(
                            unit_name="megadepth_bundle",
                            unit_type="bundle",
                        ),
                    ],
                )

    def test_all_scene_selector_discovers_remote_scene_info_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = self._make_config(tmp_dir)
            payload["datasets"]["megadepth"]["scenes"] = "*"
            payload["datasets"]["megadepth"]["selection"] = "all"
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")

            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    "prep_scene_info/0002.npz",
                    "prep_scene_info/0001.npz",
                ],
            ):
                units = list(pipeline.enumerate_download_units())
            self.assertEqual(
                units,
                [
                    MegaDepthDownloadUnit(unit_name="0001"),
                    MegaDepthDownloadUnit(unit_name="0002"),
                ],
            )

    def test_status_paths_do_not_require_scene_discovery_after_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = self._make_config(tmp_dir)
            payload["datasets"]["megadepth"]["scenes"] = "*"
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")

            self.assertEqual(list(pipeline.iter_download_artifact_paths()), [pipeline._bundle_download_root()])
            self.assertEqual(
                tuple(pipeline.enumerate_extraction_units()),
                (MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle_extract"),),
            )

    def test_bundle_mode_download_setup_does_not_require_scene_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = self._make_config(tmp_dir)
            payload["datasets"]["megadepth"]["scenes"] = "*"
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")

            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    "MegaDepth_v1.tar.gz_part00",
                    "MegaDepth_v1.tar.gz_part01",
                ],
            ):
                self.assertEqual(
                    pipeline.get_selected_download_units(),
                    [MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle")],
                )

    def test_bundle_download_does_not_require_scene_info_listing_for_wildcard_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            payload = self._make_config(tmp_dir)
            payload["datasets"]["megadepth"]["scenes"] = "*"
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")
            source_root = Path(source_dir)
            (source_root / "MegaDepth_v1.tar.gz_part00").write_bytes(b"part0")
            (source_root / "MegaDepth_v1.tar.gz_part01").write_bytes(b"part1")

            def fake_download_hf_file(repo_id: str, repo_path: str, target_path: Path) -> Path:
                del repo_id
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / Path(repo_path).name, target_path)
                return target_path

            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    "MegaDepth_v1.tar.gz_part00",
                    "MegaDepth_v1.tar.gz_part01",
                ],
            ):
                with patch.object(pipeline, "_download_hf_file", side_effect=fake_download_hf_file):
                    pipeline.download_unit(MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle"))

            self.assertTrue((pipeline.paths.raw / "_downloads" / "megadepth_bundle" / "MegaDepth_v1.tar.gz_part00").exists())
            self.assertTrue((pipeline.paths.raw / "_downloads" / "megadepth_bundle" / "MegaDepth_v1.tar.gz_part01").exists())

    def test_incomplete_bundle_blocks_extraction_until_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            payload = self._make_config(tmp_dir)
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")
            source_root = Path(source_dir)
            self._write_scene_layout(source_root)

            repo_files = [
                "MegaDepth_v1.tar.gz_part00",
                "MegaDepth_v1.tar.gz_part01",
                "prep_scene_info/0015.npz",
            ]
            with patch.object(pipeline, "_list_hf_files", return_value=repo_files):
                download_root = pipeline.paths.raw / "_downloads" / "megadepth_bundle"
                download_root.mkdir(parents=True, exist_ok=True)
                (download_root / "MegaDepth_v1.tar.gz_part00").write_bytes(b"partial")
                scene_info_path = pipeline.paths.raw / "prep_scene_info" / "0015.npz"
                scene_info_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / "prep_scene_info" / "0015.npz", scene_info_path)
                with self.assertRaisesRegex(FileNotFoundError, "requires the complete multipart bundle"):
                    pipeline.extract_unit(MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle_extract"))

    def test_bundle_download_resumes_after_interruption_and_completes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            payload = self._make_config(tmp_dir)
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload))
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")
            source_root = Path(source_dir)
            self._write_scene_layout(source_root)

            part0 = source_root / "MegaDepth_v1.tar.gz_part00"
            part1 = source_root / "MegaDepth_v1.tar.gz_part01"
            part0.write_bytes(b"part0")
            part1.write_bytes(b"part1")

            download_root = pipeline.paths.raw / "_downloads" / "megadepth_bundle"
            download_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(part0, download_root / part0.name)

            repo_files = [
                "MegaDepth_v1.tar.gz_part00",
                "MegaDepth_v1.tar.gz_part01",
                "prep_scene_info/0015.npz",
            ]
            requested_repo_paths: list[str] = []

            def fake_download_hf_file(repo_id: str, repo_path: str, target_path: Path) -> Path:
                del repo_id
                requested_repo_paths.append(repo_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source_path = source_root / "prep_scene_info" / "0015.npz" if repo_path.startswith("prep_scene_info/") else source_root / Path(repo_path).name
                shutil.copy2(source_path, target_path)
                return target_path

            with patch.object(pipeline, "_list_hf_files", return_value=repo_files):
                with patch.object(pipeline, "_download_hf_file", side_effect=fake_download_hf_file):
                    unit = MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle")
                    pipeline.download_unit(unit)

            self.assertEqual(requested_repo_paths, ["MegaDepth_v1.tar.gz_part01", "prep_scene_info/0015.npz"])
            self.assertTrue(pipeline.is_download_unit_satisfied(MegaDepthDownloadUnit("megadepth_bundle", "bundle")))
            self.assertTrue((download_root / "MegaDepth_v1.tar.gz_part00").exists())
            self.assertTrue((download_root / "MegaDepth_v1.tar.gz_part01").exists())
            self.assertTrue((pipeline.paths.raw / "prep_scene_info" / "0015.npz").exists())


if __name__ == "__main__":
    unittest.main()
