import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import HypersimPipeline
from depth_collector.datasets.hypersim import HypersimSceneUnit, HypersimSourceItem


class HypersimPipelineTest(unittest.TestCase):
    def _make_config(
        self,
        root_data_dir: str,
        selection: object = "minimum_readable",
        process_ratio: float = 1.0,
        download_mode: str = "archive",
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
                "hypersim": {
                    "enabled": True,
                    "hf_dataset_id": "GaussianWorld/Hypersim",
                    "selection": selection,
                    "download_mode": download_mode,
                    "scenes": ["ai_001_001"],
                    "camera_trajectories": ["cam_00"],
                }
            },
        }

    def _make_pipeline(self, tmp_dir: str, *, selection: object = "minimum_readable", download_mode: str = "archive") -> HypersimPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir, selection=selection, download_mode=download_mode)))
        return HypersimPipeline(load_config(config_path), "hypersim")

    def _write_hdf5(self, path: Path, array: np.ndarray) -> None:
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("dataset", data=array)

    def test_all_scene_selector_discovers_remote_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.dataset_config.options["scenes"] = "*"
            with patch.object(pipeline, "_list_remote_scene_names", return_value=["ai_001_001", "ai_002_002"]):
                units = list(pipeline.enumerate_download_units())
            self.assertEqual(
                units,
                [
                    HypersimSceneUnit(scene_name="ai_001_001"),
                ],
            )

    def test_bad_scenes_are_filtered_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.dataset_config.options["scenes"] = ["ai_001_001", "ai_001_002", "ai_001_003"]
            pipeline.dataset_config.options["bad_scenes"] = ["ai_001_001"]
            units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [HypersimSceneUnit(scene_name="ai_001_002")])

    def _write_test_archive(self, pipeline: HypersimPipeline, unit: HypersimSceneUnit) -> Path:
        archive_path = pipeline._archive_path(unit)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as build_dir:
            build_root = Path(build_dir)
            scene_root = Path(build_dir) / unit.scene_name
            (scene_root / "_detail" / "cam_00").mkdir(parents=True, exist_ok=True)
            (scene_root / "images" / "scene_cam_00_final_preview").mkdir(parents=True, exist_ok=True)
            (scene_root / "images" / "scene_cam_00_geometry_hdf5").mkdir(parents=True, exist_ok=True)

            metadata_scene = scene_root / "_detail" / "metadata_scene.csv"
            metadata_scene.write_text("parameter_name,parameter_value\nmeters_per_asset_unit,2.0\n")
            (build_root / "metadata_camera_parameters.csv").write_text(
                "scene_name,settings_output_img_height,settings_output_img_width,settings_units_info_meters_scale,"
                "M_cam_from_uv_00,M_cam_from_uv_01,M_cam_from_uv_02,"
                "M_cam_from_uv_10,M_cam_from_uv_11,M_cam_from_uv_12,"
                "M_cam_from_uv_20,M_cam_from_uv_21,M_cam_from_uv_22\n"
                "ai_001_001,2,2,2.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,-1.0\n"
            )

            orientations = np.eye(3, dtype=np.float32)[None, ...]
            camera_positions = np.zeros((1, 3), dtype=np.float32)
            self._write_hdf5(scene_root / "_detail" / "cam_00" / "camera_keyframe_orientations.hdf5", orientations)
            self._write_hdf5(scene_root / "_detail" / "cam_00" / "camera_keyframe_positions.hdf5", camera_positions)

            color = np.array(
                [
                    [[255, 128, 64], [128, 64, 32]],
                    [[64, 32, 16], [32, 16, 8]],
                ],
                dtype=np.uint8,
            )
            depth_plane_meters = np.ones((2, 2), dtype=np.float32)
            uv1 = np.array(
                [
                    [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0]],
                    [[-0.5, 0.5, 1.0], [0.5, 0.5, 1.0]],
                ],
                dtype=np.float32,
            )
            points_camera_hypersim = uv1 * depth_plane_meters[..., None]
            depth_meters = np.linalg.norm(points_camera_hypersim, axis=-1).astype(np.float32)
            Image.fromarray(color, mode="RGB").save(
                scene_root / "images" / "scene_cam_00_final_preview" / "frame.0000.tonemap.jpg"
            )
            self._write_hdf5(
                scene_root / "images" / "scene_cam_00_geometry_hdf5" / "frame.0000.depth_meters.hdf5",
                depth_meters,
            )
            np.savez(
                scene_root / "images" / "scene_cam_00_geometry_hdf5" / "frame.0000.depth_meters_plane.npz",
                data=depth_plane_meters,
            )

            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in build_root.rglob("*"):
                    if path.is_file():
                        archive.write(path, arcname=str(path.relative_to(build_root)))

        return archive_path

    def _write_test_scene_directory(self, root: Path, scene_name: str) -> Path:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_pipeline = self._make_pipeline(tmp_dir)
            archive_path = self._write_test_archive(temp_pipeline, HypersimSceneUnit(scene_name=scene_name))
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(root)
        return root / scene_name

    def test_scene_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            units = list(pipeline.enumerate_download_units())
            self.assertEqual(units, [HypersimSceneUnit(scene_name="ai_001_001")])

    def test_extract_and_enumerate_source_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = HypersimSceneUnit(scene_name="ai_001_001")
            self._write_test_archive(pipeline, unit)

            pipeline.extract_unit(unit)
            items = list(pipeline.enumerate_source_items())

            self.assertEqual(
                items,
                [
                    HypersimSourceItem(
                        scene_name="ai_001_001",
                        camera_name="cam_00",
                        frame_id="frame.0000",
                        color_relative_path="images/scene_cam_00_final_preview/frame.0000.tonemap.jpg",
                        depth_relative_path="images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5",
                        depth_plane_relative_path="images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters_plane.npz",
                    )
                ],
            )

    def test_directory_mode_downloads_scene_tree_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, download_mode="directory", selection="all")
            with tempfile.TemporaryDirectory() as source_dir:
                source_root = Path(source_dir)
                self._write_test_scene_directory(source_root, "ai_001_001")
                pipeline.dataset_config.options["local_archive_root"] = str(source_root)

                unit = HypersimSceneUnit(scene_name="ai_001_001")
                pipeline.download_unit(unit)

            scene_root = pipeline._scene_root("ai_001_001")
            self.assertTrue(scene_root.exists())
            self.assertTrue(
                (
                    scene_root
                    / "images"
                    / "scene_cam_00_geometry_hdf5"
                    / "frame.0000.depth_meters.hdf5"
                ).exists()
            )
            self.assertEqual(list(pipeline.enumerate_extraction_units()), [])

    def test_minimum_readable_directory_mode_materializes_single_sample_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir, download_mode="directory")
            with tempfile.TemporaryDirectory() as source_dir:
                source_root = Path(source_dir)
                scene_root = self._write_test_scene_directory(source_root, "ai_001_001")
                final_root = scene_root / "images" / "scene_cam_00_final_preview"
                geometry_root = scene_root / "images" / "scene_cam_00_geometry_hdf5"
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(final_root / "frame.0001.tonemap.jpg")
                self._write_hdf5(geometry_root / "frame.0001.depth_meters.hdf5", np.ones((2, 2), dtype=np.float32))
                np.savez(geometry_root / "frame.0001.depth_meters_plane.npz", data=np.ones((2, 2), dtype=np.float32))
                orientations = np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)], axis=0)
                positions = np.zeros((2, 3), dtype=np.float32)
                self._write_hdf5(scene_root / "_detail" / "cam_00" / "camera_keyframe_orientations.hdf5", orientations)
                self._write_hdf5(scene_root / "_detail" / "cam_00" / "camera_keyframe_positions.hdf5", positions)
                pipeline.dataset_config.options["local_archive_root"] = str(source_root)

                unit = HypersimSceneUnit(scene_name="ai_001_001")
                pipeline.download_unit(unit)

            downloaded_scene_root = pipeline._scene_root("ai_001_001")
            self.assertTrue((downloaded_scene_root / "images" / "scene_cam_00_final_preview" / "frame.0000.tonemap.jpg").exists())
            self.assertFalse((downloaded_scene_root / "images" / "scene_cam_00_final_preview" / "frame.0001.tonemap.jpg").exists())
            frame_index_map = json.loads((downloaded_scene_root / "_detail" / "cam_00" / "frame_index_map.json").read_text())
            self.assertEqual(frame_index_map, {"frame.0000": 0})

    def test_minimum_readable_archive_mode_materializes_single_sample_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as source_dir:
            pipeline = self._make_pipeline(tmp_dir)
            source_pipeline = self._make_pipeline(source_dir)
            unit = HypersimSceneUnit(scene_name="ai_001_001")
            archive_path = self._write_test_archive(source_pipeline, unit)
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("ai_001_001/images/scene_cam_00_final_preview/frame.0001.tonemap.jpg", b"extra")
            pipeline.dataset_config.options["local_archive_root"] = source_pipeline.paths.raw

            pipeline.download_unit(unit)

            with zipfile.ZipFile(pipeline._archive_path(unit)) as archive:
                names = sorted(archive.namelist())
            self.assertIn("ai_001_001/_detail/cam_00/frame_index_map.json", names)
            self.assertIn("metadata_camera_parameters.csv", names)
            self.assertNotIn("ai_001_001/images/scene_cam_00_final_preview/frame.0001.tonemap.jpg", names)

    def test_build_sample_uses_scale_and_camera_convention_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = HypersimSceneUnit(scene_name="ai_001_001")
            self._write_test_archive(pipeline, unit)
            pipeline.extract_unit(unit)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            sample = pipeline.build_sample(item, loaded_item, camera_model)

            self.assertEqual(sample.image.shape, (2, 2, 3))
            self.assertEqual(sample.distance.shape, (2, 2, 1))
            self.assertEqual(sample.ray_dir.shape, (2, 2, 3))
            self.assertTrue(np.all(sample.image >= 0.0))
            self.assertTrue(np.all(sample.image <= 1.0))
            expected_distance = np.sqrt(1.5, dtype=np.float32)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), float(expected_distance), places=5)
            np.testing.assert_allclose(
                sample.ray_dir[0, 0],
                np.array([0.40824828, 0.40824828, 0.81649655], dtype=np.float32),
                atol=1e-5,
            )
            self.assertEqual(sample.provenance["scene_name"], "ai_001_001")
            self.assertEqual(sample.provenance["camera_name"], "cam_00")

    def test_build_sample_rejects_plane_depth_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = HypersimSceneUnit(scene_name="ai_001_001")
            self._write_test_archive(pipeline, unit)
            pipeline.extract_unit(unit)

            item = next(iter(pipeline.enumerate_source_items()))
            loaded_item = pipeline.load_source_item(item)
            camera_model = pipeline.build_camera_model(item, loaded_item)
            loaded_item["m_cam_from_uv"] = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            loaded_item["depth_meters"] = np.asarray(loaded_item["depth_plane_meters"], dtype=np.float32)

            with self.assertRaisesRegex(ValueError, "suspiciously close to depth_meters_plane"):
                pipeline.build_sample(item, loaded_item, camera_model)


if __name__ == "__main__":
    unittest.main()
