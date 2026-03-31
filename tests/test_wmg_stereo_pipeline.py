import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import (
    WMGStereoFlyingPipeline,
    WMGStereoIndoorPipeline,
    WMGStereoNaturePipeline,
)
from depth_collector.datasets.wmg_stereo import WMGStereoArchiveUnit, WMGStereoSourceItem


class WMGStereoPipelineTest(unittest.TestCase):
    def _make_config(self, root_data_dir: str, dataset_name: str = "wmg_stereo_flying") -> dict[str, object]:
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
                dataset_name: {
                    "enabled": True,
                    "hf_dataset_id": "princeton-vl/WMGStereo",
                    "selection": "minimum_readable",
                    "release": "release_full",
                    "archives": ["seed0001.tar.gz"],
                }
            },
        }

    def _make_pipeline(
        self,
        tmp_dir: str,
        pipeline_type: type[WMGStereoFlyingPipeline] = WMGStereoFlyingPipeline,
        dataset_name: str = "wmg_stereo_flying",
    ) -> WMGStereoFlyingPipeline:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(self._make_config(tmp_dir, dataset_name=dataset_name)))
        return pipeline_type(load_config(config_path), dataset_name)

    def _write_archive(self, archive_path: Path, *, seed_name: str = "seed0001") -> None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as build_dir:
            build_root = Path(build_dir) / seed_name
            image_root = build_root / "frames" / "Image" / "camera_0"
            disparity_root = build_root / "frames" / "disparity" / "camera_0"
            camview0_root = build_root / "frames" / "camview" / "camera_0"
            camview1_root = build_root / "frames" / "camview" / "camera_1"
            sky_mask_root = build_root / "frames" / "sky_mask" / "camera_0"

            for root in (image_root, disparity_root, camview0_root, camview1_root, sky_mask_root):
                root.mkdir(parents=True, exist_ok=True)

            image = np.zeros((4, 6, 3), dtype=np.uint8)
            image[..., 0] = 64
            image[..., 1] = 128
            image[..., 2] = 255
            image_path = image_root / "Image_0_0_0001_0.png"
            Image.fromarray(image).save(image_path)

            disparity = np.ones((4, 6), dtype=np.float32)
            np.save(disparity_root / "disparity_0_0_0001_0.npy", disparity)

            k_matrix = np.array(
                [
                    [4.0, 0.0, 3.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            hw = np.array([4, 6], dtype=np.int32)
            np.savez(camview0_root / "camview_0_0_0001_0.npz", K=k_matrix, T=np.array([0.0, 0.0, 0.0]), HW=hw)
            np.savez(camview1_root / "camview_0_1_0001_0.npz", K=k_matrix, T=np.array([0.5, 0.0, 0.0]), HW=hw)

            sky_mask = np.zeros((4, 6), dtype=np.uint8)
            sky_mask[0, 0] = 255
            Image.fromarray(sky_mask).save(sky_mask_root / "skymask_0_0_0001_0.png")

            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(build_root, arcname=seed_name)

    def test_concrete_pipelines_use_distinct_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            flying = self._make_pipeline(tmp_dir, pipeline_type=WMGStereoFlyingPipeline, dataset_name="wmg_stereo_flying")
            indoor = self._make_pipeline(tmp_dir, pipeline_type=WMGStereoIndoorPipeline, dataset_name="wmg_stereo_indoor")
            nature = self._make_pipeline(tmp_dir, pipeline_type=WMGStereoNaturePipeline, dataset_name="wmg_stereo_nature")

            self.assertEqual(flying.CATEGORY_NAME, "flying")
            self.assertEqual(indoor.CATEGORY_NAME, "indoor")
            self.assertEqual(nature.CATEGORY_NAME, "nature")
            self.assertIn("/relative/wmg_stereo_flying", str(flying.paths.root))
            self.assertIn("/relative/wmg_stereo_indoor", str(indoor.paths.root))
            self.assertIn("/relative/wmg_stereo_nature", str(nature.paths.root))

    def test_archive_enumeration_discovers_category_specific_remote_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            pipeline.dataset_config.options["archives"] = "*"
            pipeline.dataset_config.options["selection"] = "all"
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "release_full/flying/seed0001.tar.gz",
                    "release_full/flying/seed0002.tar.gz",
                    "release_full/indoor/seed9000.tar.gz",
                ],
            ):
                units = list(pipeline.enumerate_download_units())

            self.assertEqual(
                units,
                [
                    WMGStereoArchiveUnit(
                        category="flying",
                        archive_name="seed0001.tar.gz",
                        repo_path="release_full/flying/seed0001.tar.gz",
                    ),
                    WMGStereoArchiveUnit(
                        category="flying",
                        archive_name="seed0002.tar.gz",
                        repo_path="release_full/flying/seed0002.tar.gz",
                    ),
                ],
            )

    def test_minimum_readable_download_materializes_single_readable_sample_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = WMGStereoArchiveUnit(
                category="flying",
                archive_name="seed0001.tar.gz",
                repo_path="release_full/flying/seed0001.tar.gz",
            )
            local_archive_root = Path(tmp_dir) / "source_archives"
            pipeline.dataset_config.options["local_archive_root"] = str(local_archive_root)
            source_archive_path = local_archive_root / "release_full" / "flying" / "seed0001.tar.gz"
            self._write_archive(source_archive_path)

            pipeline.download_unit(unit)

            archive_path = pipeline._archive_path(unit)
            with tarfile.open(archive_path, "r:gz") as archive:
                member_names = sorted(member.name for member in archive if member.isfile())
            self.assertEqual(
                member_names,
                [
                    "seed0001/frames/Image/camera_0/Image_0_0_0001_0.png",
                    "seed0001/frames/camview/camera_0/camview_0_0_0001_0.npz",
                    "seed0001/frames/camview/camera_1/camview_0_1_0001_0.npz",
                    "seed0001/frames/disparity/camera_0/disparity_0_0_0001_0.npy",
                ],
            )

    def test_minimum_readable_remote_download_uses_streaming_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = WMGStereoArchiveUnit(
                category="flying",
                archive_name="seed0001.tar.gz",
                repo_path="release_full/flying/seed0001.tar.gz",
            )
            with tempfile.TemporaryDirectory() as build_dir:
                source_archive_path = Path(build_dir) / "seed0001.tar.gz"
                self._write_archive(source_archive_path)
                source_bytes = source_archive_path.read_bytes()

            class _RemoteBytes(io.BytesIO):
                def __enter__(self) -> "_RemoteBytes":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    self.close()

            with patch.object(pipeline, "hf_open_remote_file", return_value=_RemoteBytes(source_bytes)) as mocked_open:
                pipeline.download_unit(unit)

            mocked_open.assert_called_once()
            archive_path = pipeline._archive_path(unit)
            self.assertTrue(archive_path.exists())

    def test_extract_enumerate_and_build_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._make_pipeline(tmp_dir)
            unit = WMGStereoArchiveUnit(
                category="flying",
                archive_name="seed0001.tar.gz",
                repo_path="release_full/flying/seed0001.tar.gz",
            )
            self._write_archive(pipeline._archive_path(unit))

            pipeline.extract_unit(unit)
            items = list(pipeline.enumerate_source_items())
            self.assertEqual(
                items,
                [
                    WMGStereoSourceItem(
                        seed_name="seed0001",
                        frame_key="0_0001",
                        archive_name="seed0001.tar.gz",
                        image_relative_path="seed0001/frames/Image/camera_0/Image_0_0_0001_0.png",
                        disparity_relative_path="seed0001/frames/disparity/camera_0/disparity_0_0_0001_0.npy",
                        left_camview_relative_path="seed0001/frames/camview/camera_0/camview_0_0_0001_0.npz",
                        right_camview_relative_path="seed0001/frames/camview/camera_1/camview_0_1_0001_0.npz",
                        occ_mask_relative_path=None,
                        sky_mask_relative_path="seed0001/frames/sky_mask/camera_0/skymask_0_0_0001_0.png",
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
            self.assertGreaterEqual(float(np.min(sample.distance)), 0.0)
            self.assertLessEqual(float(np.max(sample.distance)), 1.0)
            self.assertAlmostEqual(float(np.max(sample.distance)), 1.0, places=5)
            self.assertAlmostEqual(float(sample.distance[0, 0, 0]), 1.0, places=5)
            self.assertGreater(float(sample.distance[2, 3, 0]), 0.0)
            self.assertLess(float(sample.distance[2, 3, 0]), 1.0)
            self.assertEqual(sample.provenance["category"], "flying")
            self.assertEqual(sample.provenance["seed_name"], "seed0001")
            self.assertEqual(sample.provenance["distance_normalization"], "[0, 1]")


if __name__ == "__main__":
    unittest.main()
