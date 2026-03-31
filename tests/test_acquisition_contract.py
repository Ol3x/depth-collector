import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import _bootstrap  # noqa: F401
from depth_collector.config import load_config
from depth_collector.datasets import HypersimPipeline, MegaDepthPipeline, NYUDepthV2Pipeline, TartanAirPipeline, TartanGroundPipeline
from depth_collector.datasets import UnrealStereo4KPipeline
from depth_collector.datasets.diode import DIODEArchiveUnit, DIODEPipeline
from depth_collector.datasets.hypersim import HypersimSceneUnit
from depth_collector.datasets.megadepth import MegaDepthDownloadUnit
from depth_collector.datasets.nyu_depth_v2 import NYUDepthV2ArchiveUnit
from depth_collector.datasets.unrealstereo4k import UnrealStereo4KArchiveUnit
from depth_collector.datasets.tartanair import TartanAirArchiveUnit
from depth_collector.datasets.tartanground import TartanGroundArchiveUnit
from depth_collector.datasets.topair import TopAirPipeline, TopAirTrajectoryUnit
from depth_collector.datasets.tof_360 import ToF360Pipeline, ToF360SceneUnit
from depth_collector.datasets.urbansyn import UrbanSynFrameUnit, UrbanSynPipeline
from depth_collector.datasets.virtual_kitti_2 import VirtualKITTI2ArchiveUnit, VirtualKITTI2Pipeline
from depth_collector.datasets.wmg_stereo import WMGStereoArchiveUnit
from depth_collector.datasets.wmg_stereo_flying import WMGStereoFlyingPipeline


class _HfHelperCalled(RuntimeError):
    pass


class AcquisitionContractTest(unittest.TestCase):
    def _write_config(self, tmp_dir: str, datasets: dict[str, object]) -> Path:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "project": {
                        "name": "default",
                        "description": "acquisition contract",
                        "max_dist": 100.0,
                        "train_val_split": 0.95,
                    },
                    "runtime": {
                        "download_workers": 1,
                        "process_ratio": 1.0,
                        "shuffle_seed": 0,
                        "resume": True,
                        "skip_known_errors": True,
                        "write_error_traces": True,
                        "target_shard_size_gb": 1.0,
                    },
                    "output": {
                        "root_data_dir": tmp_dir,
                        "raw_subdir_name": "raw",
                        "processed_subdir_name": "processed",
                        "state_subdir_name": "state",
                        "metadata_filename": "metadata.json",
                    },
                    "datasets": datasets,
                }
            )
        )
        return config_path

    def test_dataset_modules_do_not_directly_use_non_shared_download_clients(self) -> None:
        dataset_files = [
            Path("src/depth_collector/datasets/tartan.py"),
            Path("src/depth_collector/datasets/tartanground.py"),
            Path("src/depth_collector/datasets/tartanair.py"),
            Path("src/depth_collector/datasets/diode.py"),
            Path("src/depth_collector/datasets/hypersim.py"),
            Path("src/depth_collector/datasets/megadepth.py"),
            Path("src/depth_collector/datasets/nyu_depth_v2.py"),
            Path("src/depth_collector/datasets/topair.py"),
            Path("src/depth_collector/datasets/unrealstereo4k.py"),
            Path("src/depth_collector/datasets/tof_360.py"),
            Path("src/depth_collector/datasets/urbansyn.py"),
            Path("src/depth_collector/datasets/virtual_kitti_2.py"),
            Path("src/depth_collector/datasets/wmg_stereo.py"),
            Path("src/depth_collector/datasets/wmg_stereo_flying.py"),
        ]
        forbidden_snippets = [
            "from huggingface_hub import",
            "import huggingface_hub",
            "urlretrieve(",
            "requests.get(",
        ]
        required_helper_snippets = [
            "self.hf_hub_download(",
            "self.hf_snapshot_download(",
            "self.hf_list_repo_files(",
        ]

        for path in dataset_files:
            source = path.read_text()
            for snippet in forbidden_snippets:
                self.assertNotIn(snippet, source, msg=f"{path} should not use {snippet} directly")
        combined_source = "\n".join(path.read_text() for path in dataset_files)
        self.assertTrue(any(snippet in combined_source for snippet in required_helper_snippets))

    def test_dataset_modules_implement_shared_selection_contract(self) -> None:
        dataset_files = [
            Path("src/depth_collector/datasets/diode.py"),
            Path("src/depth_collector/datasets/hypersim.py"),
            Path("src/depth_collector/datasets/megadepth.py"),
            Path("src/depth_collector/datasets/nyu_depth_v2.py"),
            Path("src/depth_collector/datasets/tartan.py"),
            Path("src/depth_collector/datasets/topair.py"),
            Path("src/depth_collector/datasets/unrealstereo4k.py"),
            Path("src/depth_collector/datasets/tof_360.py"),
            Path("src/depth_collector/datasets/urbansyn.py"),
            Path("src/depth_collector/datasets/virtual_kitti_2.py"),
            Path("src/depth_collector/datasets/wmg_stereo.py"),
        ]
        for path in dataset_files:
            source = path.read_text()
            self.assertTrue(
                "apply_dataset_selection(" in source or "dataset_selection()" in source,
                msg=f"{path} should use the shared selection contract",
            )

    def test_tartanair_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "tartanair": {
                        "enabled": True,
                        "hf_dataset_id": "theairlabcmu/tartanair",
                        "selection": "all",
                        "environments": ["neighborhood"],
                        "difficulties": ["Easy"],
                        "modalities": ["image_left"],
                    }
                },
            )
            pipeline = TartanAirPipeline(load_config(config_path), "tartanair")
            unit = TartanAirArchiveUnit(environment="neighborhood", difficulty="Easy", modality="image_left")
            with patch.object(pipeline, "hf_hub_download", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_diode_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "diode_subset_train": {
                        "enabled": True,
                        "hf_dataset_id": "sayakpaul/diode-subset-train",
                        "selection": "minimum_readable",
                        "archive_filename": "train_subset.tar.gz",
                        "splits": ["train"],
                    }
                },
            )
            pipeline = DIODEPipeline(load_config(config_path), "diode_subset_train")
            unit = DIODEArchiveUnit(archive_name="train_subset.tar.gz")
            with patch.object(pipeline, "hf_open_remote_file", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_tartanground_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "tartanground": {
                        "enabled": True,
                        "hf_dataset_id": "theairlabcmu/TartanGround",
                        "selection": "all",
                        "environments": ["AbandonedCable"],
                        "versions": ["omni"],
                        "trajectories": ["P0000"],
                        "camera_names": ["lcam_front"],
                        "modalities": ["image"],
                    }
                },
            )
            pipeline = TartanGroundPipeline(load_config(config_path), "tartanground")
            unit = TartanGroundArchiveUnit(
                environment="AbandonedCable",
                version="omni",
                trajectory="P0000",
                modality="image",
                camera_name="lcam_front",
            )
            with patch.object(pipeline, "hf_hub_download", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_hypersim_archive_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "hypersim": {
                        "enabled": True,
                        "hf_dataset_id": "ritianyu/Hypersim",
                        "selection": "minimum_readable",
                        "download_mode": "archive",
                        "scenes": ["ai_001_001"],
                    }
                },
            )
            pipeline = HypersimPipeline(load_config(config_path), "hypersim")
            unit = HypersimSceneUnit(scene_name="ai_001_001")
            with patch.object(pipeline, "hf_open_remote_zip", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_hypersim_directory_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "hypersim": {
                        "enabled": True,
                        "hf_dataset_id": "ritianyu/Hypersim",
                        "selection": "minimum_readable",
                        "download_mode": "directory",
                        "scenes": ["ai_001_001"],
                    }
                },
            )
            pipeline = HypersimPipeline(load_config(config_path), "hypersim")
            unit = HypersimSceneUnit(scene_name="ai_001_001")
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "ai_001_001/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg",
                    "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5",
                    "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters_plane.npz",
                    "ai_001_001/_detail/cam_00/camera_keyframe_orientations.hdf5",
                    "ai_001_001/_detail/cam_00/camera_keyframe_positions.hdf5",
                    "ai_001_001/_detail/metadata_scene.csv",
                    "metadata_camera_parameters.csv",
                ],
            ):
                with patch.object(pipeline, "hf_hub_download", side_effect=_HfHelperCalled):
                    with self.assertRaises(_HfHelperCalled):
                        pipeline.download_unit(unit)

    def test_megadepth_bundle_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "megadepth": {
                        "enabled": True,
                        "hf_dataset_id": "ssbai/MegaDepth_v1",
                        "selection": "minimum_readable",
                        "bundles": ["megadepth_bundle"],
                        "scene_info_dir": "prep_scene_info",
                        "scenes": ["0015"],
                    }
                },
            )
            pipeline = MegaDepthPipeline(load_config(config_path), "megadepth")
            unit = MegaDepthDownloadUnit(unit_name="megadepth_bundle", unit_type="bundle")
            with patch.object(
                pipeline,
                "_list_hf_files",
                return_value=[
                    "MegaDepth_v1.tar.gz_part00",
                    "prep_scene_info/0015.npz",
                ],
            ):
                with patch.object(pipeline, "hf_hub_download", side_effect=_HfHelperCalled):
                    with self.assertRaises(_HfHelperCalled):
                        pipeline.download_unit(unit)

    def test_wmg_stereo_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "wmg_stereo_flying": {
                        "enabled": True,
                        "hf_dataset_id": "princeton-vl/WMGStereo",
                        "selection": "minimum_readable",
                        "release": "release_full",
                        "archives": ["seed0001.tar.gz"],
                    }
                },
            )
            pipeline = WMGStereoFlyingPipeline(load_config(config_path), "wmg_stereo_flying")
            unit = WMGStereoArchiveUnit(
                category="flying",
                archive_name="seed0001.tar.gz",
                repo_path="release_full/flying/seed0001.tar.gz",
            )
            with patch.object(pipeline, "hf_open_remote_file", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_urbansyn_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "urbansyn": {
                        "enabled": True,
                        "hf_dataset_id": "UrbanSyn/UrbanSyn",
                        "selection": "minimum_readable",
                        "frames": ["0001"],
                        "use_semantic_masks": True,
                        "camera_intrinsics": {
                            "width": 2048,
                            "height": 1024,
                            "fx": 2262.52,
                            "fy": 2265.30,
                            "cx": 1096.98,
                            "cy": 513.137,
                        },
                    }
                },
            )
            pipeline = UrbanSynPipeline(load_config(config_path), "urbansyn")
            unit = UrbanSynFrameUnit(frame_id="0001")
            with patch.object(pipeline, "hf_hub_download", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_tof_360_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "tof_360": {
                        "enabled": True,
                        "hf_dataset_id": "COLE-Ricoh/ToF-360",
                        "selection": "minimum_readable",
                        "scenes": ["scene_0001"],
                        "rgb_dir": "rgb",
                        "depth_dir": "depth",
                        "depth_scale_divisor": 512.0,
                    }
                },
            )
            pipeline = ToF360Pipeline(load_config(config_path), "tof_360")
            unit = ToF360SceneUnit(scene_name="scene_0001")
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "scene_0001/rgb/frame_0001.png",
                    "scene_0001/depth/frame_0001.png",
                ],
            ):
                with patch.object(pipeline, "hf_snapshot_download", side_effect=_HfHelperCalled):
                    with self.assertRaises(_HfHelperCalled):
                        pipeline.download_unit(unit)

    def test_nyu_depth_v2_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "nyu_depth_v2": {
                        "enabled": True,
                        "hf_dataset_id": "sayakpaul/nyu_depth_v2",
                        "selection": "minimum_readable",
                        "shards": ["val-000001.tar"],
                    }
                },
            )
            pipeline = NYUDepthV2Pipeline(load_config(config_path), "nyu_depth_v2")
            unit = NYUDepthV2ArchiveUnit(repo_path="data/val-000001.tar")
            with patch.object(pipeline, "hf_hub_download", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_topair_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "topair": {
                        "enabled": True,
                        "hf_dataset_id": "yaraalaa0/TopAir",
                        "selection": "minimum_readable",
                        "trajectories": ["AssetsvilleTown_2"],
                        "use_semantic_masks": True,
                        "sky_class_id": 0,
                    }
                },
            )
            pipeline = TopAirPipeline(load_config(config_path), "topair")
            unit = TopAirTrajectoryUnit(trajectory_name="AssetsvilleTown_2")
            with patch.object(
                pipeline,
                "hf_list_repo_files",
                return_value=[
                    "AssetsvilleTown_2/images/0001.png",
                    "AssetsvilleTown_2/depth/0001.png",
                    "AssetsvilleTown_2/seg_id/0001.png",
                    "AssetsvilleTown_2/camera_loc.txt",
                ],
            ):
                with patch.object(pipeline, "hf_snapshot_download", side_effect=_HfHelperCalled):
                    with self.assertRaises(_HfHelperCalled):
                        pipeline.download_unit(unit)

    def test_virtual_kitti_2_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "virtual_kitti_2": {
                        "enabled": True,
                        "hf_dataset_id": "ZhengGuangze/VKITTI2_vlbm",
                        "selection": "minimum_readable",
                        "archive_filename": "vkitti2_vlbm.tar.gz",
                    }
                },
            )
            pipeline = VirtualKITTI2Pipeline(load_config(config_path), "virtual_kitti_2")
            unit = VirtualKITTI2ArchiveUnit(archive_name="vkitti2_vlbm.tar.gz")
            with patch.object(pipeline, "hf_open_remote_file", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)

    def test_unrealstereo4k_remote_download_uses_hf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "unrealstereo4k": {
                        "enabled": True,
                        "hf_dataset_id": "fabiotosi92/UnrealStereo4K",
                        "selection": "minimum_readable",
                        "archives": ["00008.zip"],
                    }
                },
            )
            pipeline = UnrealStereo4KPipeline(load_config(config_path), "unrealstereo4k")
            unit = UnrealStereo4KArchiveUnit(archive_name="00008.zip", repo_path="00008.zip")
            with patch.object(pipeline, "hf_open_remote_zip", side_effect=_HfHelperCalled):
                with self.assertRaises(_HfHelperCalled):
                    pipeline.download_unit(unit)


if __name__ == "__main__":
    unittest.main()
