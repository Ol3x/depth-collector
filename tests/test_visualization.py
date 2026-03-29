import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
from depth_collector.core import SampleRecord
from depth_collector.geometry import PinholeCameraModel, generate_pinhole_rays
from depth_collector.visualization import create_contact_sheet


class VisualizationTest(unittest.TestCase):
    def _sample(self, sample_id: str, provenance: dict[str, object]) -> SampleRecord:
        image = np.zeros((4, 4, 3), dtype=np.float32)
        distance = np.ones((4, 4, 1), dtype=np.float32)
        ray_dir = generate_pinhole_rays(
            PinholeCameraModel(width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
        ).astype(np.float32)
        return SampleRecord(
            sample_id=sample_id,
            image=image,
            distance=distance,
            ray_dir=ray_dir,
            provenance=provenance,
        )

    def test_contact_sheets_are_grouped_by_scene_name(self) -> None:
        samples = [
            self._sample("ai_001_001/cam_00/frame.0000", {"scene_name": "ai_001_001", "camera_name": "cam_00"}),
            self._sample("ai_001_001/cam_00/frame.0001", {"scene_name": "ai_001_001", "camera_name": "cam_00"}),
            self._sample("ai_002_002/cam_00/frame.0000", {"scene_name": "ai_002_002", "camera_name": "cam_00"}),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = create_contact_sheet(
                samples=samples,
                output_dir=Path(tmp_dir),
                dataset_name="hypersim",
                samples_per_image=1,
                sample_columns=1,
            )

            self.assertEqual(len(output_paths), 3)
            self.assertTrue((Path(tmp_dir) / "ai_001_001" / "hypersim-visualization-000.png").exists())
            self.assertTrue((Path(tmp_dir) / "ai_001_001" / "hypersim-visualization-001.png").exists())
            self.assertTrue((Path(tmp_dir) / "ai_002_002" / "hypersim-visualization-000.png").exists())

    def test_contact_sheets_fall_back_to_environment_grouping(self) -> None:
        samples = [
            self._sample(
                "AbandonedCable/omni/P0000/frame.0000",
                {
                    "environment": "AbandonedCable",
                    "version": "omni",
                    "trajectory": "P0000",
                    "camera_name": "lcam_front",
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = create_contact_sheet(
                samples=samples,
                output_dir=Path(tmp_dir),
                dataset_name="tartanground",
                samples_per_image=24,
                sample_columns=4,
            )

            self.assertEqual(len(output_paths), 1)
            self.assertEqual(output_paths[0].parent.name, "AbandonedCable__omni__P0000__lcam_front")


if __name__ == "__main__":
    unittest.main()
