import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from depth_collector.core import SampleRecord
from depth_collector.validation import CanonicalSampleValidator


class ValidatorTest(unittest.TestCase):
    def test_valid_sample_passes(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.ones((4, 5, 1), dtype=np.float32)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("ok", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(max_dist=10.0).validate(sample)
        self.assertTrue(report.valid)

    def test_invalid_distance_fails(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.full((4, 5, 1), 20.0, dtype=np.float32)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("bad", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(max_dist=10.0).validate(sample)
        self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()
