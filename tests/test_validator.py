import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from depth_collector.core import SampleRecord
from depth_collector.validation import CanonicalSampleValidator


class ValidatorTest(unittest.TestCase):
    def test_valid_sample_passes(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.linspace(0.2, 1.8, 20, dtype=np.float32).reshape(4, 5, 1)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("ok", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(max_dist=10.0, min_metric_distance_std_m=0.05).validate(sample)
        self.assertTrue(report.valid)

    def test_invalid_distance_fails(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.full((4, 5, 1), 20.0, dtype=np.float32)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("bad", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(max_dist=10.0).validate(sample)
        self.assertFalse(report.valid)

    def test_relative_distance_above_one_fails(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.full((4, 5, 1), 1.01, dtype=np.float32)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("bad-relative-range", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(max_dist=1.0, is_metric_scale=False).validate(sample)
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.code == "relative_distance_range" for issue in report.issues))

    def test_normalized_far_fraction_above_threshold_fails(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.full((4, 5, 1), 9.5, dtype=np.float32)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("bad-far-fraction", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(
            max_dist=10.0,
            max_relative_far_distance_fraction=0.9,
        ).validate(sample)
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.code == "relative_far_distance_fraction" for issue in report.issues))

    def test_metric_distance_std_below_threshold_fails(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.full((4, 5, 1), 1.0, dtype=np.float32)
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("bad-metric-std", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(
            max_dist=10.0,
            min_metric_distance_std_m=0.1,
        ).validate(sample)
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.code == "metric_distance_std_low" for issue in report.issues))

    def test_relative_distance_std_above_threshold_fails(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        distance = np.array(
            [[[0.0], [0.0], [0.0], [1.0], [1.0]],
             [[0.0], [0.0], [0.0], [1.0], [1.0]],
             [[0.0], [0.0], [0.0], [1.0], [1.0]],
             [[0.0], [0.0], [0.0], [1.0], [1.0]]],
            dtype=np.float32,
        )
        ray_dir = np.zeros((4, 5, 3), dtype=np.float32)
        ray_dir[..., 2] = 1.0
        sample = SampleRecord("bad-relative-std", image=image, distance=distance, ray_dir=ray_dir)

        report = CanonicalSampleValidator(
            max_dist=1.0,
            is_metric_scale=False,
            max_relative_distance_std=0.3,
        ).validate(sample)
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.code == "relative_distance_std_high" for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
