import unittest

from tests import _bootstrap  # noqa: F401
from depth_collector.validation import summarize_metrics


class MetricsSummaryTest(unittest.TestCase):
    def test_summarize_metrics(self) -> None:
        summary = summarize_metrics(
            [
                {"distance_mean": 1.0, "distance_std": 0.1, "max_dist_fraction": 0.0, "relative_far_distance_fraction": 0.0},
                {"distance_mean": 3.0, "distance_std": 0.3, "max_dist_fraction": 0.5, "relative_far_distance_fraction": 0.75},
            ]
        )
        self.assertEqual(summary.sample_count, 2)
        self.assertAlmostEqual(summary.metric_means["distance_mean"], 2.0)
        self.assertAlmostEqual(summary.metric_maxs["max_dist_fraction"], 0.5)
        self.assertAlmostEqual(summary.metric_means["distance_std"], 0.2)
        self.assertAlmostEqual(summary.metric_maxs["relative_far_distance_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
