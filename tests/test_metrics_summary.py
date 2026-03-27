import unittest

from tests import _bootstrap  # noqa: F401
from depth_collector.validation import summarize_metrics


class MetricsSummaryTest(unittest.TestCase):
    def test_summarize_metrics(self) -> None:
        summary = summarize_metrics(
            [
                {"distance_mean": 1.0, "max_dist_fraction": 0.0},
                {"distance_mean": 3.0, "max_dist_fraction": 0.5},
            ]
        )
        self.assertEqual(summary.sample_count, 2)
        self.assertAlmostEqual(summary.metric_means["distance_mean"], 2.0)
        self.assertAlmostEqual(summary.metric_maxs["max_dist_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
