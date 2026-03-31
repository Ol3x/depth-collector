from __future__ import annotations

import numpy as np

from depth_collector.core.pipeline_types import DatasetMetricsSummary, SampleRecord


def compute_sample_metrics(sample: SampleRecord, max_dist: float, *, is_metric_scale: bool = True) -> dict[str, float]:
    ray_norm = np.linalg.norm(sample.ray_dir, axis=-1)
    distance = sample.distance[..., 0]
    if is_metric_scale:
        normalized_distance = np.clip(distance / max(max_dist, 1e-6), 0.0, 1.0)
    else:
        normalized_distance = np.clip(distance, 0.0, 1.0)

    max_dist_fraction = float(np.mean(np.isclose(distance, max_dist)))
    near_zero_fraction = float(np.mean(distance <= 1e-6))
    relative_far_distance_fraction = float(np.mean(normalized_distance >= 0.9))

    return {
        "distance_min": float(np.min(distance)),
        "distance_max": float(np.max(distance)),
        "distance_mean": float(np.mean(distance)),
        "distance_std": float(np.std(distance)),
        "normalized_distance_mean": float(np.mean(normalized_distance)),
        "normalized_distance_std": float(np.std(normalized_distance)),
        "ray_norm_mean": float(np.mean(ray_norm)),
        "ray_norm_std": float(np.std(ray_norm)),
        "max_dist_fraction": max_dist_fraction,
        "near_zero_fraction": near_zero_fraction,
        "relative_far_distance_fraction": relative_far_distance_fraction,
    }


def summarize_metrics(metric_records: list[dict[str, float]]) -> DatasetMetricsSummary:
    if not metric_records:
        return DatasetMetricsSummary(sample_count=0)

    keys = sorted(metric_records[0])
    metric_means: dict[str, float] = {}
    metric_mins: dict[str, float] = {}
    metric_maxs: dict[str, float] = {}
    for key in keys:
        values = np.array([record[key] for record in metric_records], dtype=np.float64)
        metric_means[key] = float(np.mean(values))
        metric_mins[key] = float(np.min(values))
        metric_maxs[key] = float(np.max(values))
    return DatasetMetricsSummary(
        sample_count=len(metric_records),
        metric_means=metric_means,
        metric_mins=metric_mins,
        metric_maxs=metric_maxs,
    )
