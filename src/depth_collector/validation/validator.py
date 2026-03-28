from __future__ import annotations

import numpy as np

from depth_collector.core.pipeline_types import SampleRecord, ValidationIssue, ValidationReport
from .metrics import compute_sample_metrics


class CanonicalSampleValidator:
    def __init__(self, max_dist: float, ray_norm_tolerance: float = 1e-3) -> None:
        self.max_dist = max_dist
        self.ray_norm_tolerance = ray_norm_tolerance

    def validate(self, sample: SampleRecord) -> ValidationReport:
        issues: list[ValidationIssue] = []

        if sample.image.ndim != 3 or sample.image.shape[-1] != 3:
            issues.append(ValidationIssue("image_shape", "image must have shape (H, W, 3)", "error"))
        if sample.distance.ndim != 3 or sample.distance.shape[-1] != 1:
            issues.append(ValidationIssue("distance_shape", "distance must have shape (H, W, 1)", "error"))
        if sample.ray_dir.ndim != 3 or sample.ray_dir.shape[-1] != 3:
            issues.append(ValidationIssue("ray_shape", "ray_dir must have shape (H, W, 3)", "error"))

        if not issues:
            h, w = sample.image.shape[:2]
            if sample.distance.shape[:2] != (h, w) or sample.ray_dir.shape[:2] != (h, w):
                issues.append(
                    ValidationIssue("spatial_mismatch", "image, distance, and ray_dir must share H and W", "error")
                )

        if not np.isfinite(sample.image).all():
            issues.append(ValidationIssue("image_finite", "image contains non-finite values", "error"))
        if sample.image.size > 0:
            if float(np.min(sample.image)) < 0.0:
                issues.append(ValidationIssue("image_range_low", "image contains values below 0", "error"))
            if float(np.max(sample.image)) > 1.0 + 1e-6:
                issues.append(ValidationIssue("image_range_high", "image contains values above 1", "error"))
        if not np.isfinite(sample.distance).all():
            issues.append(ValidationIssue("distance_finite", "distance contains non-finite values", "error"))
        if not np.isfinite(sample.ray_dir).all():
            issues.append(ValidationIssue("ray_finite", "ray_dir contains non-finite values", "error"))

        if sample.distance.size > 0:
            if float(np.min(sample.distance)) < 0.0:
                issues.append(ValidationIssue("distance_negative", "distance contains negative values", "error"))
            if float(np.max(sample.distance)) > self.max_dist + 1e-6:
                issues.append(ValidationIssue("distance_range", "distance exceeds max_dist", "error"))

        ray_norm = np.linalg.norm(sample.ray_dir, axis=-1)
        if sample.ray_dir.size > 0 and not np.allclose(ray_norm, 1.0, atol=self.ray_norm_tolerance):
            issues.append(ValidationIssue("ray_norm", "ray_dir is not normalized", "error"))

        metrics = compute_sample_metrics(sample, self.max_dist)
        return ValidationReport(valid=not issues, issues=tuple(issues), metrics=metrics)
