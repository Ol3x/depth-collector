from __future__ import annotations

import io
from pathlib import Path
import tarfile
import sys
import re

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm.auto import tqdm

from depth_collector.core.pipeline_types import SampleRecord


def load_processed_samples(processed_files_dir: Path, max_samples: int | None) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    shard_paths = sorted(processed_files_dir.glob("*.tar"))
    shard_iterator = _progress(shard_paths, desc="visualize load", unit="shard")
    for shard_path in shard_iterator:
        with tarfile.open(shard_path, "r") as archive:
            grouped_members: dict[str, dict[str, tarfile.TarInfo]] = {}
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                stem, field_name = _split_member_name(member.name)
                grouped_members.setdefault(stem, {})[field_name] = member

            for stem in sorted(grouped_members):
                members = grouped_members[stem]
                if not {"image.pt", "distance.pt", "ray_dir.pt", "meta.json"} <= set(members):
                    continue
                image_tensor = _load_torch_member(archive, members["image.pt"])
                distance_tensor = _load_torch_member(archive, members["distance.pt"])
                ray_dir_tensor = _load_torch_member(archive, members["ray_dir.pt"])
                meta_payload = _load_json_member(archive, members["meta.json"])
                samples.append(
                    SampleRecord(
                        sample_id=str(meta_payload["sample_id"]),
                        image=np.asarray(image_tensor, dtype=np.float32),
                        distance=np.asarray(distance_tensor, dtype=np.float32),
                        ray_dir=np.asarray(ray_dir_tensor, dtype=np.float32),
                        provenance=dict(meta_payload.get("provenance", {})),
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
    return samples


def create_contact_sheet(
    samples: list[SampleRecord],
    output_dir: Path,
    dataset_name: str,
    samples_per_image: int,
    sample_columns: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    grouped_samples = _group_samples_for_visualization(samples)
    for group_name, group_samples in grouped_samples.items():
        group_output_dir = output_dir / _sanitize_path_component(group_name)
        group_output_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(group_samples), samples_per_image):
            chunk = group_samples[start : start + samples_per_image]
            if not chunk:
                continue
            panel_iterator = _progress(chunk, desc=f"{dataset_name} render", unit="sample")
            panels = [_build_sample_panel(sample) for sample in panel_iterator]
            columns = max(1, min(sample_columns, len(panels)))
            rows = int(np.ceil(len(panels) / columns))
            panel_width = max(panel.width for panel in panels)
            panel_height = max(panel.height for panel in panels)
            gap_x = 16
            gap_y = 16
            width = columns * panel_width + gap_x * max(0, columns - 1)
            total_height = rows * panel_height + gap_y * max(0, rows - 1)
            canvas = Image.new("RGB", (width, total_height), color=(245, 245, 245))
            for index, panel in enumerate(panels):
                row = index // columns
                col = index % columns
                offset_x = col * (panel_width + gap_x)
                offset_y = row * (panel_height + gap_y)
                canvas.paste(panel, (offset_x, offset_y))
            output_path = group_output_dir / f"{dataset_name}-visualization-{start // samples_per_image:03d}.png"
            canvas.save(output_path)
            output_paths.append(output_path)
    return output_paths


def _group_samples_for_visualization(samples: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    groups: dict[str, list[SampleRecord]] = {}
    for sample in samples:
        group_name = _sample_visualization_group(sample)
        groups.setdefault(group_name, []).append(sample)
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def _sample_visualization_group(sample: SampleRecord) -> str:
    provenance = sample.provenance

    if "scene_name" in provenance:
        return str(provenance["scene_name"])

    if "environment" in provenance:
        parts = [str(provenance["environment"])]
        for key in ("difficulty", "version", "trajectory", "camera_name"):
            value = provenance.get(key)
            if value not in (None, ""):
                parts.append(str(value))
        return "__".join(parts)

    sample_parts = [part for part in sample.sample_id.split("/") if part]
    if len(sample_parts) >= 2:
        return "__".join(sample_parts[:-1])
    return "ungrouped"


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "ungrouped"


def _build_sample_panel(sample: SampleRecord) -> Image.Image:
    rgb = _to_uint8_image(sample.image)
    reprojection = _render_same_camera_view(sample)
    distance = _render_distance_map(sample)
    z_depth = _render_z_depth_map(sample)
    gap = 24
    label_height = 28
    panel_width = rgb.width + reprojection.width + distance.width + z_depth.width + 3 * gap
    panel_height = label_height + max(rgb.height, reprojection.height, distance.height, z_depth.height)
    canvas = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), sample.sample_id, fill=(0, 0, 0))
    draw.text((0, 14), "rgb", fill=(0, 0, 0))
    draw.text((rgb.width + gap, 14), "reprojection", fill=(0, 0, 0))
    draw.text((rgb.width + reprojection.width + 2 * gap, 14), "distance", fill=(0, 0, 0))
    draw.text((rgb.width + reprojection.width + distance.width + 3 * gap, 14), "z-depth", fill=(0, 0, 0))
    canvas.paste(rgb, (0, label_height))
    canvas.paste(reprojection, (rgb.width + gap, label_height))
    canvas.paste(distance, (rgb.width + reprojection.width + 2 * gap, label_height))
    canvas.paste(z_depth, (rgb.width + reprojection.width + distance.width + 3 * gap, label_height))
    return canvas


def _render_same_camera_view(sample: SampleRecord) -> Image.Image:
    image = np.asarray(sample.image, dtype=np.float32)
    distance = np.asarray(sample.distance[..., 0], dtype=np.float32)
    ray_dir = np.asarray(sample.ray_dir, dtype=np.float32)
    height, width = distance.shape
    fx, fy, cx, cy = _infer_pinhole_intrinsics(ray_dir)

    points = ray_dir * distance[..., None]
    x_left = points[..., 0].reshape(-1)
    y_down = points[..., 1].reshape(-1)
    z_forward = points[..., 2].reshape(-1)
    colors = np.clip(image.reshape(-1, 3), 0.0, 1.0)

    valid = np.isfinite(x_left) & np.isfinite(y_down) & np.isfinite(z_forward) & (z_forward > 1e-6)
    x_left = x_left[valid]
    y_down = y_down[valid]
    z_forward = z_forward[valid]
    colors = colors[valid]

    u = np.rint(cx - fx * (x_left / z_forward)).astype(np.int32)
    v = np.rint(cy + fy * (y_down / z_forward)).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    z_forward = z_forward[in_bounds]
    colors = colors[in_bounds]

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    order = np.argsort(z_forward)
    for idx in order:
        px = u[idx]
        py = v[idx]
        depth = z_forward[idx]
        if depth < z_buffer[py, px]:
            z_buffer[py, px] = depth
            canvas[py, px] = colors[idx]
    return _to_uint8_image(canvas)


def _render_z_depth_map(sample: SampleRecord) -> Image.Image:
    z_depth = np.asarray(sample.distance * sample.ray_dir[..., 2:3], dtype=np.float32)[..., 0]
    return _render_scalar_map(z_depth)


def _render_distance_map(sample: SampleRecord) -> Image.Image:
    distance = np.asarray(sample.distance[..., 0], dtype=np.float32)
    return _render_scalar_map(distance)


def _render_scalar_map(values: np.ndarray) -> Image.Image:
    scalar = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(scalar)
    if not np.any(valid):
        return Image.new("RGB", (scalar.shape[1], scalar.shape[0]), color=(0, 0, 0))
    near = float(np.min(scalar[valid]))
    far = float(np.max(scalar[valid]))
    if far - near < 1e-6:
        normalized = np.zeros_like(scalar, dtype=np.float32)
    else:
        normalized = np.clip((scalar - near) / (far - near), 0.0, 1.0)
    colorized = _reverse_jet_colormap(normalized)
    colorized[~valid] = 0.0
    return _to_uint8_image(colorized)


def _infer_pinhole_intrinsics(ray_dir: np.ndarray) -> tuple[float, float, float, float]:
    height, width = ray_dir.shape[:2]
    center_row = height // 2
    center_col = width // 2

    row = ray_dir[center_row]
    col = ray_dir[:, center_col]
    tx = row[:, 0] / np.maximum(row[:, 2], 1e-6)
    ty = col[:, 1] / np.maximum(col[:, 2], 1e-6)

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    tx_slope, tx_intercept = np.polyfit(xs, tx, deg=1)
    ty_slope, ty_intercept = np.polyfit(ys, ty, deg=1)

    fx = float(-1.0 / tx_slope)
    fy = float(1.0 / ty_slope)
    cx = float(tx_intercept * fx)
    cy = float(-ty_intercept * fy)
    return fx, fy, cx, cy


def _to_uint8_image(image: np.ndarray) -> Image.Image:
    array = np.clip(image, 0.0, 1.0)
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def _reverse_jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    base = np.stack(
        (
            np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0),
        ),
        axis=-1,
    )
    return base[..., ::-1]


def _split_member_name(name: str) -> tuple[str, str]:
    if name.endswith(".meta.json"):
        return name[: -len(".meta.json")], "meta.json"
    if name.endswith(".image.pt"):
        return name[: -len(".image.pt")], "image.pt"
    if name.endswith(".distance.pt"):
        return name[: -len(".distance.pt")], "distance.pt"
    if name.endswith(".ray_dir.pt"):
        return name[: -len(".ray_dir.pt")], "ray_dir.pt"
    raise ValueError(f"unsupported shard member name: {name}")


def _load_torch_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    payload = archive.extractfile(member)
    assert payload is not None
    return np.asarray(torch.load(io.BytesIO(payload.read()), weights_only=False))


def _load_json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, object]:
    payload = archive.extractfile(member)
    assert payload is not None
    import json

    return json.loads(payload.read())


def _progress(items: list[object], desc: str, unit: str) -> list[object] | tqdm:
    if not items or not sys.stdout.isatty():
        return items
    return tqdm(items, desc=desc, unit=unit, leave=False)
