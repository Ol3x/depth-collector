from __future__ import annotations

import io
from pathlib import Path
import tarfile
import sys
import re

from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    absolute_scale_max: float,
) -> list[Path]:
    del sample_columns
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
            row_images = [_build_sample_row(sample, absolute_scale_max=absolute_scale_max) for sample in panel_iterator]
            output_path = group_output_dir / f"{dataset_name}-visualization-{start // samples_per_image:03d}.png"
            _save_visualization_table(
                sample_rows=row_images,
                sample_ids=[sample.sample_id for sample in chunk],
                output_path=output_path,
            )
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


def _build_sample_row(sample: SampleRecord, *, absolute_scale_max: float) -> list[Image.Image]:
    rgb = _to_uint8_image(sample.image)
    reprojection = _render_same_camera_view(sample)
    relative_distance = _render_distance_map(sample, absolute_scale_max=absolute_scale_max, relative=True)
    absolute_distance = _render_distance_map(sample, absolute_scale_max=absolute_scale_max, relative=False)
    relative_z_depth = _render_z_depth_map(sample, absolute_scale_max=absolute_scale_max, relative=True)
    absolute_z_depth = _render_z_depth_map(sample, absolute_scale_max=absolute_scale_max, relative=False)
    histogram = _render_histogram_panel(sample, absolute_scale_max=absolute_scale_max)
    return [
        rgb,
        reprojection,
        relative_distance,
        absolute_distance,
        relative_z_depth,
        absolute_z_depth,
        histogram,
    ]


def _save_visualization_table(sample_rows: list[list[Image.Image]], sample_ids: list[str], output_path: Path) -> None:
    if not sample_rows:
        Image.new("RGB", (1, 1), color=(255, 255, 255)).save(output_path)
        return

    column_labels = [
        "sample",
        "rgb",
        "reprojection",
        "rel distance",
        "abs distance",
        "rel depth",
        "abs depth",
        "histogram",
    ]
    row_count = len(sample_rows) + 1
    col_count = len(column_labels)
    first_row = sample_rows[0]
    row_height = max(tile.height for tile in first_row)
    tile_widths = [max(tile.width for tile in column_tiles) for column_tiles in zip(*sample_rows)]
    id_width = max(row_height * 2, max(len(sample_id) for sample_id in sample_ids) * max(14, row_height // 10))
    column_widths = [id_width, *tile_widths]
    width_ratios = [max(1.0, float(width)) for width in column_widths]
    height_ratios = [max(48.0, row_height * 0.24)] + [float(row_height) for _ in sample_rows]

    fig_width = max(18.0, sum(width_ratios) / 110.0)
    fig_height = max(4.0, sum(height_ratios) / 90.0)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
    )
    if row_count == 1:
        axes = np.asarray([axes])
    if col_count == 1:
        axes = axes[:, None]

    fig.patch.set_facecolor((0.96, 0.96, 0.96))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.10)

    for col_index, label in enumerate(column_labels):
        ax = axes[0, col_index]
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.02,
            0.5,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=max(14, int(row_height / 10)),
            fontweight="bold",
        )

    for row_index, (sample_id, row_tiles) in enumerate(zip(sample_ids, sample_rows), start=1):
        id_ax = axes[row_index, 0]
        id_ax.set_facecolor("white")
        id_ax.set_xticks([])
        id_ax.set_yticks([])
        for spine in id_ax.spines.values():
            spine.set_visible(False)
        id_ax.text(
            0.02,
            0.5,
            sample_id,
            transform=id_ax.transAxes,
            ha="left",
            va="center",
            fontsize=max(12, int(row_height / 11)),
            wrap=True,
        )
        for col_index, tile in enumerate(row_tiles, start=1):
            ax = axes[row_index, col_index]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.imshow(np.asarray(tile))

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _render_same_camera_view(sample: SampleRecord) -> Image.Image:
    projection = str(sample.provenance.get("projection", "")).strip().lower()
    if projection == "equirectangular":
        return _render_same_equirectangular_view(sample)
    return _render_same_pinhole_view(sample)


def _render_same_pinhole_view(sample: SampleRecord) -> Image.Image:
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


def _render_same_equirectangular_view(sample: SampleRecord) -> Image.Image:
    image = np.asarray(sample.image, dtype=np.float32)
    distance = np.asarray(sample.distance[..., 0], dtype=np.float32)
    ray_dir = np.asarray(sample.ray_dir, dtype=np.float32)
    height, width = distance.shape

    left = ray_dir[..., 0].reshape(-1)
    down = ray_dir[..., 1].reshape(-1)
    forward = ray_dir[..., 2].reshape(-1)
    radial_distance = distance.reshape(-1)
    colors = np.clip(image.reshape(-1, 3), 0.0, 1.0)

    horizontal_norm = np.sqrt(np.maximum(left * left + forward * forward, 1e-12))
    lon = np.arctan2(-left, forward)
    lat = np.arctan2(-down, horizontal_norm)

    u = np.rint(((lon / (2.0 * np.pi)) + 0.5) * width - 0.5).astype(np.int32)
    v = np.rint((0.5 - (lat / np.pi)) * height - 0.5).astype(np.int32)

    # Wrap the panorama horizontally and clamp vertically.
    u = np.mod(u, width)
    v = np.clip(v, 0, height - 1)

    valid = (
        np.isfinite(left)
        & np.isfinite(down)
        & np.isfinite(forward)
        & np.isfinite(radial_distance)
        & (radial_distance > 1e-6)
    )
    u = u[valid]
    v = v[valid]
    radial_distance = radial_distance[valid]
    colors = colors[valid]

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    order = np.argsort(radial_distance)
    for idx in order:
        px = u[idx]
        py = v[idx]
        depth = radial_distance[idx]
        if depth < z_buffer[py, px]:
            z_buffer[py, px] = depth
            canvas[py, px] = colors[idx]
    return _to_uint8_image(canvas)


def _render_z_depth_map(sample: SampleRecord, *, absolute_scale_max: float, relative: bool) -> Image.Image:
    z_depth = np.asarray(sample.distance * sample.ray_dir[..., 2:3], dtype=np.float32)[..., 0]
    return _render_scalar_map(z_depth, absolute_scale_max=absolute_scale_max, relative=relative)


def _render_distance_map(sample: SampleRecord, *, absolute_scale_max: float, relative: bool) -> Image.Image:
    distance = np.asarray(sample.distance[..., 0], dtype=np.float32)
    return _render_scalar_map(distance, absolute_scale_max=absolute_scale_max, relative=relative)


def _render_histogram_panel(sample: SampleRecord, *, absolute_scale_max: float) -> Image.Image:
    height, width = sample.image.shape[:2]
    canvas = np.ones((height, width, 3), dtype=np.float32)
    panel_font = _load_font(max(18, height // 12))

    distance = np.asarray(sample.distance[..., 0], dtype=np.float32)
    distance_hist = _compute_histogram(distance, upper=absolute_scale_max)

    margin_left = max(24, width // 12)
    margin_right = max(12, width // 24)
    margin_top = max(18, height // 14)
    margin_bottom = max(18, height // 10)
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, height - margin_top - margin_bottom)
    baseline = margin_top + plot_height

    canvas[margin_top:baseline, margin_left : margin_left + plot_width] = 0.98

    # Axes.
    canvas[baseline - 1 : baseline + 1, margin_left : margin_left + plot_width] = 0.15
    canvas[margin_top:baseline, margin_left - 1 : margin_left + 1] = 0.15

    distance_color = np.array([0.850, 0.325, 0.098], dtype=np.float32)
    bar_gap = 1
    bin_count = len(distance_hist)
    bin_width = max(1, plot_width // bin_count)

    for index in range(bin_count):
        x0 = margin_left + index * bin_width
        x1 = min(margin_left + plot_width, x0 + max(1, bin_width - bar_gap))
        if x0 >= x1:
            continue
        distance_bar_height = int(distance_hist[index] * max(1, plot_height - 1))
        if distance_bar_height > 0:
            canvas[baseline - distance_bar_height : baseline, x0:x1] = distance_color

    panel = _to_uint8_image(canvas)
    draw = ImageDraw.Draw(panel)
    label_y = max(0, margin_top - max(18, height // 18))
    draw.text((margin_left, label_y), "distance", fill=(217, 83, 25), font=panel_font)
    axis_y = max(0, height - max(20, height // 14))
    draw.text((margin_left, axis_y), "0", fill=(0, 0, 0), font=panel_font)
    upper_label = f"{absolute_scale_max:.0f}" if absolute_scale_max >= 10.0 else f"{absolute_scale_max:.2f}"
    right_label_x = max(margin_left, width - margin_right - max(24, len(upper_label) * 8))
    draw.text((right_label_x, axis_y), upper_label, fill=(0, 0, 0), font=panel_font)
    return panel


def _compute_histogram(values: np.ndarray, *, upper: float, bins: int = 64) -> np.ndarray:
    scalar = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(scalar)
    if upper <= 1e-6 or not np.any(valid):
        return np.zeros(bins, dtype=np.float32)
    clipped = np.clip(scalar[valid], 0.0, upper)
    counts, _ = np.histogram(clipped, bins=bins, range=(0.0, upper))
    counts = counts.astype(np.float32)
    peak = float(np.max(counts))
    if peak <= 1e-6:
        return np.zeros(bins, dtype=np.float32)
    return counts / peak


def _render_scalar_map(values: np.ndarray, *, absolute_scale_max: float | None = None, relative: bool = True) -> Image.Image:
    scalar = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(scalar)
    if not np.any(valid):
        return Image.new("RGB", (scalar.shape[1], scalar.shape[0]), color=(0, 0, 0))
    valid_values = scalar[valid]
    if relative:
        near = float(np.percentile(valid_values, 1.0))
        far = float(np.percentile(valid_values, 99.0))
        absolute_near = float(np.min(valid_values))
        absolute_far = float(np.max(valid_values))
        if far - near < 1e-6:
            near = absolute_near
            far = absolute_far
        if far - near < 1e-6:
            normalized = np.zeros_like(scalar, dtype=np.float32)
        else:
            normalized = np.clip((scalar - near) / (far - near), 0.0, 1.0)
    else:
        scale_max = float(absolute_scale_max or 1.0)
        if scale_max <= 1e-6:
            normalized = np.zeros_like(scalar, dtype=np.float32)
        else:
            normalized = np.clip(scalar / scale_max, 0.0, 1.0)
    # Spectral already maps lower values toward the warm end and higher values
    # toward the cool end, which matches the shared near-red / far-blue rule.
    colorized = _spectral_colormap(normalized)
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


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidate_paths:
        font_path = Path(path)
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _measure_text_height(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    try:
        bbox = font.getbbox("Ag")
        return max(1, int(bbox[3] - bbox[1]))
    except AttributeError:
        return max(1, int(getattr(font, "size", 12)))


def _spectral_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.asarray(colormaps["Spectral"](values)[..., :3], dtype=np.float32)


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
