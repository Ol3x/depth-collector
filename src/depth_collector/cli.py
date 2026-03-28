from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

from tqdm.auto import tqdm

from depth_collector.app import list_project_configs, load_enabled_pipelines, resolve_project_config_path
from depth_collector.config import load_config
from depth_collector.visualization import create_contact_sheet, load_processed_samples


def _count_files(root: Path | None) -> int:
    if root is None or not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if getattr(args, "config", None):
        return Path(args.config)
    project = getattr(args, "project", None) or "default"
    return resolve_project_config_path(project)


def _count_already_processed(pipeline: object, selected_items: list[object]) -> int:
    if not selected_items:
        return 0
    iterator = selected_items
    progress = None
    if getattr(pipeline, "verbose", False) and sys.stdout.isatty():
        progress = tqdm(
            selected_items,
            desc=f"{pipeline.dataset_name} state",
            unit="sample",
            leave=False,
        )
        iterator = progress
    try:
        return sum(1 for item in iterator if pipeline.processing_state.is_complete(pipeline.get_source_item_id(item)))
    finally:
        if progress is not None:
            progress.close()


def _effective_download_workers(pipeline: object) -> int:
    dataset_override = pipeline.dataset_config.options.get("download_workers")
    if dataset_override is None:
        return max(1, int(pipeline.config.runtime.download_workers))
    return max(1, int(dataset_override))


def _has_materialized_processed_outputs(pipeline: object) -> bool:
    return pipeline.paths.metadata.exists() and any(pipeline.paths.processed_files.glob("*.tar"))


def _reconcile_processing_state(pipeline: object) -> None:
    if _has_materialized_processed_outputs(pipeline):
        return
    if _count_lines(pipeline.paths.state / "processed.jsonl") == 0:
        return
    pipeline.processing_state.clear()
    print(f"[{pipeline.dataset_name}] cleared stale processed-state because processed outputs were missing")


def _retry_empty_source_selection_if_needed(pipeline: object, selected_items: list[object]) -> list[object]:
    if selected_items:
        return selected_items
    extracted_files = 0
    for unit in pipeline.enumerate_extraction_units():
        extracted_files += _count_files(pipeline.get_extracted_artifact_root(unit))
    if extracted_files == 0:
        return selected_items
    clear_manifest = getattr(pipeline, "clear_enumeration_manifest_cache", None)
    if clear_manifest is None:
        return selected_items
    clear_manifest()
    pipeline.reset_source_selection_cache()
    print(
        f"[{pipeline.dataset_name}] retrying source enumeration because extracted files exist "
        f"but the first pass returned zero items"
    )
    return pipeline.get_selected_source_items()


def cmd_download(args: argparse.Namespace) -> int:
    for pipeline in load_enabled_pipelines(str(_resolve_config_path(args))):
        pipeline.prepare_directories()
        selected_units = pipeline.get_selected_download_units()
        download_workers = _effective_download_workers(pipeline)
        print(f"[{pipeline.dataset_name}] download units selected: {len(selected_units)}")
        print(f"[{pipeline.dataset_name}] download workers: {min(download_workers, max(1, len(selected_units)))}")
        skipped = 0
        downloaded = 0
        failed = 0
        pending_units: list[tuple[str, object]] = []
        for unit in selected_units:
            unit_id = pipeline.get_download_unit_id(unit)
            if pipeline.is_download_unit_satisfied(unit):
                if not pipeline.download_state.is_complete(unit_id):
                    pipeline.download_state.mark_complete(unit_id)
                print(f"[{pipeline.dataset_name}] skip download {unit_id} (already present)")
                skipped += 1
                continue
            pending_units.append((unit_id, unit))

        if pending_units:
            with ThreadPoolExecutor(max_workers=min(download_workers, len(pending_units))) as executor:
                future_to_unit: dict[Future[None], tuple[str, object]] = {
                    executor.submit(pipeline.download_unit, unit): (unit_id, unit)
                    for unit_id, unit in pending_units
                }
                while future_to_unit:
                    done, _ = wait(set(future_to_unit), return_when=FIRST_COMPLETED)
                    for future in done:
                        unit_id, _ = future_to_unit.pop(future)
                        try:
                            future.result()
                            pipeline.download_state.mark_complete(unit_id)
                            print(f"[{pipeline.dataset_name}] downloaded {unit_id}")
                            downloaded += 1
                        except Exception as exc:
                            pipeline.handle_stage_exception(stage="download", item_id=unit_id, exc=exc)
                            print(f"[{pipeline.dataset_name}] download failed {unit_id}: {exc}")
                            failed += 1
        print(
            f"[{pipeline.dataset_name}] download summary: downloaded={downloaded} skipped={skipped} failed={failed}"
        )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    remove_archives = not getattr(args, "keep_archives", False)
    for pipeline in load_enabled_pipelines(str(_resolve_config_path(args))):
        pipeline.prepare_directories()
        extraction_units = list(pipeline.enumerate_extraction_units())
        print(f"[{pipeline.dataset_name}] extraction units selected: {len(extraction_units)}")
        skipped = 0
        extracted = 0
        failed = 0
        for unit in extraction_units:
            unit_id = pipeline.get_extraction_unit_id(unit)
            if pipeline.is_extraction_unit_satisfied(unit):
                if not pipeline.extraction_state.is_complete(unit_id):
                    pipeline.extraction_state.mark_complete(unit_id)
                extracted_root = pipeline.get_extracted_artifact_root(unit)
                extracted_count = _count_files(extracted_root)
                print(f"[{pipeline.dataset_name}] skip extract {unit_id} (already extracted, files={extracted_count})")
                skipped += 1
                continue
            try:
                archive_path = pipeline.get_download_artifact_path(unit)
                archive_note = ""
                if archive_path is not None and archive_path.exists():
                    archive_note = f" from {archive_path} ({_format_bytes(archive_path.stat().st_size)})"
                print(f"[{pipeline.dataset_name}] extracting {unit_id}{archive_note}")
                pipeline.extract_unit(unit)
                if remove_archives:
                    pipeline.remove_download_artifact(unit)
                pipeline.extraction_state.mark_complete(unit_id)
                extracted_root = pipeline.get_extracted_artifact_root(unit)
                extracted_count = _count_files(extracted_root)
                action = "extracted+removed" if remove_archives else "extracted"
                print(f"[{pipeline.dataset_name}] {action} {unit_id}, files={extracted_count}")
                extracted += 1
            except Exception as exc:
                pipeline.handle_stage_exception(stage="extraction", item_id=unit_id, exc=exc)
                print(f"[{pipeline.dataset_name}] extraction failed {unit_id}: {exc}")
                failed += 1
        print(
            f"[{pipeline.dataset_name}] extraction summary: extracted={extracted} skipped={skipped} failed={failed}"
        )
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    for pipeline in load_enabled_pipelines(str(_resolve_config_path(args))):
        pipeline.verbose = bool(getattr(args, "verbose", False))
        pipeline.prepare_directories()
        _reconcile_processing_state(pipeline)
        print(f"[{pipeline.dataset_name}] process stage starting")
        print(f"[{pipeline.dataset_name}] scanning extracted source items under {pipeline.paths.raw}")
        selected_items = pipeline.get_selected_source_items()
        selected_items = _retry_empty_source_selection_if_needed(pipeline, selected_items)
        already_complete = _count_already_processed(pipeline, selected_items)
        pending_items = len(selected_items) - already_complete
        print(
            f"[{pipeline.dataset_name}] source items available={pipeline._run_stats['available_source_item_count']} "
            f"selected={len(selected_items)} "
            f"skipped_by_process_ratio={pipeline._run_stats['skipped_by_process_ratio_count']} "
            f"already_processed={already_complete} pending={pending_items}"
        )
        if pending_items == 0 and _has_materialized_processed_outputs(pipeline):
            print(
                f"[{pipeline.dataset_name}] process summary: valid=0 invalid=0 processing_errors=0 "
                f"shards={len(list(pipeline.paths.processed_files.glob('*.tar')))}"
            )
            if not selected_items:
                print(f"[{pipeline.dataset_name}] no extracted source items matched the current config")
            else:
                print(f"[{pipeline.dataset_name}] all selected source items were already processed")
            if pipeline.verbose:
                print(f"[{pipeline.dataset_name}] metadata: {pipeline.paths.metadata}")
                print(f"[{pipeline.dataset_name}] metrics: {pipeline.paths.processed / 'metrics_summary.json'}")
                print(f"[{pipeline.dataset_name}] run report: {pipeline.paths.run_report}")
            continue
        pipeline.write_samples(pipeline.iter_valid_samples())
        pipeline.build_metrics_summary()
        pipeline.build_metadata()
        pipeline.build_run_report()
        pipeline.validate_output()
        print(
            f"[{pipeline.dataset_name}] process summary: valid={pipeline._run_stats['valid_sample_count']} "
            f"invalid={pipeline._run_stats['invalid_sample_count']} "
            f"processing_errors={pipeline._run_stats['processing_error_count']} "
            f"shards={len(getattr(pipeline, '_written_shards', []))}"
        )
        if not selected_items:
            print(f"[{pipeline.dataset_name}] no extracted source items matched the current config")
        elif pending_items == 0:
            print(f"[{pipeline.dataset_name}] all selected source items were already processed")
        written_shards = getattr(pipeline, "_written_shards", [])
        if pipeline.verbose or written_shards:
            print(f"[{pipeline.dataset_name}] metadata: {pipeline.paths.metadata}")
            print(f"[{pipeline.dataset_name}] metrics: {pipeline.paths.processed / 'metrics_summary.json'}")
            print(f"[{pipeline.dataset_name}] run report: {pipeline.paths.run_report}")
        if written_shards:
            shard_names = ", ".join(str(shard["shard_name"]) for shard in written_shards)
            print(f"[{pipeline.dataset_name}] written shards: {shard_names}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args)
    config = load_config(config_path)
    print(f"project: {config.project.name}")
    print(f"config: {config_path}")
    print(f"data_root: {Path(config.output.root_data_dir) / config.project.name}")
    for pipeline in load_enabled_pipelines(str(config_path)):
        archives = list(pipeline.iter_download_artifact_paths())
        archive_count = sum(1 for path in archives if path.exists())
        extracted_files = 0
        for unit in pipeline.enumerate_extraction_units():
            extracted_files += _count_files(pipeline.get_extracted_artifact_root(unit))
        shard_paths = sorted(pipeline.paths.processed_files.glob("*.tar"))
        print(
            f"[{pipeline.dataset_name}] archives_present={archive_count}/{len(archives)} "
            f"extracted_files={extracted_files} shards={len(shard_paths)} "
            f"downloads_state={_count_lines(pipeline.paths.state / 'downloads.jsonl')} "
            f"extractions_state={_count_lines(pipeline.paths.state / 'extractions.jsonl')} "
            f"processed_state={_count_lines(pipeline.paths.state / 'processed.jsonl')} "
            f"errors={_count_lines(pipeline.paths.state / 'errors.jsonl')}"
        )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args)
    config = load_config(config_path)
    project_root = Path(config.output.root_data_dir) / config.project.name
    if not args.yes:
        raise SystemExit(f"refusing to remove {project_root} without --yes")
    if project_root.exists():
        shutil.rmtree(project_root)
        print(f"removed {project_root}")
    else:
        print(f"nothing to remove at {project_root}")
    return 0


def cmd_clean_process(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit("refusing to remove process artifacts without --yes")
    for pipeline in load_enabled_pipelines(str(_resolve_config_path(args))):
        removed_any = False
        if pipeline.paths.processed.exists():
            shutil.rmtree(pipeline.paths.processed)
            removed_any = True
        visualizations_dir = pipeline.paths.root / "visualizations"
        if visualizations_dir.exists():
            shutil.rmtree(visualizations_dir)
            removed_any = True
        if (pipeline.paths.state / "processed.jsonl").exists():
            pipeline.processing_state.clear()
            removed_any = True
        enumeration_manifest_path = pipeline.paths.state / "enumeration_manifest.json"
        if enumeration_manifest_path.exists():
            enumeration_manifest_path.unlink()
            clear_manifest = getattr(pipeline, "clear_enumeration_manifest_cache", None)
            if clear_manifest is not None:
                clear_manifest()
            removed_any = True
        removed_errors = pipeline.error_store.remove_stages({"enumeration", "processing"})
        if removed_errors:
            removed_any = True
        if removed_any:
            print(
                f"[{pipeline.dataset_name}] removed process artifacts "
                f"(processed outputs, visualizations, processed state, enumeration manifest, "
                f"stage errors={removed_errors})"
            )
        else:
            print(f"[{pipeline.dataset_name}] no process artifacts to remove")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    if args.all and args.max_samples is not None:
        raise SystemExit("use either --all or --max-samples, not both")
    max_samples = None if args.all else int(args.max_samples or 24)
    samples_per_image = int(args.samples_per_image)
    sample_columns = int(args.sample_columns)
    for pipeline in load_enabled_pipelines(str(_resolve_config_path(args))):
        output_dir = pipeline.paths.root / "visualizations"
        samples = load_processed_samples(pipeline.paths.processed_files, max_samples=max_samples)
        if not samples:
            print(f"[{pipeline.dataset_name}] no processed samples available for visualization")
            continue
        output_paths = create_contact_sheet(
            samples=samples,
            output_dir=output_dir,
            dataset_name=pipeline.dataset_name,
            samples_per_image=samples_per_image,
            sample_columns=sample_columns,
        )
        print(
            f"[{pipeline.dataset_name}] visualization summary: samples={len(samples)} images={len(output_paths)} "
            f"output_dir={output_dir}"
        )
        for path in output_paths:
            print(f"[{pipeline.dataset_name}] visualization: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth Collector project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download dataset archives for a project.")
    download_parser.add_argument("project", nargs="?", default="default", help="Project name from configs/<name>.json.")
    download_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    download_parser.set_defaults(func=cmd_download)

    extract_parser = subparsers.add_parser("extract", help="Extract downloaded archives.")
    extract_parser.add_argument("project", nargs="?", default="default", help="Project name from configs/<name>.json.")
    extract_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    extract_parser.add_argument("--keep-archives", action="store_true", help="Keep archives after extraction.")
    extract_parser.set_defaults(func=cmd_extract)

    process_parser = subparsers.add_parser("process", help="Process extracted data into shards.")
    process_parser.add_argument("project", nargs="?", default="default", help="Project name from configs/<name>.json.")
    process_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    process_parser.add_argument("--verbose", action="store_true", help="Show detailed preflight progress.")
    process_parser.set_defaults(func=cmd_process)

    status_parser = subparsers.add_parser("status", help="Show project status.")
    status_parser.add_argument("project", nargs="?", default="default", help="Project name from configs/<name>.json.")
    status_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    status_parser.set_defaults(func=cmd_status)

    visualize_parser = subparsers.add_parser("visualize", help="Render processed sample diagnostics.")
    visualize_parser.add_argument(
        "project", nargs="?", default="default", help="Project name from configs/<name>.json."
    )
    visualize_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    visualize_parser.add_argument("--max-samples", type=int, help="Maximum samples to visualize per dataset.")
    visualize_parser.add_argument("--all", action="store_true", help="Visualize all processed samples.")
    visualize_parser.add_argument(
        "--samples-per-image", type=int, default=24, help="How many sample panels to pack into each output image."
    )
    visualize_parser.add_argument("--sample-columns", type=int, default=4, help="How many sample panels per row.")
    visualize_parser.set_defaults(func=cmd_visualize)

    clean_parser = subparsers.add_parser("clean", help="Remove a project's data directory.")
    clean_parser.add_argument("project", nargs="?", default="default", help="Project name from configs/<name>.json.")
    clean_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    clean_parser.add_argument("--yes", action="store_true", help="Confirm deletion.")
    clean_parser.set_defaults(func=cmd_clean)

    clean_process_parser = subparsers.add_parser(
        "clean_process",
        help="Remove processed outputs and process-stage state while keeping raw extracted data.",
    )
    clean_process_parser.add_argument(
        "project", nargs="?", default="default", help="Project name from configs/<name>.json."
    )
    clean_process_parser.add_argument("--config", help="Explicit config JSON path. Overrides project name.")
    clean_process_parser.add_argument("--yes", action="store_true", help="Confirm deletion.")
    clean_process_parser.set_defaults(func=cmd_clean_process)

    projects_parser = subparsers.add_parser("projects", help="List known project configs.")
    projects_parser.set_defaults(func=cmd_projects)
    return parser


def cmd_projects(args: argparse.Namespace) -> int:
    del args
    for project_name, path in list_project_configs():
        print(f"{project_name}: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))
