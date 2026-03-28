from __future__ import annotations

import json
from pathlib import Path

from depth_collector.core.pipeline_types import ErrorRecord
from .store import DownloadStateStore, ErrorStore, ExtractionStateStore, ProcessingStateStore


class _JsonlIdStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_ids: set[str] | None = None
        self._cached_signature: tuple[int, int] | None = None

    def _current_signature(self) -> tuple[int, int] | None:
        if not self.path.exists():
            return None
        stat = self.path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def _read_ids(self) -> set[str]:
        signature = self._current_signature()
        if signature is None:
            self._cached_ids = set()
            self._cached_signature = None
            return set()
        if self._cached_ids is not None and self._cached_signature == signature:
            return self._cached_ids
        ids: set[str] = set()
        for line in self.path.read_text().splitlines():
            if not line:
                continue
            ids.add(json.loads(line)["id"])
        self._cached_ids = ids
        self._cached_signature = signature
        return ids

    def is_complete(self, unit_id: str) -> bool:
        return unit_id in self._read_ids()

    def mark_complete(self, unit_id: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": unit_id}, sort_keys=True) + "\n")
        if self._cached_ids is not None:
            self._cached_ids.add(unit_id)
            self._cached_signature = self._current_signature()

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._cached_ids = set()
        self._cached_signature = None


class FileDownloadStateStore(_JsonlIdStore, DownloadStateStore):
    pass


class FileExtractionStateStore(_JsonlIdStore, ExtractionStateStore):
    pass


class FileProcessingStateStore(_JsonlIdStore, ProcessingStateStore):
    pass


class JsonlErrorStore(ErrorStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_keys: set[tuple[str, str, str, str]] | None = None
        self._cached_signature: tuple[int, int] | None = None

    def _current_signature(self) -> tuple[int, int] | None:
        if not self.path.exists():
            return None
        stat = self.path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def _read_keys(self) -> set[tuple[str, str, str, str]]:
        signature = self._current_signature()
        if signature is None:
            self._cached_keys = set()
            self._cached_signature = None
            return set()
        if self._cached_keys is not None and self._cached_signature == signature:
            return self._cached_keys
        keys: set[tuple[str, str, str, str]] = set()
        for line in self.path.read_text().splitlines():
            if not line:
                continue
            payload = json.loads(line)
            keys.add(
                (
                    str(payload["stage"]),
                    str(payload["dataset_name"]),
                    str(payload["item_id"]),
                    str(payload["error_message"]),
                )
            )
        self._cached_keys = keys
        self._cached_signature = signature
        return keys

    def has_matching_record(self, error: ErrorRecord) -> bool:
        return (
            error.stage,
            error.dataset_name,
            error.item_id,
            error.error_message,
        ) in self._read_keys()

    def record(self, error: ErrorRecord) -> None:
        payload = {
            "stage": error.stage,
            "dataset_name": error.dataset_name,
            "item_id": error.item_id,
            "error_message": error.error_message,
            "traceback_text": error.traceback_text,
            "retry_count": error.retry_count,
            "terminal": error.terminal,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._cached_keys is not None:
            self._cached_keys.add(
                (
                    error.stage,
                    error.dataset_name,
                    error.item_id,
                    error.error_message,
                )
            )
            self._cached_signature = self._current_signature()

    def remove_stages(self, stages: set[str]) -> int:
        if not self.path.exists():
            self._cached_keys = set()
            self._cached_signature = None
            return 0
        kept_lines: list[str] = []
        removed_count = 0
        for line in self.path.read_text().splitlines():
            if not line:
                continue
            payload = json.loads(line)
            if str(payload.get("stage")) in stages:
                removed_count += 1
                continue
            kept_lines.append(json.dumps(payload, sort_keys=True))
        if kept_lines:
            self.path.write_text("\n".join(kept_lines) + "\n")
        else:
            self.path.unlink()
        self._cached_keys = None
        self._cached_signature = None
        return removed_count
