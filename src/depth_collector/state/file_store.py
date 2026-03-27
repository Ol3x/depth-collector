from __future__ import annotations

import json
from pathlib import Path

from depth_collector.core.pipeline_types import ErrorRecord
from .store import DownloadStateStore, ErrorStore, ExtractionStateStore, ProcessingStateStore


class _JsonlIdStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        for line in self.path.read_text().splitlines():
            if not line:
                continue
            ids.add(json.loads(line)["id"])
        return ids

    def is_complete(self, unit_id: str) -> bool:
        return unit_id in self._read_ids()

    def mark_complete(self, unit_id: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": unit_id}, sort_keys=True) + "\n")


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
