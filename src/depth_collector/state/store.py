from __future__ import annotations

from abc import ABC, abstractmethod

from depth_collector.core.pipeline_types import ErrorRecord


class DownloadStateStore(ABC):
    @abstractmethod
    def is_complete(self, unit_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mark_complete(self, unit_id: str) -> None:
        raise NotImplementedError


class ExtractionStateStore(ABC):
    @abstractmethod
    def is_complete(self, unit_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mark_complete(self, unit_id: str) -> None:
        raise NotImplementedError


class ProcessingStateStore(ABC):
    @abstractmethod
    def is_complete(self, item_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mark_complete(self, item_id: str) -> None:
        raise NotImplementedError


class ErrorStore(ABC):
    @abstractmethod
    def record(self, error: ErrorRecord) -> None:
        raise NotImplementedError
