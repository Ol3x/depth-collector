from __future__ import annotations

from pathlib import Path
from typing import Iterable

from depth_collector.core.pipeline_types import SampleRecord


class ShardWriter:
    """Placeholder shared shard writer surface."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def write(self, _samples: Iterable[SampleRecord]) -> None:
        raise NotImplementedError("WebDataset shard writing is not implemented yet.")
