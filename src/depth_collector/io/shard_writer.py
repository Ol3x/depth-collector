from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import tarfile
from typing import Iterable

import torch

from depth_collector.core.pipeline_types import SampleRecord


class ShardWriter:
    """Write grouped WebDataset-style sample fields into size-bounded tar shards."""

    def __init__(
        self,
        output_dir: Path,
        target_shard_size_bytes: int,
        ensure_split_pair: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.target_shard_size_bytes = target_shard_size_bytes
        self.ensure_split_pair = ensure_split_pair

    def write(self, samples: Iterable[SampleRecord]) -> list[dict[str, object]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_partial_shards()

        shard_summaries: list[dict[str, object]] = []
        shard_index = 0
        shard_sample_count = 0
        shard_size_bytes = 0
        tar_handle: tarfile.TarFile | None = None
        shard_path: Path | None = None
        final_shard_path: Path | None = None

        def open_new_shard() -> tuple[tarfile.TarFile, Path, Path]:
            final_path = self.output_dir / f"shard-{shard_index:06d}.tar"
            partial_path = self.output_dir / f"{final_path.name}.partial"
            return tarfile.open(partial_path, "w"), partial_path, final_path

        def close_current_shard() -> None:
            nonlocal tar_handle, shard_path, final_shard_path, shard_sample_count, shard_size_bytes, shard_index
            if tar_handle is None or shard_path is None or final_shard_path is None:
                return
            tar_handle.close()
            shard_path.replace(final_shard_path)
            shard_summaries.append(
                {
                    "shard_name": final_shard_path.name,
                    "sample_count": shard_sample_count,
                    "payload_bytes": shard_size_bytes,
                }
            )
            tar_handle = None
            shard_path = None
            final_shard_path = None
            shard_sample_count = 0
            shard_size_bytes = 0
            shard_index += 1

        try:
            for sample in samples:
                payload_members = self._serialize_sample_members(sample)
                payload_size_bytes = sum(len(payload) for _, payload in payload_members)
                if tar_handle is None:
                    tar_handle, shard_path, final_shard_path = open_new_shard()
                elif shard_sample_count > 0 and shard_size_bytes + payload_size_bytes > self.target_shard_size_bytes:
                    close_current_shard()
                    tar_handle, shard_path, final_shard_path = open_new_shard()

                assert tar_handle is not None
                for member_name, payload_bytes in payload_members:
                    info = tarfile.TarInfo(name=member_name)
                    info.size = len(payload_bytes)
                    tar_handle.addfile(info, io.BytesIO(payload_bytes))
                shard_sample_count += 1
                shard_size_bytes += payload_size_bytes

            close_current_shard()
        finally:
            if tar_handle is not None:
                tar_handle.close()
            if shard_path is not None and shard_path.exists():
                shard_path.unlink()

        if self.ensure_split_pair and len(shard_summaries) == 1 and int(shard_summaries[0]["sample_count"]) > 0:
            source_name = str(shard_summaries[0]["shard_name"])
            source_path = self.output_dir / source_name
            duplicate_name = "shard-000001.tar"
            duplicate_path = self.output_dir / duplicate_name
            if not duplicate_path.exists():
                shutil.copy2(source_path, duplicate_path)
            shard_summaries.append(
                {
                    "shard_name": duplicate_name,
                    "sample_count": shard_summaries[0]["sample_count"],
                    "payload_bytes": shard_summaries[0]["payload_bytes"],
                }
            )
        return shard_summaries

    def _serialize_sample_members(self, sample: SampleRecord) -> list[tuple[str, bytes]]:
        stem = self._sample_stem(sample.sample_id)
        return [
            (f"{stem}.image.pt", self._serialize_torch_value(torch.from_numpy(sample.image))),
            (f"{stem}.distance.pt", self._serialize_torch_value(torch.from_numpy(sample.distance))),
            (f"{stem}.ray_dir.pt", self._serialize_torch_value(torch.from_numpy(sample.ray_dir))),
            (
                f"{stem}.meta.json",
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "provenance": sample.provenance,
                    },
                    sort_keys=True,
                ).encode("utf-8"),
            ),
        ]

    def _serialize_torch_value(self, value: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    def _sample_stem(self, sample_id: str) -> str:
        return sample_id.replace("/", "__")

    def _cleanup_partial_shards(self) -> None:
        for path in self.output_dir.glob("shard-*.tar.partial"):
            path.unlink()
