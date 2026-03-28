import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
from depth_collector.core.pipeline_types import SampleRecord
from depth_collector.io import ShardWriter


class ShardWriterTest(unittest.TestCase):
    def _sample(self, sample_id: str) -> SampleRecord:
        return SampleRecord(
            sample_id=sample_id,
            image=np.zeros((2, 2, 3), dtype=np.float32),
            distance=np.ones((2, 2, 1), dtype=np.float32),
            ray_dir=np.dstack(
                [
                    np.zeros((2, 2), dtype=np.float32),
                    np.zeros((2, 2), dtype=np.float32),
                    np.ones((2, 2), dtype=np.float32),
                ]
            ),
            provenance={"sample_id": sample_id},
        )

    def test_writer_uses_final_tar_names_without_leaving_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            writer = ShardWriter(output_dir=output_dir, target_shard_size_bytes=10_000_000)
            summaries = writer.write([self._sample("demo/sample-0")])

            self.assertEqual(len(summaries), 1)
            self.assertTrue((output_dir / "shard-000000.tar").exists())
            self.assertFalse((output_dir / "shard-000000.tar.partial").exists())

    def test_writer_cleans_stale_partial_shards_before_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            stale_partial = output_dir / "shard-000000.tar.partial"
            stale_partial.write_bytes(b"incomplete")

            writer = ShardWriter(output_dir=output_dir, target_shard_size_bytes=10_000_000)
            summaries = writer.write([self._sample("demo/sample-0")])

            self.assertEqual(len(summaries), 1)
            self.assertTrue((output_dir / "shard-000000.tar").exists())
            self.assertFalse(stale_partial.exists())


if __name__ == "__main__":
    unittest.main()
