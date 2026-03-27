import json
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from depth_collector.core import ErrorRecord
from depth_collector.state import FileProcessingStateStore, JsonlErrorStore


class StateStoreTest(unittest.TestCase):
    def test_processing_store_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "processed.jsonl"
            store = FileProcessingStateStore(path)
            self.assertFalse(store.is_complete("item-1"))
            store.mark_complete("item-1")
            self.assertTrue(store.is_complete("item-1"))

    def test_error_store_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "errors.jsonl"
            store = JsonlErrorStore(path)
            store.record(ErrorRecord(stage="processing", dataset_name="demo", item_id="a", error_message="bad"))
            payload = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(payload["dataset_name"], "demo")
            self.assertEqual(payload["item_id"], "a")


if __name__ == "__main__":
    unittest.main()
