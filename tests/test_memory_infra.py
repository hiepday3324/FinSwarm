import datetime as dt
import os
import shutil
import unittest

from puppy.memory_ext.faiss_store import FaissVectorStore
from puppy.memory_ext.sqlite_memory_store import SQLiteMemoryStore
from puppy.memory_ext.vector_store import MemoryVectorService


class MemoryInfraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sqlite_store = SQLiteMemoryStore(db_path=":memory:")
        self.sqlite_store.connect()
        self.vector_store = FaissVectorStore(dim=8, metric="cosine")
        self.service = MemoryVectorService(self.sqlite_store, self.vector_store)

    def test_sqlite_crud_and_ordered_vector_lookup(self) -> None:
        for index, memory_id in enumerate(["mem_1", "mem_2"]):
            self.sqlite_store.add_memory(
                memory_id=memory_id,
                vector_id=index,
                agent_id="sector_tsla",
                symbol="TSLA",
                layer="short",
                text=f"text {index}",
                event_date="2026-01-02",
                created_at="2026-01-02T10:00:00",
                available_at="2026-01-02T10:05:00",
                source="test",
                importance=0.8,
                metadata_dict={"tag": "earnings"},
            )

        rec = self.sqlite_store.get_memory("mem_1")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["layer"], "short")
        recs = self.sqlite_store.get_by_vector_ids([1, 0])
        self.assertEqual([item["memory_id"] for item in recs], ["mem_2", "mem_1"])

    def test_vector_add_search_passes_with_or_without_faiss(self) -> None:
        ids = self.vector_store.add_texts(["hello world", "goodbye world"])
        self.assertEqual(ids, [0, 1])
        results = self.vector_store.search("hello", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("vector_id", results[0])

    def test_memory_leakage_prevention(self) -> None:
        t = dt.datetime(2026, 1, 2, 12, 0, 0)
        self.service.add_memory(
            memory_id="mem_safe",
            agent_id="sector_tsla",
            symbol="TSLA",
            layer="short",
            text="TSLA chip shortage is recovering.",
            event_date=dt.date(2026, 1, 2),
            created_at=dt.datetime(2026, 1, 2, 11, 0, 0),
            available_at=dt.datetime(2026, 1, 2, 11, 5, 0),
            source="test",
            importance=0.5,
        )
        self.service.add_memory(
            memory_id="mem_future",
            agent_id="sector_tsla",
            symbol="TSLA",
            layer="short",
            text="TSLA future announcement details.",
            event_date=dt.date(2026, 1, 2),
            created_at=dt.datetime(2026, 1, 2, 13, 0, 0),
            available_at=dt.datetime(2026, 1, 2, 13, 5, 0),
            source="test",
            importance=0.9,
        )

        results = self.service.search("TSLA announcement", top_k=5, as_of=t)
        memory_ids = [res["memory_id"] for res in results]
        self.assertIn("mem_safe", memory_ids)
        self.assertNotIn("mem_future", memory_ids)

    def test_save_load_skips_if_backend_does_not_support_persistence(self) -> None:
        if not self.vector_store.supports_persistence:
            self.skipTest("numpy fallback vector store does not support save/load")

        self.vector_store.add_texts(["hello world", "goodbye world"])
        test_dir = "data/test_faiss"
        test_path = os.path.join(test_dir, "test.index")
        try:
            self.vector_store.save(test_path)
            self.assertTrue(os.path.exists(test_path))
            new_store = FaissVectorStore(dim=8, metric="cosine")
            new_store.load(test_path)
            self.assertEqual(new_store.vector_id_counter, 2)
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
