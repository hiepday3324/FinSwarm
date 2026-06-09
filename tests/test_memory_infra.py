import unittest
import datetime as dt
import os
import shutil

from puppy.memory_ext.sqlite_memory_store import SQLiteMemoryStore
from puppy.memory_ext.faiss_store import FaissVectorStore, FAISS_AVAILABLE
from puppy.memory_ext.vector_store import MemoryVectorService


class MemoryInfraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sqlite_store = SQLiteMemoryStore(db_path=":memory:")
        self.sqlite_store.connect()
        
        self.faiss_store = FaissVectorStore(dim=8, metric="cosine")
        self.service = MemoryVectorService(self.sqlite_store, self.faiss_store)

    def test_sqlite_crud(self) -> None:
        # Verify schema
        self.sqlite_store.add_memory(
            memory_id="mem_1",
            vector_id=0,
            agent_id="sector_tsla",
            symbol="TSLA",
            layer="short",
            text="TSLA delivery looks good.",
            event_date="2026-01-02",
            created_at="2026-01-02T10:00:00",
            available_at="2026-01-02T10:05:00",
            source="test",
            importance=0.8,
            metadata_dict={"tag": "earnings"}
        )

        rec = self.sqlite_store.get_memory("mem_1")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["vector_id"], 0)
        self.assertEqual(rec["layer"], "short")
        self.assertEqual(rec["importance"], 0.8)

        # get_by_vector_ids
        recs = self.sqlite_store.get_by_vector_ids([0])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["memory_id"], "mem_1")

    @unittest.skipIf(not FAISS_AVAILABLE, "FAISS is not installed/available in the environment")
    def test_faiss_vector_store(self) -> None:
        # Reset and add
        self.faiss_store.reset()
        vids = self.faiss_store.add_texts(["hello world", "goodbye world"])
        self.assertEqual(vids, [0, 1])

        # Search
        results = self.faiss_store.search("hello", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["vector_id"], 0)

        # Index saving/loading
        test_dir = "data/test_faiss"
        test_path = os.path.join(test_dir, "test.index")
        try:
            self.faiss_store.save(test_path)
            self.assertTrue(os.path.exists(test_path))

            new_store = FaissVectorStore(dim=8, metric="cosine")
            new_store.load(test_path)
            self.assertEqual(new_store.vector_id_counter, 2)
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_memory_leakage_prevention(self) -> None:
        # Ingest past and future memories
        t = dt.datetime(2026, 1, 2, 12, 0, 0) # cutoff

        # Safe memory (available_at = 11:00)
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
            importance=0.5
        )

        # Future memory (available_at = 13:00 - Leakage threat)
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
            importance=0.9
        )

        # Search with as_of = 12:00
        results = self.service.search("TSLA announcement", top_k=5, as_of=t)

        # Search should return only mem_safe because mem_future is not available yet
        memory_ids = [res["memory_id"] for res in results]
        self.assertIn("mem_safe", memory_ids)
        self.assertNotIn("mem_future", memory_ids)


if __name__ == "__main__":
    unittest.main()
