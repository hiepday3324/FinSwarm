import datetime as dt
from typing import Any

from .sqlite_memory_store import SQLiteMemoryStore
from .faiss_store import FaissVectorStore


class MemoryVectorService:
    """Composite memory service linking SQLite metadata and FAISS vector index.
    
    Provides unified memory insertion, persistence, and querying with time fencing.
    """
    def __init__(self, sqlite_store: SQLiteMemoryStore, faiss_store: FaissVectorStore) -> None:
        self.sqlite_store = sqlite_store
        self.faiss_store = faiss_store

    def add_memory(
        self,
        memory_id: str,
        agent_id: str,
        symbol: str,
        layer: str,
        text: str,
        event_date: dt.date | str,
        created_at: dt.datetime | str,
        available_at: dt.datetime | str,
        source: str = "system",
        importance: float = 0.0,
        metadata_dict: dict[str, Any] | None = None
    ) -> int:
        """Insert text to FAISS index and record metadata in SQLite."""
        # 1. Add vector to FAISS
        vector_ids = self.faiss_store.add_texts([text])
        vector_id = vector_ids[0]
        
        # 2. Add metadata to SQLite linked by vector_id
        self.sqlite_store.add_memory(
            memory_id=memory_id,
            vector_id=vector_id,
            agent_id=agent_id,
            symbol=symbol,
            layer=layer,
            text=text,
            event_date=event_date,
            created_at=created_at,
            available_at=available_at,
            source=source,
            importance=importance,
            metadata_dict=metadata_dict
        )
        return vector_id

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        as_of: dt.datetime | str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve top k memories matching query_text, filtered by temporal fencing."""
        # Fetch double the top_k elements from FAISS to allow filtering room
        faiss_results = self.faiss_store.search(query_text, top_k=top_k * 2)
        if not faiss_results:
            return self.sqlite_store.search_metadata(as_of=as_of, limit=top_k)
            
        vector_ids = [res["vector_id"] for res in faiss_results]
        
        # Fetch metadata records
        sqlite_records = self.sqlite_store.get_by_vector_ids(vector_ids)
        if not sqlite_records:
            return self.sqlite_store.search_metadata(as_of=as_of, limit=top_k)
        
        # Map back to FAISS scores and filter by time
        res_map = {res["vector_id"]: res for res in faiss_results}
        final_results = []
        
        as_of_str = as_of.isoformat() if isinstance(as_of, (dt.date, dt.datetime)) else str(as_of) if as_of else None
        
        for rec in sqlite_records:
            vid = rec["vector_id"]
            
            # Apply time filter
            if as_of_str is not None and rec["available_at"] > as_of_str:
                continue
                
            final_results.append({
                "memory_id": rec["memory_id"],
                "vector_id": vid,
                "agent_id": rec["agent_id"],
                "symbol": rec["symbol"],
                "layer": rec["layer"],
                "text": rec["text"],
                "score": res_map[vid]["score"],
                "rank": res_map[vid]["rank"],
                "event_date": rec["event_date"],
                "created_at": rec["created_at"],
                "available_at": rec["available_at"],
                "importance": rec["importance"],
            })
            
        # Re-sort to preserve FAISS relative similarity ranks
        final_results.sort(key=lambda x: x["rank"])
        return final_results[:top_k]

    def save(self, faiss_path: str | None = None) -> None:
        """Persist FAISS index to file."""
        self.faiss_store.save(faiss_path)

    def load(self, faiss_path: str | None = None) -> None:
        """Reload FAISS index from file."""
        self.faiss_store.load(faiss_path)

    def reset(self) -> None:
        """Clear both FAISS and SQLite stores."""
        self.faiss_store.reset()
        
        self.sqlite_store.connect()
        cursor = self.sqlite_store.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS memory_records")
        self.sqlite_store.initialize_schema()
