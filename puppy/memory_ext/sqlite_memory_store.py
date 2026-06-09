import os
import sqlite3
import json
import datetime as dt
from typing import Any

class SQLiteMemoryStore:
    def __init__(self, db_path: str = "data/sqlite/memory_metadata.db") -> None:
        self.db_path = db_path
        self.conn = None
        
        if db_path != ":memory:":
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

    def connect(self) -> None:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.initialize_schema()

    def initialize_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_records (
                memory_id TEXT PRIMARY KEY,
                vector_id INTEGER UNIQUE,
                agent_id TEXT,
                symbol TEXT,
                layer TEXT,
                text TEXT,
                event_date TEXT,
                created_at TEXT,
                available_at TEXT,
                source TEXT,
                importance REAL,
                metadata_json TEXT
            )
        """)
        self.conn.commit()

    def add_memory(
        self,
        memory_id: str,
        vector_id: int,
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
    ) -> None:
        self.connect()
        
        event_date_str = event_date.isoformat() if isinstance(event_date, (dt.date, dt.datetime)) else str(event_date)
        created_at_str = created_at.isoformat() if isinstance(created_at, (dt.date, dt.datetime)) else str(created_at)
        available_at_str = available_at.isoformat() if isinstance(available_at, (dt.date, dt.datetime)) else str(available_at)
        meta_json = json.dumps(metadata_dict or {})
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_records (
                memory_id, vector_id, agent_id, symbol, layer, text, event_date, created_at, available_at, source, importance, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, vector_id, agent_id, symbol, layer, text, event_date_str, created_at_str, available_at_str, source, importance, meta_json
        ))
        self.conn.commit()

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        self.connect()
        cursor = self.conn.cursor()
        row = cursor.execute("SELECT * FROM memory_records WHERE memory_id = ?", (memory_id,)).fetchone()
        return dict(row) if row else None

    def get_by_vector_ids(self, vector_ids: list[int]) -> list[dict[str, Any]]:
        if not vector_ids:
            return []
            
        self.connect()
        cursor = self.conn.cursor()
        placeholders = ", ".join("?" for _ in vector_ids)
        rows = cursor.execute(
            f"SELECT * FROM memory_records WHERE vector_id IN ({placeholders})",
            vector_ids
        ).fetchall()
        
        # Keep the order of vector_ids if needed, but a simple lookup is safer
        records = {row["vector_id"]: dict(row) for row in rows}
        return [records[vid] for vid in vector_ids if vid in records]

    def search_metadata(
        self,
        agent_id: str | None = None,
        symbol: str | None = None,
        layer: str | None = None,
        as_of: dt.datetime | str | None = None,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        self.connect()
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM memory_records WHERE 1=1"
        params = []
        
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        if layer is not None:
            query += " AND layer = ?"
            params.append(layer)
        if as_of is not None:
            as_of_str = as_of.isoformat() if isinstance(as_of, (dt.date, dt.datetime)) else str(as_of)
            query += " AND available_at <= ?"
            params.append(as_of_str)
            
        query += " ORDER BY event_date DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = cursor.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
