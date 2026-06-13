from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any

import faiss
import numpy as np

FAISS_AVAILABLE = True


class FaissVectorStore:
    """FAISS-backed vector memory store."""

    def __init__(
        self,
        dim: int | None = None,
        index_path: str | None = None,
        embedding_func: Callable[[list[str]], np.ndarray] | None = None,
        metric: str = "cosine",
    ) -> None:
        self.dim = dim or 1536
        self.index_path = index_path
        self.embedding_func = embedding_func
        self.metric = metric.lower()
        self.vector_id_counter = 0
        self.index = None
        self.reset()

    @property
    def backend_name(self) -> str:
        return "faiss"

    @property
    def supports_persistence(self) -> bool:
        return True

    def _get_mock_embeddings(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            embeddings.append(rng.normal(size=self.dim).astype(np.float32))
        return np.vstack(embeddings) if embeddings else np.empty((0, self.dim), dtype=np.float32)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedding_func(texts) if self.embedding_func else self._get_mock_embeddings(texts)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}")
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        return embeddings

    def add_texts(self, texts: list[str], metadata_ids: list[int | str] | None = None) -> list[int]:
        embeddings = self._embed_texts(texts)
        self.index.add(embeddings)
        vector_ids = [self.vector_id_counter + index for index in range(len(texts))]
        self.vector_id_counter += len(texts)
        return vector_ids

    def search(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0 or self.index.ntotal == 0:
            return []
        query_embedding = self._embed_texts([query_text])
        scores, ids = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        return [
            {"vector_id": int(vector_id), "score": float(score), "rank": rank + 1}
            for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]))
            if int(vector_id) != -1
        ]

    def save(self, path: str | None = None) -> None:
        target_path = path or self.index_path
        if not target_path:
            raise ValueError("No path specified to save the FAISS index.")
        dir_name = os.path.dirname(target_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        faiss.write_index(self.index, target_path)

    def load(self, path: str | None = None) -> None:
        target_path = path or self.index_path
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"FAISS index file not found at: {target_path}")
        self.index = faiss.read_index(target_path)
        self.dim = self.index.d
        self.vector_id_counter = self.index.ntotal

    def reset(self) -> None:
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Use 'cosine' or 'l2'.")
        self.vector_id_counter = 0
