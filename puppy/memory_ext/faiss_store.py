import os
import numpy as np
from typing import Any, Callable

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissVectorStore:
    """FAISS-backed vector memory store.
    
    If 'cosine' metric is selected, vectors are L2-normalized before insertion
    and search, so the Inner Product index (IndexFlatIP) calculates cosine similarity.
    If 'l2' is selected, distance metrics are returned.
    """
    def __init__(
        self,
        dim: int | None = None,
        index_path: str | None = None,
        embedding_func: Callable[[list[str]], np.ndarray] | None = None,
        metric: str = "cosine"
    ) -> None:
        self.dim = dim or 1536
        self.index_path = index_path
        self.embedding_func = embedding_func
        self.metric = metric.lower()
        self.vector_id_counter = 0
        self.index = None
        
        if FAISS_AVAILABLE:
            self.reset()

    def _get_mock_embeddings(self, texts: list[str]) -> np.ndarray:
        """Fallback deterministic mock embeddings for testing without APIs."""
        embs = []
        for text in texts:
            # Generate a deterministic vector based on characters
            np.random.seed(sum(ord(c) for c in text) % 997)
            vec = np.random.randn(self.dim).astype(np.float32)
            embs.append(vec)
        return np.vstack(embs)

    def add_texts(self, texts: list[str], metadata_ids: list[int | str] | None = None) -> list[int]:
        if not FAISS_AVAILABLE:
            # Mock behavior if FAISS is not installed
            vector_ids = [self.vector_id_counter + i for i in range(len(texts))]
            self.vector_id_counter += len(texts)
            return vector_ids
            
        if self.embedding_func is not None:
            embeddings = self.embedding_func(texts)
        else:
            embeddings = self._get_mock_embeddings(texts)
            
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}")
            
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
            
        self.index.add(embeddings)
        vector_ids = [self.vector_id_counter + i for i in range(len(texts))]
        self.vector_id_counter += len(texts)
        return vector_ids

    def search(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not FAISS_AVAILABLE:
            return []
            
        if self.index.ntotal == 0:
            return []
            
        if self.embedding_func is not None:
            query_embedding = self.embedding_func([query_text])
        else:
            query_embedding = self._get_mock_embeddings([query_text])
            
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
            
        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (score, vector_id) in enumerate(zip(D[0], I[0])):
            if vector_id == -1:
                continue
            results.append({
                "vector_id": int(vector_id),
                "score": float(score),
                "rank": rank + 1
            })
        return results

    def save(self, path: str | None = None) -> None:
        if not FAISS_AVAILABLE:
            return
            
        target_path = path or self.index_path
        if not target_path:
            raise ValueError("No path specified to save the FAISS index.")
            
        dir_name = os.path.dirname(target_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        faiss.write_index(self.index, target_path)

    def load(self, path: str | None = None) -> None:
        if not FAISS_AVAILABLE:
            return
            
        target_path = path or self.index_path
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"FAISS index file not found at: {target_path}")
            
        self.index = faiss.read_index(target_path)
        self.dim = self.index.d
        self.vector_id_counter = self.index.ntotal

    def reset(self) -> None:
        if not FAISS_AVAILABLE:
            return
            
        if self.metric == "cosine":
            # Inner Product index
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.metric == "l2":
            # L2 Euclidean Distance index
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Use 'cosine' or 'l2'.")
        self.vector_id_counter = 0
