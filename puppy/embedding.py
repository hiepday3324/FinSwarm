from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, List, Union

import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings


class OpenAILongerThanContextEmb:
    """Explicit legacy OpenAI memory embedding backend."""

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.openai_api_key,
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        values = [text] if isinstance(text, str) else text
        return self.emb_model.embed_documents(texts=values, chunk_size=None)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        return np.asarray(self._emb(text), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        match self.emb_model.model:
            case "text-embedding-ada-002":
                return 1536
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.emb_model.model} not implemented"
                )


class SentenceTransformerMemoryEmb:
    """Local sentence-transformer memory embeddings with an optional shared encoder."""

    def __init__(
        self,
        embedding_model: str,
        *,
        transformer_backend: Any | None = None,
        model: Any | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        if embedding_model.strip().lower() == "qwen3:4b":
            raise ValueError("qwen3:4b is a chat model and cannot be used for memory embeddings")
        if transformer_backend is not None and model is not None:
            raise ValueError("Provide transformer_backend or model, not both")
        self.embedding_model = embedding_model
        self.transformer_backend = transformer_backend
        self.model = model
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        if self.batch_size <= 0:
            raise ValueError("Memory embedding batch_size must be positive")
        if self.transformer_backend is None and self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                embedding_model,
                device=device,
                local_files_only=local_files_only,
            )
        dimension = (
            getattr(self.transformer_backend, "dimension", None)
            if self.transformer_backend is not None
            else self.model.get_sentence_embedding_dimension()
        )
        if dimension is None or int(dimension) <= 0:
            raise ValueError("Memory embedding model does not expose a valid dimension")
        self._dimension = int(dimension)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        values = [text] if isinstance(text, str) else list(text)
        if not values:
            return np.empty((0, self._dimension), dtype=np.float32)
        if self.transformer_backend is not None:
            encoded = self.transformer_backend.encode_texts(
                values,
                batch_size=self.batch_size,
            )
        else:
            encoded = self.model.encode(
                values,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        result = np.asarray(encoded, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        expected = (len(values), self._dimension)
        if result.shape != expected:
            raise ValueError(
                f"Memory embedding shape mismatch: expected {expected}, got {result.shape}"
            )
        if not np.isfinite(result).all():
            raise ValueError("Memory embedding produced NaN or infinite values")
        return result

    def get_embedding_dimension(self) -> int:
        return self._dimension


def build_memory_embedding(
    config: Mapping[str, Any],
    *,
    transformer_backend: Any | None = None,
) -> OpenAILongerThanContextEmb | SentenceTransformerMemoryEmb:
    """Build the configured memory embedding backend without key-based inference."""

    values = dict(config)
    provider = str(values.pop("provider", "")).strip().lower()
    if not provider:
        raise ValueError(
            "Memory embedding provider is required; use 'sentence_transformers' "
            "or explicit legacy 'openai'"
        )
    if provider == "sentence_transformers":
        model_name = str(values.pop("embedding_model", "")).strip()
        if not model_name:
            raise ValueError("sentence_transformers memory embedding requires embedding_model")
        return SentenceTransformerMemoryEmb(
            model_name,
            transformer_backend=transformer_backend,
            **values,
        )
    if provider == "openai":
        if transformer_backend is not None:
            raise ValueError("OpenAI memory embedding cannot use a local transformer backend")
        return OpenAILongerThanContextEmb(**values)
    raise ValueError(f"Unsupported memory embedding provider: {provider!r}")
