from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from puppy.common.schemas import AgentContext, TextFeature
from puppy.swarm.context_builder import build_raw_context


CANONICAL_TEXT_VIEW_VERSION = "full-causal-agent-context-v1"


@dataclass
class FrozenTransformerBackend:
    """Frozen Transformer mean-pooling backend owned by the canonical encoder."""

    model_name: str
    max_length: int = 512
    normalize: bool = True
    local_files_only: bool = False
    device: str = "cpu"
    tokenizer: Any | None = None
    model: Any | None = None
    last_effective_batch_size: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        import torch

        if self.tokenizer is None or self.model is None:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        self.model.to(torch.device(self.device))
        self.model.eval()

    @property
    def revision(self) -> str:
        config = getattr(self.model, "config", None)
        return str(getattr(config, "_commit_hash", None) or "unknown")

    @property
    def dimension(self) -> int:
        config = getattr(self.model, "config", None)
        value = getattr(config, "hidden_size", None)
        if value is None:
            raise ValueError("Encoder config does not expose hidden_size")
        return int(value)

    def encode_texts(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        effective_batch_size = min(int(batch_size), 8) if self.device == "cpu" else int(batch_size)
        self.last_effective_batch_size = effective_batch_size
        batches: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                encoded = self.tokenizer(
                    values[start : start + effective_batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                if self.normalize:
                    pooled = functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32))
        result = np.concatenate(batches, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Semantic encoder produced NaN or infinite values")
        return result

    def encode(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        """Compatibility alias; canonical callers use ``TextStreamEncoder``."""

        return self.encode_texts(texts, batch_size=batch_size)


class TextStreamEncoder:
    def __init__(
        self,
        emb_config: dict | None = None,
        embedding_func: Callable[[str], np.ndarray] | None = None,
        embedding_model: str | None = None,
        transformer_backend: FrozenTransformerBackend | None = None,
    ) -> None:
        self.emb_config = emb_config or {}
        self.embedding_func = embedding_func
        self.transformer_backend = transformer_backend
        self.embedding_model = (
            embedding_model
            or (transformer_backend.model_name if transformer_backend else None)
            or self.emb_config.get("embedding_model")
            or "text-embedding-ada-002"
        )
        self._embedding_backend = None
        if self.embedding_func is None and self.transformer_backend is None:
            from puppy.embedding import OpenAILongerThanContextEmb

            self._embedding_backend = OpenAILongerThanContextEmb(**self.emb_config)

    @classmethod
    def from_transformer(
        cls,
        model_name: str,
        *,
        max_length: int = 512,
        normalize: bool = True,
        local_files_only: bool = False,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> "TextStreamEncoder":
        backend = FrozenTransformerBackend(
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            local_files_only=local_files_only,
            device=device,
            tokenizer=tokenizer,
            model=model,
        )
        return cls(transformer_backend=backend, embedding_model=model_name)

    @property
    def dimension(self) -> int | None:
        return self.transformer_backend.dimension if self.transformer_backend else None

    @property
    def revision(self) -> str:
        return self.transformer_backend.revision if self.transformer_backend else "unknown"

    @property
    def last_effective_batch_size(self) -> int | None:
        return (
            self.transformer_backend.last_effective_batch_size
            if self.transformer_backend
            else None
        )

    def coerce_context(self, value: AgentContext | Mapping[str, Any] | str) -> AgentContext:
        if isinstance(value, AgentContext):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Serialized AgentContext must be a JSON object, not arbitrary text"
                ) from exc
        if not isinstance(value, Mapping):
            raise TypeError("TextStreamEncoder requires AgentContext or serialized AgentContext")
        return AgentContext.model_validate(dict(value))

    def serialize_context(self, context: AgentContext | Mapping[str, Any] | str) -> str:
        return build_raw_context(self.coerce_context(context))

    def encode_batch_arrays(
        self,
        contexts: Iterable[AgentContext | Mapping[str, Any] | str],
        *,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, list[str]]:
        normalized = [self.coerce_context(value) for value in contexts]
        raw_contexts = [build_raw_context(value) for value in normalized]
        if self.transformer_backend is not None:
            matrix = self.transformer_backend.encode_texts(
                raw_contexts,
                batch_size=batch_size,
            )
        else:
            vectors = []
            for raw_context in raw_contexts:
                if self.embedding_func is not None:
                    vector = self.embedding_func(raw_context)
                else:
                    vector = self._embedding_backend(raw_context)  # type: ignore[operator]
                array = np.asarray(vector, dtype=np.float32)
                if array.ndim == 2 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim != 1:
                    raise ValueError("Embedding backend must return one vector per context")
                vectors.append(array)
            if not vectors:
                dimension = self.dimension or 0
                matrix = np.empty((0, dimension), dtype=np.float32)
            else:
                dimensions = {vector.shape[0] for vector in vectors}
                if len(dimensions) != 1:
                    raise ValueError("Embedding backend returned inconsistent dimensions")
                matrix = np.stack(vectors).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Text encoder produced NaN or infinite values")
        return matrix.astype(np.float32), raw_contexts

    def encode_array(
        self,
        context: AgentContext | Mapping[str, Any] | str,
    ) -> tuple[np.ndarray, str]:
        matrix, raw_contexts = self.encode_batch_arrays([context], batch_size=1)
        return matrix[0], raw_contexts[0]

    def encode(self, context: AgentContext | Mapping[str, Any] | str) -> TextFeature:
        normalized = self.coerce_context(context)
        vector, raw_context = self.encode_array(normalized)
        return TextFeature(
            agent_id=normalized.agent_id,
            symbol=normalized.symbol,
            date=normalized.date,
            h_text=vector.tolist(),
            raw_context=raw_context,
            embedding_model=self.embedding_model,
            dim=int(vector.shape[0]),
            dtype="float32",
            as_of=normalized.as_of,
            context_hash=normalized.context_hash,
            encoder_version=(
                f"transformers-frozen:{self.revision}"
                if self.transformer_backend
                else self.embedding_model
            ),
            metadata={"text_view_version": CANONICAL_TEXT_VIEW_VERSION},
        )

    def encode_batch(
        self,
        contexts: Iterable[AgentContext | Mapping[str, Any] | str],
        *,
        batch_size: int = 32,
    ) -> list[TextFeature]:
        normalized = [self.coerce_context(value) for value in contexts]
        matrix, raw_contexts = self.encode_batch_arrays(normalized, batch_size=batch_size)
        return [
            TextFeature(
                agent_id=context.agent_id,
                symbol=context.symbol,
                date=context.date,
                h_text=vector.tolist(),
                raw_context=raw_context,
                embedding_model=self.embedding_model,
                dim=int(vector.shape[0]),
                dtype="float32",
                as_of=context.as_of,
                context_hash=context.context_hash,
                encoder_version=(
                    f"transformers-frozen:{self.revision}"
                    if self.transformer_backend
                    else self.embedding_model
                ),
                metadata={"text_view_version": CANONICAL_TEXT_VIEW_VERSION},
            )
            for context, vector, raw_context in zip(normalized, matrix, raw_contexts)
        ]
