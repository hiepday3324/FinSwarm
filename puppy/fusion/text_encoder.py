from collections.abc import Callable

import numpy as np

from puppy.common.schemas import AgentContext, TextFeature
from puppy.swarm.context_builder import build_raw_context


class TextStreamEncoder:
    def __init__(
        self,
        emb_config: dict | None = None,
        embedding_func: Callable[[str], np.ndarray] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.emb_config = emb_config or {}
        self.embedding_func = embedding_func
        self.embedding_model = (
            embedding_model
            or self.emb_config.get("embedding_model")
            or "text-embedding-ada-002"
        )
        self._embedding_backend = None
        if self.embedding_func is None:
            from puppy.embedding import OpenAILongerThanContextEmb

            self._embedding_backend = OpenAILongerThanContextEmb(**self.emb_config)

    def encode_array(self, context: AgentContext) -> tuple[np.ndarray, str]:
        raw_context = build_raw_context(context)
        if self.embedding_func is not None:
            vector = self.embedding_func(raw_context)
        else:
            vector = self._embedding_backend(raw_context)  # type: ignore[operator]
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 2:
            vector = vector[0]
        return vector.astype(np.float32), raw_context

    def encode(self, context: AgentContext) -> TextFeature:
        vector, raw_context = self.encode_array(context)
        return TextFeature(
            agent_id=context.agent_id,
            symbol=context.symbol,
            date=context.date,
            h_text=vector.tolist(),
            raw_context=raw_context,
            embedding_model=self.embedding_model,
            dim=int(vector.shape[0]),
            dtype="float32",
        )
