from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puppy.fusion.scalar_logit_stack import ScalarLogitStack
from puppy.fusion.text_score_head import TextScoreHead
from puppy.training.checkpoint import TrainOnlyScaler, state_dict_sha256
from puppy.training.fusion_trainer import (
    SCALAR_FUSION_SELECTION,
    load_fusion_model,
)
from puppy.training.graph_trainer import load_gat
from puppy.training.text_trainer import predict_mlp

from demo.constants import DEFAULT_BUNDLE_DIR, NODE_ORDER


class FrozenReplayEngine:
    def __init__(
        self,
        bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
        *,
        device: str = "cpu",
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        self.device = device
        self.text_model = TextScoreHead.load(
            self.bundle_dir / "text_model.pt", device=device
        )
        self.graph_model = load_gat(self.bundle_dir / "graph_model.pt", device=device)
        fusion_model, selected = load_fusion_model(
            self.bundle_dir / "fusion_model.pt", device=device
        )
        if selected != SCALAR_FUSION_SELECTION or not isinstance(
            fusion_model, ScalarLogitStack
        ):
            raise ValueError("Replay requires the trained ScalarLogitStack checkpoint")
        self.fusion_model = fusion_model
        self.text_scaler = TrainOnlyScaler.load(self.bundle_dir / "text_scaler.json")
        self.graph_scaler = TrainOnlyScaler.load(self.bundle_dir / "graph_scaler.json")
        self.feature_schema = json.loads(
            (self.bundle_dir / "feature_schema.json").read_text(encoding="utf-8")
        )
        with np.load(self.bundle_dir / "replay_inputs.npz", allow_pickle=False) as data:
            self.dates = data["dates"].astype(str)
            self.symbols = tuple(data["symbols"].astype(str).tolist())
            self.decision_ids = data["decision_ids"].astype(str)
            self.text_embeddings = np.asarray(data["text_embeddings"], dtype=np.float32)
            self.raw_graph_features = np.asarray(
                data["raw_graph_features"], dtype=np.float32
            )
            self.graph_feature_mask = np.asarray(data["graph_feature_mask"], dtype=bool)
            self.adjacency = np.asarray(data["adjacency"], dtype=np.float32)
            self.graph_feature_names = tuple(
                data["graph_feature_names"].astype(str).tolist()
            )
        if self.symbols != NODE_ORDER:
            raise ValueError("Frozen replay node order mismatch")
        if tuple(self.feature_schema["feature_names"]) != self.graph_feature_names:
            raise ValueError("Frozen replay Graph feature schema mismatch")
        self.text_score_index = self.graph_feature_names.index("text_score")
        with np.load(
            self.bundle_dir / "replay_expected_outputs.npz", allow_pickle=False
        ) as data:
            self.expected = {name: np.asarray(data[name]) for name in data.files}
        self.coefficients = dict(
            torch.load(
                self.bundle_dir / "fusion_model.pt",
                map_location="cpu",
                weights_only=False,
            )["coefficients"]
        )
        self._freeze_all()
        self.startup_state_hashes = self.state_hashes()

    def _freeze_all(self) -> None:
        for module in (self.text_model, self.graph_model, self.fusion_model):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def state_hashes(self) -> dict[str, str]:
        return {
            "text_state_sha256": state_dict_sha256(self.text_model.state_dict()),
            "gat_state_sha256": state_dict_sha256(self.graph_model.state_dict()),
            "scalar_fusion_state_sha256": state_dict_sha256(
                self.fusion_model.state_dict()
            ),
        }

    @property
    def all_parameters_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for module in (self.text_model, self.graph_model, self.fusion_model)
            for parameter in module.parameters()
        )

    def infer_date(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.dates):
            raise IndexError(f"Replay date index out of range: {index}")
        before = self.state_hashes()
        embeddings = self.text_embeddings[index]
        text_features = self.text_scaler.transform(embeddings)
        text_logits, text_probability = predict_mlp(
            self.text_model, text_features, device=self.device
        )

        raw_graph = self.raw_graph_features[index].copy()
        raw_graph[:, self.text_score_index] = text_probability
        feature_mask = self.graph_feature_mask[index]
        values = np.where(feature_mask, raw_graph, np.nan)
        scaled_graph = self.graph_scaler.transform(values)
        scaled_graph = np.where(feature_mask, scaled_graph, 0.0).astype(np.float32)
        with torch.no_grad():
            graph_output = self.graph_model(
                torch.as_tensor(
                    scaled_graph[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    self.adjacency[index][None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    feature_mask[None], dtype=torch.bool, device=self.device
                ),
                symbols=self.symbols,
                feature_names=self.graph_feature_names,
                feature_provenance={"source": "frozen_train_replay"},
            )
            if self.graph_model.last_attention_heads is None:
                raise RuntimeError("GAT did not expose sample-specific attention")
            graph_logits = graph_output.logits[0].detach().cpu().numpy()
            graph_probability = graph_output.probabilities[0].detach().cpu().numpy()
            attention_heads = (
                self.graph_model.last_attention_heads[0].detach().cpu().numpy()
            )
            fusion_logits = self.fusion_model(
                torch.as_tensor(text_logits, dtype=torch.float32, device=self.device),
                torch.as_tensor(
                    graph_logits, dtype=torch.float32, device=self.device
                ),
            ).detach().cpu().numpy()
        fusion_probability = 1.0 / (
            1.0 + np.exp(-np.clip(fusion_logits, -60.0, 60.0))
        )
        after = self.state_hashes()
        if before != after or before != self.startup_state_hashes:
            raise RuntimeError("Frozen model parameters changed during replay inference")
        return {
            "date": str(self.dates[index]),
            "date_index": index,
            "symbols": self.symbols,
            "decision_ids": self.decision_ids[index].copy(),
            "text_logits": text_logits.astype(np.float32),
            "text_probability": text_probability.astype(np.float32),
            "raw_graph_features": raw_graph.astype(np.float32),
            "scaled_graph_features": scaled_graph,
            "graph_feature_mask": feature_mask.copy(),
            "adjacency": self.adjacency[index].copy(),
            "graph_logits": graph_logits.astype(np.float32),
            "graph_probability": graph_probability.astype(np.float32),
            "attention_heads": attention_heads.astype(np.float32),
            "attention_mean": attention_heads.mean(axis=0).astype(np.float32),
            "fusion_logits": fusion_logits.astype(np.float32),
            "fusion_probability": fusion_probability.astype(np.float32),
            "model_state_hash_before": before,
            "model_state_hash_after": after,
        }

    def assert_expected(self, index: int, *, atol: float = 1e-6) -> dict[str, float]:
        output = self.infer_date(index)
        mapping = {
            "text_logits": output["text_logits"],
            "text_probability": output["text_probability"],
            "graph_logits": output["graph_logits"],
            "graph_probability": output["graph_probability"],
            "attention_heads": output["attention_heads"],
            "attention_mean": output["attention_mean"],
            "fusion_logits": output["fusion_logits"],
            "fusion_probability": output["fusion_probability"],
        }
        deltas: dict[str, float] = {}
        for name, actual in mapping.items():
            expected = self.expected[name][index]
            delta = float(np.max(np.abs(np.asarray(actual) - expected)))
            deltas[name] = delta
            if not np.allclose(actual, expected, atol=atol, rtol=1e-6):
                raise AssertionError(
                    f"Frozen replay output differs from training artifact: {name}"
                )
        return deltas

    def fusion_decomposition(self, output: dict[str, Any]) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        intercept = float(self.coefficients["intercept"])
        weight_text = float(self.coefficients["weight_text"])
        weight_graph = float(self.coefficients["weight_graph"])
        for index, symbol in enumerate(self.symbols):
            text_term = weight_text * float(output["text_logits"][index])
            graph_term = weight_graph * float(output["graph_logits"][index])
            reconstructed = intercept + text_term + graph_term
            rows.append(
                {
                    "symbol": symbol,
                    "intercept_contribution": intercept,
                    "text_contribution": text_term,
                    "graph_contribution": graph_term,
                    "fusion_logit": float(output["fusion_logits"][index]),
                    "reconstructed_logit": reconstructed,
                    "fusion_probability": float(output["fusion_probability"][index]),
                }
            )
        return rows
