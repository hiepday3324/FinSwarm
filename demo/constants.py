from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "swarm" / "training_e4_demo_fit.yaml"
DEFAULT_BUNDLE_DIR = REPO_ROOT / "data" / "model_artifacts" / "finswarm_e4_demo_fit"
SCENARIO_DIR = REPO_ROOT / "demo" / "scenarios"

NODE_ORDER = ("TSLA", "NVDA", "AAPL")
DISPLAY_ORDER = ("AAPL", "NVDA", "TSLA")

TRAIN_START = "2024-09-06"
TRAIN_END = "2025-08-11"
EXPECTED_TRAIN_DATES = 232
EXPECTED_ALIGNED_ROWS = EXPECTED_TRAIN_DATES * len(NODE_ORDER)
EXPECTED_SUPERVISED_ROWS = 651

TEXT_SCORE_SOURCE = "in_sample_demo"
GRAPH_PREDICTION_SOURCE = "in_sample_demo"
FUSION_PREDICTION_SOURCE = "in_sample_demo"

ATTENTION_CONVENTION = {
    "tensor_shape": "[batch, heads, targets, sources]",
    "mean_tensor_shape": "[batch, targets, sources]",
    "row_axis": "target",
    "column_axis": "source",
    "edge_direction": "source_to_target",
    "coefficient_lookup": "alpha[target, source]",
    "normalization_axis": -1,
    "normalization_axis_name": "source",
    "terminology": {
        "parameters": "Frozen trained GAT parameters",
        "attention": "Sample-specific attention coefficients",
        "recalculation": "Attention recalculation",
        "change": "Attention shift between replay dates",
    },
}

UNAVAILABLE = "Unavailable in artifact"

