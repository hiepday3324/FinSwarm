from .gated_fusion import (
    GatedFusion,
    LatentGatedFusion,
    LegacyScalarGatedFusion,
    build_fusion_output,
    build_latent_fusion_output,
)
from .text_feature_contract import (
    TextFeatureContractError,
    build_text_feature_matrix,
    validate_text_feature,
    validate_text_feature_batch,
)
from .text_encoder import TextStreamEncoder
from .text_score_head import NumpyLogisticTextScoreBaseline, TextScoreHead

__all__ = [
    "GatedFusion",
    "LatentGatedFusion",
    "LegacyScalarGatedFusion",
    "NumpyLogisticTextScoreBaseline",
    "TextScoreHead",
    "TextFeatureContractError",
    "TextStreamEncoder",
    "build_fusion_output",
    "build_latent_fusion_output",
    "build_text_feature_matrix",
    "validate_text_feature",
    "validate_text_feature_batch",
]
