from .gated_fusion import GatedFusion, build_fusion_output
from .text_feature_contract import (
    TextFeatureContractError,
    build_text_feature_matrix,
    validate_text_feature,
    validate_text_feature_batch,
)
from .text_encoder import TextStreamEncoder

__all__ = [
    "GatedFusion",
    "TextFeatureContractError",
    "TextStreamEncoder",
    "build_fusion_output",
    "build_text_feature_matrix",
    "validate_text_feature",
    "validate_text_feature_batch",
]
