"""
The 'alibi.confidence' module includes trust scores.
"""

from .trustscore import TrustScore
from .model_linearity import LinearityMeasure, linearity_measure, infer_feature_range

__all__ = ["linearity_measure",
           "LinearityMeasure",
           "TrustScore",
           "infer_feature_range"]
