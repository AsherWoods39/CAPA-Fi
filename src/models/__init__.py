"""
CAPA-Fi: Models Package (Member 2)
Permutation-Invariant Multi-User Modeling for Wi-Fi HAR.
"""

from src.models.backbone import SpatialTemporalBackbone
from src.models.classifier import MultiSlotLinear
from src.models.rotation_head import BinaryRotationMLP
from src.models.har_model import HARModel
from src.models.matcher import HungarianMatcher
from src.models.losses import MatchedCrossEntropyLoss

__all__ = [
    "SpatialTemporalBackbone",
    "MultiSlotLinear",
    "BinaryRotationMLP",
    "HARModel",
    "HungarianMatcher",
    "MatchedCrossEntropyLoss",
]
