"""
CAPA-Fi: Category-Aware Partial Adaptation Filtering for Wi-Fi Sensing
Module 3: Source-Free Adaptation & SSL Engine (Member 3: Athishta P. A.)
"""

from src.adaptation.confidence_masking import (
    CategoryMaskEngine,
    compute_occupancy,
    compute_probs,
    compute_sample_confidence,
    compute_weights_and_masks,
)
from src.adaptation.loss_engine import (
    CAPAFiLossEngine,
    compute_entropy_loss,
    compute_masked_diversity_loss,
    compute_rotation_loss,
)
from src.adaptation.trainer import TargetAdaptationTrainer

__all__ = [
    "mock_inputs",
    "compute_probs",
    "compute_sample_confidence",
    "compute_occupancy",
    "CategoryMaskEngine",
    "compute_weights_and_masks",
    "compute_entropy_loss",
    "compute_masked_diversity_loss",
    "compute_rotation_loss",
    "CAPAFiLossEngine",
    "TargetAdaptationTrainer",
]
