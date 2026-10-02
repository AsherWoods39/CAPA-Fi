"""
CAPA-Fi: Binary Rotation SSL Head (Member 2)

Maps latent embedding z [B, D] → binary rotation logits [B, 2].
Predicts whether the input CSI was original (label 0) or 180°-inverted (label 1).

Architecture:
    Linear(D, rot_hidden) → ReLU → Linear(rot_hidden, 2)

This head consumes rotation labels produced upstream by Member 1's preprocessing.
It does NOT generate the 180° rotation transform itself.
"""

from typing import Dict

import torch
import torch.nn as nn


class BinaryRotationMLP(nn.Module):
    """
    Binary rotation prediction head for self-supervised learning.

    Classifies whether a CSI sample has been 180°-inverted or not.
    Output: [B, 2] raw logits (no softmax).

    Args:
        latent_dim: Input feature dimension D (default: 256)
        hidden_dim: Hidden layer dimension (default: 64)
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent embedding [B, D]

        Returns:
            rot_logits: Binary rotation logits [B, 2]
        """
        return self.head(z)

    @classmethod
    def from_config(cls, config: Dict) -> "BinaryRotationMLP":
        """
        Construct from a config dictionary (parsed from YAML).

        Expected config keys under 'model':
            - latent_dim_D: 256
            - rot_hidden_dim: 64
        """
        model_cfg = config.get("model", {})
        return cls(
            latent_dim=model_cfg.get("latent_dim_D", 256),
            hidden_dim=model_cfg.get("rot_hidden_dim", 64),
        )
