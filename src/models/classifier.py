"""
CAPA-Fi: Multi-Slot Classification Head (Member 2)

Maps latent embedding z [B, D] → multi-slot logits [B, M, K+1].

Architecture:
    Linear(D, M × (K+1)) → reshape → [B, M, K+1]

No softmax is applied — raw logits are returned.
Cross-entropy loss applies log-softmax internally.

This module must be independently freezable for the adaptation phase:
    model.classifier.requires_grad_(False)
"""

from typing import Dict

import torch
import torch.nn as nn


class MultiSlotLinear(nn.Module):
    """
    Multi-slot linear classifier for permutation-invariant multi-user prediction.

    Predicts activity class logits for M concurrent user slots.
    Output shape: [B, M, K+1] where K+1 includes the No-Person class (index 0).

    Args:
        latent_dim: Input feature dimension D (default: 256)
        num_slots: Number of prediction slots M (default: 6)
        total_classes: Total number of classes K+1 including No-Person (default: 7)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_slots: int = 6,
        total_classes: int = 7,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.total_classes = total_classes
        self.fc = nn.Linear(latent_dim, num_slots * total_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent embedding [B, D] = [B, 256]

        Returns:
            logits: Multi-slot class logits [B, M, K+1] = [B, 6, 7]
        """
        B = z.size(0)
        out = self.fc(z)  # [B, M*(K+1)] = [B, 42]
        return out.view(B, self.num_slots, self.total_classes)  # [B, M, K+1]

    @classmethod
    def from_config(cls, config: Dict) -> "MultiSlotLinear":
        """
        Construct from a config dictionary (parsed from YAML).

        Expected config keys under 'model':
            - latent_dim_D: 256
            - max_users_M: 6
            - total_classes: 7
        """
        model_cfg = config.get("model", {})
        return cls(
            latent_dim=model_cfg.get("latent_dim_D", 256),
            num_slots=model_cfg.get("max_users_M", 6),
            total_classes=model_cfg.get("total_classes", 7),
        )
