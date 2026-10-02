"""
CAPA-Fi: Matched Cross-Entropy Loss (Member 2)

Computes permutation-invariant classification loss by:
1. Using the Hungarian matcher's assignment to reorder predictions (or targets)
2. Computing cross-entropy only on matched (prediction, target) pairs

This is where permutation invariance is actually enforced and must be tested:
    If target_slots = [Walk, Sit] or [Sit, Walk], the loss should be identical
    for the same underlying correct predictions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.matcher import HungarianMatcher


class MatchedCrossEntropyLoss(nn.Module):
    """
    Permutation-invariant cross-entropy loss using Hungarian matching.

    Steps:
        1. Run Hungarian matcher to find optimal slot assignment
        2. Reorder predictions to match ground-truth slot ordering
        3. Compute cross-entropy on the matched pairs
        4. Average across slots and batch

    Args:
        matcher: HungarianMatcher instance (default: creates one)
    """

    def __init__(self, matcher: Optional[HungarianMatcher] = None):
        super().__init__()
        self.matcher = matcher or HungarianMatcher()

    def forward(
        self,
        logits: torch.Tensor,
        target_slots: torch.Tensor,
        assignments: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> torch.Tensor:
        """
        Compute matched cross-entropy loss.

        Args:
            logits: Predicted logits [B, M, K+1]
            target_slots: Ground-truth class indices [B, M], values in {0..K}
            assignments: Optional pre-computed assignments from matcher.
                         If None, matcher is called internally.

        Returns:
            loss: Scalar tensor (mean over batch and slots), with gradient
        """
        B, M, num_classes = logits.shape

        # Get assignments if not provided
        if assignments is None:
            assignments = self.matcher.match(logits, target_slots)

        # Reorder predictions per assignment and compute CE
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        for b in range(B):
            row_ind, col_ind = assignments[b]

            for k in range(len(row_ind)):
                pred_slot = row_ind[k]   # predicted slot index
                gt_slot = col_ind[k]     # matched ground-truth slot index

                # Cross-entropy for this matched pair
                pred_logit = logits[b, pred_slot, :]       # [K+1]
                gt_class = target_slots[b, gt_slot]        # scalar

                loss_k = F.cross_entropy(
                    pred_logit.unsqueeze(0),                # [1, K+1]
                    gt_class.unsqueeze(0).long(),           # [1]
                )
                total_loss = total_loss + loss_k

        # Average over all matched pairs
        total_loss = total_loss / (B * M)

        return total_loss


class SourceTrainingLoss(nn.Module):
    """
    Combined source training loss:
        L_source = L_matched_CE + lambda_rot_source × L_rotation

    Args:
        lambda_rot_source: Weight for the rotation SSL loss.
                           Must not be None — will raise ValueError.
        matcher: Optional HungarianMatcher instance.
    """

    def __init__(
        self,
        lambda_rot_source: float,
        matcher: Optional[HungarianMatcher] = None,
    ):
        super().__init__()

        if lambda_rot_source is None:
            raise ValueError(
                "lambda_rot_source must be set in config (source_training.lambda_rot_source). "
                "This is an open team decision — do not hard-code a default value."
            )

        self.lambda_rot_source = lambda_rot_source
        self.matched_ce = MatchedCrossEntropyLoss(matcher=matcher)

    def forward(
        self,
        logits: torch.Tensor,
        target_slots: torch.Tensor,
        rot_logits: torch.Tensor,
        rot_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined source training loss.

        Args:
            logits: Predicted class logits [B, M, K+1]
            target_slots: Ground-truth class indices [B, M]
            rot_logits: Rotation head logits [2B, 2] (original + rotated)
            rot_labels: Rotation labels [2B] (0=original, 1=rotated)

        Returns:
            Dict with keys:
                - 'total': Total combined loss (scalar)
                - 'matched_ce': Matched cross-entropy loss (scalar)
                - 'rotation': Rotation CE loss (scalar)
        """
        # Matched cross-entropy (with Hungarian assignment)
        loss_ce = self.matched_ce(logits, target_slots)

        # Rotation SSL loss (standard cross-entropy)
        loss_rot = F.cross_entropy(rot_logits, rot_labels.long())

        # Combined loss
        total = loss_ce + self.lambda_rot_source * loss_rot

        return {
            "total": total,
            "matched_ce": loss_ce,
            "rotation": loss_rot,
        }
