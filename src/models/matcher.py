"""
CAPA-Fi: Hungarian Bipartite Matcher (Member 2)

Finds the optimal assignment between predicted slots and ground-truth slots
using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

Builds a [M × M] cost matrix per sample where:
    cost[i, j] = -log_softmax(predicted_slot_i)[ground_truth_class_j]

CRITICAL: This module is NON-DIFFERENTIABLE.
    - Predictions are detached before cost computation.
    - No gradient flows through this file.
    - Only assignment indices are returned.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    """
    Hungarian bipartite matcher for permutation-invariant multi-user prediction.

    For each sample in the batch, computes the optimal (minimum-cost) assignment
    between M predicted slots and M ground-truth slots.

    The cost for assigning predicted slot i to ground-truth slot j is the
    negative log-probability of ground-truth class j under predicted slot i's
    softmax distribution.
    """

    @staticmethod
    def compute_cost_matrix(
        logits: torch.Tensor,
        target_slots: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute cost matrices for a batch of samples.

        Args:
            logits: Predicted logits [B, M, K+1] (DETACHED, no gradient)
            target_slots: Ground-truth class indices [B, M], values in {0..K}

        Returns:
            cost_matrices: numpy array [B, M, M] of assignment costs
        """
        B, M, num_classes = logits.shape

        # Compute log-softmax over classes (last dim)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, M, K+1]

        # Build cost matrix: cost[b, i, j] = -log_prob[b, i, target[b, j]]
        cost_matrices = np.zeros((B, M, M), dtype=np.float64)

        log_probs_np = log_probs.detach().cpu().numpy()
        target_np = target_slots.detach().cpu().numpy().astype(int)

        for b in range(B):
            for i in range(M):
                for j in range(M):
                    target_class = target_np[b, j]
                    cost_matrices[b, i, j] = -log_probs_np[b, i, target_class]

        return cost_matrices

    @staticmethod
    @torch.no_grad()
    def match(
        logits: torch.Tensor,
        target_slots: torch.Tensor,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find optimal assignment between predicted slots and ground-truth slots.

        Args:
            logits: Predicted logits [B, M, K+1]
            target_slots: Ground-truth class indices [B, M], values in {0..K}

        Returns:
            assignments: List of (row_indices, col_indices) tuples, one per sample.
                         row_indices[k] is the predicted slot index,
                         col_indices[k] is the matched ground-truth slot index.
        """
        # CRITICAL: Detach to prevent any gradient flow
        logits_detached = logits.detach()

        cost_matrices = HungarianMatcher.compute_cost_matrix(
            logits_detached, target_slots
        )

        B = cost_matrices.shape[0]
        assignments = []

        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(cost_matrices[b])
            assignments.append((row_ind, col_ind))

        return assignments
