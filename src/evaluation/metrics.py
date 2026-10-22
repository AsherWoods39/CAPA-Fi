"""
CAPA-Fi: Evaluation Metrics Module

Provides comprehensive evaluation metrics for multi-user Wi-Fi HAR:
    1. Hungarian Matcher for Permutation-Invariant Multi-Slot Evaluation
    2. Slot-Wise Accuracy (active and all slots)
    3. Multi-User Occupancy Mean Absolute Error (MAE)
    4. Macro F1-Score across activity categories
    5. Exact Match Ratio (set-level match across all M slots)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def match_slots_hungarian(
    logits: torch.Tensor,
    target_slots: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solves optimal bipartite matching between predicted slot logits and ground truth slots.

    Args:
        logits: Model output logits [B, M, K+1]
        target_slots: Ground truth class indices [B, M]

    Returns:
        matched_preds: Permuted predicted class indices [B, M] aligned to target slots.
        matched_probs: Permuted predicted probabilities [B, M, K+1] aligned to target slots.
    """
    B, M, num_classes = logits.shape
    probs = F.softmax(logits, dim=-1)  # [B, M, K+1]

    device = logits.device
    matched_preds = torch.zeros((B, M), dtype=torch.long, device=device)
    matched_probs = torch.zeros((B, M, num_classes), dtype=torch.float32, device=device)

    # Convert to CPU for scipy assignment
    probs_cpu = probs.detach().cpu()
    target_cpu = target_slots.detach().cpu()

    for b in range(B):
        # Cost matrix: cost(m, j) = -log P(slot m predicts target class j)
        target_b = target_cpu[b]  # [M]
        # Clamp probabilities to [1e-7, 1.0] to prevent -inf in log space
        # (numerical underflow guard for cost matrix inversion stability)
        probs_clamped = torch.clamp(probs_cpu[b], min=1e-7, max=1.0)  # [M, K+1]
        log_p = torch.log(probs_clamped)  # [M, K+1]
        cost_matrix = -log_p[:, target_b].numpy()  # [M, M]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # col_ind tells which target slot matches row_ind (predicted slot)
        # To align predictions to target_slots order:
        aligned_pred_classes = torch.argmax(logits[b], dim=-1)  # [M]
        for pred_slot, tgt_slot in zip(row_ind, col_ind):
            matched_preds[b, tgt_slot] = aligned_pred_classes[pred_slot]
            matched_probs[b, tgt_slot] = probs[b, pred_slot]

    return matched_preds, matched_probs


def compute_slot_wise_accuracy(
    pred_slots: torch.Tensor,
    target_slots: torch.Tensor,
    ignore_empty: bool = False,
    no_person_idx: int = 0,
) -> float:
    """
    Computes slot-wise classification accuracy.

    Args:
        pred_slots: Predicted class indices [B, M]
        target_slots: Target class indices [B, M]
        ignore_empty: If True, calculates accuracy exclusively on active users (target != 0)
        no_person_idx: Class index for No-Person (default: 0)

    Returns:
        Accuracy as float in [0.0, 1.0].
    """
    if ignore_empty:
        mask = target_slots != no_person_idx
        if mask.sum() == 0:
            return 1.0
        correct = (pred_slots[mask] == target_slots[mask]).float().sum()
        total = mask.sum().float()
        return (correct / total).item()
    else:
        correct = (pred_slots == target_slots).float().sum()
        total = float(pred_slots.numel())
        return (correct / total).item()


def compute_occupancy_mae(
    pred_slots: torch.Tensor,
    target_slots: torch.Tensor,
    no_person_idx: int = 0,
) -> float:
    """
    Computes Mean Absolute Error (MAE) between estimated occupancy and ground truth.

    Occupancy is the count of co-present active users: count(slots != no_person_idx).

    Args:
        pred_slots: Predicted class indices [B, M]
        target_slots: Target class indices [B, M]
        no_person_idx: Class index for No-Person (default: 0)

    Returns:
        Occupancy MAE as float >= 0.0.
    """
    pred_occ = (pred_slots != no_person_idx).sum(dim=-1).float()
    true_occ = (target_slots != no_person_idx).sum(dim=-1).float()
    mae = torch.abs(pred_occ - true_occ).mean().item()
    return mae


def compute_macro_f1(
    pred_slots: torch.Tensor,
    target_slots: torch.Tensor,
    num_classes: int = 7,
    ignore_no_person: bool = False,
    no_person_idx: int = 0,
) -> float:
    """
    Computes macro-averaged F1 score across activity categories.

    Args:
        pred_slots: Predicted class indices [B, M]
        target_slots: Target class indices [B, M]
        num_classes: Total class count (default: 7)
        ignore_no_person: If True, excludes Class 0 (No-Person) from macro average

    Returns:
        Macro F1-score as float in [0.0, 1.0].
    """
    y_pred = pred_slots.detach().cpu().numpy().flatten()
    y_true = target_slots.detach().cpu().numpy().flatten()

    labels = list(range(1, num_classes)) if ignore_no_person else list(range(num_classes))
    if len(labels) == 0:
        return 0.0

    score = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return float(score)


def compute_exact_match_ratio(
    pred_slots: torch.Tensor,
    target_slots: torch.Tensor,
) -> float:
    """
    Computes exact set-match ratio: proportion of samples where all M slots are correct.

    Args:
        pred_slots: Predicted class indices [B, M]
        target_slots: Target class indices [B, M]

    Returns:
        Exact match ratio as float in [0.0, 1.0].
    """
    sample_matches = (pred_slots == target_slots).all(dim=-1).float()
    return sample_matches.mean().item()


def evaluate_predictions(
    logits: torch.Tensor,
    target_slots: torch.Tensor,
    num_classes: int = 7,
    no_person_idx: int = 0,
) -> Dict[str, float]:
    """
    Runs full evaluation pipeline:
        1. Hungarian bipartite matching to resolve slot permutations
        2. Computes slot-wise accuracy, active-only accuracy, occupancy MAE, macro F1, exact match.

    Args:
        logits: Model output logits [B, M, K+1]
        target_slots: Target class indices [B, M]
        num_classes: Number of total classes (7)
        no_person_idx: Index of No-Person class (0)

    Returns:
        Dictionary of computed evaluation metrics.
    """
    matched_preds, _ = match_slots_hungarian(logits, target_slots)

    slot_acc = compute_slot_wise_accuracy(matched_preds, target_slots, ignore_empty=False, no_person_idx=no_person_idx)
    active_acc = compute_slot_wise_accuracy(matched_preds, target_slots, ignore_empty=True, no_person_idx=no_person_idx)
    occ_mae = compute_occupancy_mae(matched_preds, target_slots, no_person_idx=no_person_idx)
    macro_f1 = compute_macro_f1(matched_preds, target_slots, num_classes=num_classes, ignore_no_person=False)
    macro_f1_active = compute_macro_f1(matched_preds, target_slots, num_classes=num_classes, ignore_no_person=True)
    exact_match = compute_exact_match_ratio(matched_preds, target_slots)

    return {
        "slot_wise_accuracy": slot_acc,
        "active_slot_accuracy": active_acc,
        "occupancy_mae": occ_mae,
        "macro_f1": macro_f1,
        "macro_f1_active": macro_f1_active,
        "exact_match_ratio": exact_match,
    }
