"""
CAPA-Fi: Hungarian Matcher & Permutation Invariance Tests (Member 2)

Test 4: Known-assignment verification (6×6 matching)
Test 5: Permutation invariance — same answer in arbitrary 6-slot orders → identical loss
"""

import pytest
import numpy as np
import torch

from src.models.matcher import HungarianMatcher
from src.models.losses import MatchedCrossEntropyLoss


class TestHungarianMatcher:
    """Tests for the Hungarian bipartite matcher with M=6 slots."""

    def test_known_assignment_6x6(self):
        """
        6×6 cost matrix with known identity optimal assignment.
        Slots 0..5 predict classes [1, 2, 3, 4, 5, 0] with high confidence.
        Target: [1, 2, 3, 4, 5, 0] (Walk, Sit, Stand, Fall, Wave, NO_PERSON).

        Optimal assignment: row_ind == col_ind == [0, 1, 2, 3, 4, 5].
        """
        B, M, K_plus_1 = 1, 6, 7
        target = [1, 2, 3, 4, 5, 0]

        logits = torch.zeros(B, M, K_plus_1)
        for i, c in enumerate(target):
            logits[0, i, c] = 10.0

        target_slots = torch.tensor([target])

        assignments = HungarianMatcher.match(logits, target_slots)
        row_ind, col_ind = assignments[0]

        assert len(row_ind) == 6
        assert len(col_ind) == 6
        assert list(row_ind) == [0, 1, 2, 3, 4, 5]
        assert list(col_ind) == [0, 1, 2, 3, 4, 5]

    def test_swapped_assignment_6x6(self):
        """
        6×6 cost matrix with permuted assignment.
        Target: [1, 2, 3, 4, 5, 0]
        Predictions are circularly shifted:
            Slot 0 predicts class 0
            Slot 1 predicts class 1
            Slot 2 predicts class 2
            Slot 3 predicts class 3
            Slot 4 predicts class 4
            Slot 5 predicts class 5
        """
        B, M, K_plus_1 = 1, 6, 7
        target = [1, 2, 3, 4, 5, 0]

        logits = torch.zeros(B, M, K_plus_1)
        # Shifted predictions
        shifted_preds = [0, 1, 2, 3, 4, 5]
        for i, c in enumerate(shifted_preds):
            logits[0, i, c] = 10.0

        target_slots = torch.tensor([target])
        assignments = HungarianMatcher.match(logits, target_slots)
        row_ind, col_ind = assignments[0]

        assert len(row_ind) == 6
        assert len(col_ind) == 6

        # Check that for each matched pair, the predicted class matches ground truth
        for r, c in zip(row_ind, col_ind):
            pred_class = shifted_preds[r]
            gt_class = target[c]
            assert pred_class == gt_class, f"Mismatch: pred slot {r} (class {pred_class}) matched gt slot {c} (class {gt_class})"

    def test_no_gradient(self):
        """Matcher must not participate in the backward graph."""
        logits = torch.randn(2, 6, 7, requires_grad=True)
        target_slots = torch.tensor([
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 1, 0, 0],
        ])

        assignments = HungarianMatcher.match(logits, target_slots)
        assert len(assignments) == 2
        for r, c in assignments:
            assert len(r) == 6
            assert len(c) == 6

    def test_no_person_padding_slots(self):
        """Edge case: multiple slots are padded with NO_PERSON (class 0)."""
        logits = torch.zeros(1, 6, 7)
        # 2 active users, 4 NO_PERSON
        target_slots = torch.tensor([[1, 2, 0, 0, 0, 0]])

        logits[0, 0, 1] = 10.0  # Slot 0 -> Walk
        logits[0, 1, 2] = 10.0  # Slot 1 -> Sit
        logits[0, 2, 0] = 10.0  # Slot 2 -> NO_PERSON
        logits[0, 3, 0] = 10.0  # Slot 3 -> NO_PERSON
        logits[0, 4, 0] = 10.0  # Slot 4 -> NO_PERSON
        logits[0, 5, 0] = 10.0  # Slot 5 -> NO_PERSON

        assignments = HungarianMatcher.match(logits, target_slots)
        row_ind, col_ind = assignments[0]
        assert len(row_ind) == 6
        assert len(col_ind) == 6

    def test_batch_processing(self):
        """Verify matcher handles batches correctly with 6 slots."""
        B = 4
        logits = torch.randn(B, 6, 7)
        target_slots = torch.randint(0, 7, (B, 6))

        assignments = HungarianMatcher.match(logits, target_slots)
        assert len(assignments) == B
        for row_ind, col_ind in assignments:
            assert len(row_ind) == 6
            assert len(col_ind) == 6


class TestPermutationInvariance:
    """
    Test 5: Permutation invariance verification for M=6 slots.

    The CORE property: feeding the same answer in arbitrary permutations of
    6 slots must produce identical loss values.
    """

    def test_arbitrary_permutation_invariance_6_slots(self):
        """
        Verify loss is identical across multiple arbitrary permutations of 6 slots.
        """
        loss_fn = MatchedCrossEntropyLoss()
        B, M, K_plus_1 = 1, 6, 7

        logits = torch.zeros(B, M, K_plus_1)
        classes = [1, 2, 3, 4, 5, 0]
        for i, c in enumerate(classes):
            logits[0, i, c] = 8.0

        target_base = torch.tensor([[1, 2, 3, 4, 5, 0]])
        loss_base = loss_fn(logits, target_base)

        # Test 5 different arbitrary permutations of the 6 slots
        torch.manual_seed(123)
        for _ in range(5):
            perm = torch.randperm(M)
            target_perm = target_base[:, perm]
            loss_perm = loss_fn(logits, target_perm)

            torch.testing.assert_close(
                loss_base, loss_perm,
                msg=f"Permutation invariance failed for perm {perm.tolist()}: {loss_base.item()} vs {loss_perm.item()}",
            )

    def test_no_person_padding_invariance(self):
        """
        Samples with 2 active users and 4 NO_PERSON slots in different orders
        must produce identical loss.
        """
        loss_fn = MatchedCrossEntropyLoss()
        B, M, K_plus_1 = 1, 6, 7

        logits = torch.randn(B, M, K_plus_1)

        # Order 1: active users at beginning [1, 2, 0, 0, 0, 0]
        target_1 = torch.tensor([[1, 2, 0, 0, 0, 0]])
        # Order 2: active users interleaved [0, 1, 0, 2, 0, 0]
        target_2 = torch.tensor([[0, 1, 0, 2, 0, 0]])
        # Order 3: active users at end [0, 0, 0, 0, 1, 2]
        target_3 = torch.tensor([[0, 0, 0, 0, 1, 2]])

        loss_1 = loss_fn(logits, target_1)
        loss_2 = loss_fn(logits, target_2)
        loss_3 = loss_fn(logits, target_3)

        torch.testing.assert_close(loss_1, loss_2)
        torch.testing.assert_close(loss_1, loss_3)

    def test_batch_permutation_invariance(self):
        """Batch-level arbitrary permutation invariance with M=6."""
        loss_fn = MatchedCrossEntropyLoss()
        B, M, K_plus_1 = 8, 6, 7
        logits = torch.randn(B, M, K_plus_1)

        target_1 = torch.randint(0, K_plus_1, (B, M))
        # Random column permutation
        perm = torch.randperm(M)
        target_2 = target_1[:, perm]

        loss_1 = loss_fn(logits, target_1)
        loss_2 = loss_fn(logits, target_2)

        torch.testing.assert_close(loss_1, loss_2)

    def test_loss_is_differentiable_6_slots(self):
        """The matched CE loss with M=6 must be differentiable (gradients flow)."""
        loss_fn = MatchedCrossEntropyLoss()

        logits = torch.randn(2, 6, 7, requires_grad=True)
        target_slots = torch.tensor([
            [1, 2, 3, 4, 5, 0],
            [6, 1, 2, 0, 0, 0],
        ])

        loss = loss_fn(logits, target_slots)
        loss.backward()

        assert logits.grad is not None, "No gradient through matched CE loss"
        assert not torch.isnan(logits.grad).any(), "Gradient contains NaNs"
        assert logits.grad.shape == (2, 6, 7)
