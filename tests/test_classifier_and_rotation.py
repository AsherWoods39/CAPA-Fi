"""
CAPA-Fi: Classifier & Rotation Head Unit Tests (Member 2)

Test 2: Classifier [B,256] → [B,6,7] (6*7 = 42 logits)
Test 3: Rotation head [B,256] → [B,2]
"""

import pytest
import torch

from src.models.classifier import MultiSlotLinear
from src.models.rotation_head import BinaryRotationMLP


class TestMultiSlotLinear:
    """Tests for the multi-slot classification head (M=6)."""

    @pytest.fixture
    def classifier(self):
        return MultiSlotLinear(latent_dim=256, num_slots=6, total_classes=7)

    def test_output_shape(self, classifier):
        """Primary contract: [B,256] → [B,6,7]."""
        z = torch.randn(4, 256)
        logits = classifier(z)
        assert logits.shape == (4, 6, 7), f"Expected (4,6,7), got {logits.shape}"
        assert classifier.fc.out_features == 6 * 7, f"Expected 42 logits, got {classifier.fc.out_features}"

    def test_no_softmax(self, classifier):
        """Output should be raw logits, not probabilities (can be negative)."""
        z = torch.randn(4, 256)
        logits = classifier(z)
        # Raw logits can be negative (softmax would make them positive)
        has_negative = (logits < 0).any()
        assert has_negative, "Logits appear to have softmax applied (no negatives)"

    def test_freezable(self, classifier):
        """Classifier must be independently freezable."""
        classifier.requires_grad_(False)
        for param in classifier.parameters():
            assert not param.requires_grad, "Parameter still requires grad after freeze"

    def test_batch_size_1(self, classifier):
        z = torch.randn(1, 256)
        logits = classifier(z)
        assert logits.shape == (1, 6, 7)

    def test_from_config(self):
        config = {"model": {"latent_dim_D": 256, "max_users_M": 6, "total_classes": 7}}
        clf = MultiSlotLinear.from_config(config)
        z = torch.randn(2, 256)
        assert clf(z).shape == (2, 6, 7)


class TestBinaryRotationMLP:
    """Tests for the binary rotation SSL head."""

    @pytest.fixture
    def rot_head(self):
        return BinaryRotationMLP(latent_dim=256, hidden_dim=64)

    def test_output_shape(self, rot_head):
        """Primary contract: [B,256] → [B,2]."""
        z = torch.randn(4, 256)
        rot_logits = rot_head(z)
        assert rot_logits.shape == (4, 2), f"Expected (4,2), got {rot_logits.shape}"

    def test_double_batch_rotation(self, rot_head):
        """Rotation head should handle 2B inputs (original + rotated)."""
        z = torch.randn(8, 256)  # 2B where B=4
        rot_logits = rot_head(z)
        assert rot_logits.shape == (8, 2), f"Expected (8,2), got {rot_logits.shape}"

    def test_gradient_flow(self, rot_head):
        """Gradients should flow through the rotation head."""
        z = torch.randn(4, 256, requires_grad=True)
        rot_logits = rot_head(z)
        loss = rot_logits.sum()
        loss.backward()
        assert z.grad is not None

    def test_from_config(self):
        config = {"model": {"latent_dim_D": 256, "rot_hidden_dim": 64}}
        head = BinaryRotationMLP.from_config(config)
        z = torch.randn(2, 256)
        assert head(z).shape == (2, 2)
