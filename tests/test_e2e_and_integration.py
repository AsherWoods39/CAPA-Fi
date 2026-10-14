"""
CAPA-Fi: End-to-End & Integration Tests (Member 2)

Test 6: Augmentation shape preservation
Test 7: End-to-end forward pass (M=6 slots, no crashes)
Test 8: Tiny overfit test (M=6 slots)
Test 9: Mock replacement test (HARModel M=6)
Test 10: Source dataset contract ([B,3,3000,30] and [B,6]) & NO_PERSON padding
"""

import os
import pytest
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (
    MultiUserCSIDatasetWithPreprocessing,
    extract_target_slots,
    map_activity_to_class,
    ACTIVITY_TO_CLASS,
)
from src.models.har_model import HARModel
from src.models.losses import SourceTrainingLoss, MatchedCrossEntropyLoss
from src.models.train_source import create_rotation_pair
from src.utils.augmentation import (
    CSIAugmentor,
    jitter,
    scaling,
    slice_shuffle,
    magnitude_warp,
    window_warp,
)


class TestAugmentation:
    """Test 6: Augmentation shape preservation and validity."""

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 3, 3000, 30)

    def test_jitter_shape(self, sample_input):
        out = jitter(sample_input)
        assert out.shape == sample_input.shape

    def test_scaling_shape(self, sample_input):
        out = scaling(sample_input)
        assert out.shape == sample_input.shape

    def test_slice_shuffle_shape(self, sample_input):
        out = slice_shuffle(sample_input)
        assert out.shape == sample_input.shape

    def test_magnitude_warp_shape(self, sample_input):
        out = magnitude_warp(sample_input)
        assert out.shape == sample_input.shape

    def test_window_warp_shape(self, sample_input):
        out = window_warp(sample_input)
        assert out.shape == sample_input.shape

    def test_augmentor_shape(self, sample_input):
        augmentor = CSIAugmentor(augment_prob=1.0)
        out = augmentor(sample_input)
        assert out.shape == sample_input.shape

    def test_no_nans(self, sample_input):
        augmentor = CSIAugmentor(augment_prob=1.0)
        out = augmentor(sample_input)
        assert not torch.isnan(out).any(), "Augmented output contains NaNs"
        assert not torch.isinf(out).any(), "Augmented output contains Infs"

    def test_augmentor_skip(self, sample_input):
        """With augment_prob=0, output should be identical to input."""
        augmentor = CSIAugmentor(augment_prob=0.0)
        out = augmentor(sample_input)
        torch.testing.assert_close(out, sample_input)


class TestEndToEndForward:
    """Test 7: Full pipeline forward pass with M=6 slots, no crashes."""

    @pytest.fixture
    def model(self):
        return HARModel(
            in_channels=3,
            spatial_channels=[32, 64, 128],
            temporal_hidden_dim=128,
            temporal_layers=2,
            latent_dim=256,
            num_slots=6,
            total_classes=7,
            rot_hidden_dim=64,
            dropout=0.0,
        )

    def test_full_pipeline(self, model):
        """CSI → augment → backbone → classifier/rotation → match → loss. No crashes."""
        B = 2
        x = torch.randn(B, 3, 3000, 30)
        target_slots = torch.randint(0, 7, (B, 6))

        # Augment
        augmentor = CSIAugmentor(augment_prob=1.0)
        x_aug = augmentor(x)
        assert x_aug.shape == (B, 3, 3000, 30)

        # Create rotation pair
        x_combined, rot_labels, split_B = create_rotation_pair(x_aug)
        assert x_combined.shape == (2 * B, 3, 3000, 30)
        assert rot_labels.shape == (2 * B,)

        # Forward
        logits_combined, z_combined = model(x_combined)
        assert logits_combined.shape == (2 * B, 6, 7)
        assert z_combined.shape == (2 * B, 256)

        # Split for classification (use original only)
        logits = logits_combined[:B]

        # Rotation head
        rot_logits = model.forward_rotation(z_combined)
        assert rot_logits.shape == (2 * B, 2)

        # Loss
        loss_fn = SourceTrainingLoss(lambda_rot_source=0.5)
        losses = loss_fn(logits, target_slots, rot_logits, rot_labels)

        assert "total" in losses
        assert "matched_ce" in losses
        assert "rotation" in losses
        assert not torch.isnan(losses["total"]), "Total loss is NaN"

    def test_checkpoint_save_load(self, model, tmp_path):
        """Checkpoint save/load round-trip with M=6."""
        path = str(tmp_path / "test_checkpoint.pt")

        # Save
        model.save_checkpoint(path, epoch=5, config={"test": True})

        # Load into a new model
        model2 = HARModel(
            in_channels=3,
            spatial_channels=[32, 64, 128],
            temporal_hidden_dim=128,
            temporal_layers=2,
            latent_dim=256,
            num_slots=6,
            total_classes=7,
            rot_hidden_dim=64,
            dropout=0.0,
        )

        checkpoint = model2.load_checkpoint(path)
        assert checkpoint["epoch"] == 5
        assert checkpoint["config"] == {"test": True}

        # Verify weights match
        x = torch.randn(1, 3, 3000, 30)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1, z1 = model(x)
            out2, z2 = model2(x)
        torch.testing.assert_close(out1, out2)
        torch.testing.assert_close(z1, z2)


class TestTinyOverfit:
    """Test 8: Tiny overfit — 5 samples, loss → ~0 within 200 epochs (M=6)."""

    def test_overfit_small_dataset(self):
        """Model should memorize 5 samples with 6 slots."""
        torch.manual_seed(42)

        model = HARModel(
            in_channels=3,
            spatial_channels=[32, 64, 128],
            temporal_hidden_dim=128,
            temporal_layers=2,
            latent_dim=256,
            num_slots=6,
            total_classes=7,
            rot_hidden_dim=64,
            dropout=0.0,
        )

        N = 5
        x_data = torch.randn(N, 3, 3000, 30)
        # 6 slots with mixtures of active activities and NO_PERSON
        target_data = torch.randint(0, 7, (N, 6))

        loss_fn = MatchedCrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        initial_loss = None
        final_loss = None

        for epoch in range(200):
            logits, z = model(x_data)
            loss = loss_fn(logits, target_data)

            if epoch == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1, (
            f"Tiny overfit failed: initial={initial_loss:.4f}, final={final_loss:.4f}. "
            f"Expected final < {initial_loss * 0.1:.4f}"
        )


class TestMockReplacement:
    """Test 9: HARModel is a drop-in replacement for MockHARModel (M=6)."""

    def test_api_compatibility(self):
        """Verify HARModel exposes the same interface as MockHARModel."""
        model = HARModel()

        # Check attributes exist
        assert hasattr(model, "backbone"), "Missing 'backbone' attribute"
        assert hasattr(model, "classifier"), "Missing 'classifier' attribute"
        assert hasattr(model, "rotation_head"), "Missing 'rotation_head' attribute"
        assert hasattr(model, "slot_head"), "Missing 'slot_head' alias"
        assert hasattr(model, "rot_head"), "Missing 'rot_head' alias"

        # Check aliases point to the right objects
        assert model.slot_head is model.classifier
        assert model.rot_head is model.rotation_head

        # Check methods exist
        assert callable(getattr(model, "forward", None))
        assert callable(getattr(model, "forward_rotation", None))
        assert callable(getattr(model, "freeze_classifier", None))
        assert callable(getattr(model, "unfreeze_backbone", None))

    def test_forward_signature(self):
        """forward(x) → (logits [B, 6, 7], z [B, 256])."""
        model = HARModel()
        x = torch.randn(2, 3, 3000, 30)

        result = model(x)
        assert isinstance(result, tuple), "forward() should return a tuple"
        assert len(result) == 2, "forward() should return (logits, z)"

        logits, z = result
        assert logits.shape == (2, 6, 7), f"logits shape: {logits.shape}"
        assert z.shape == (2, 256), f"z shape: {z.shape}"

    def test_forward_rotation_signature(self):
        """forward_rotation(z) → rot_logits [B, 2]."""
        model = HARModel()
        z = torch.randn(2, 256)
        rot_logits = model.forward_rotation(z)
        assert rot_logits.shape == (2, 2)

    def test_freeze_classifier(self):
        """freeze_classifier() sets requires_grad=False on classifier."""
        model = HARModel()
        model.freeze_classifier()

        for param in model.classifier.parameters():
            assert not param.requires_grad

        # Backbone should still be trainable
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self):
        """unfreeze_backbone() sets requires_grad=True on backbone and rotation head."""
        model = HARModel()

        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Then unfreeze backbone
        model.unfreeze_backbone()

        for param in model.backbone.parameters():
            assert param.requires_grad
        for param in model.rotation_head.parameters():
            assert param.requires_grad


class TestDatasetContract:
    """Test 10: MultiUserCSIDatasetWithPreprocessing and NO_PERSON padding contract."""

    def test_no_person_padding_examples(self):
        """
        Verify the required examples:
          0 users → [0, 0, 0, 0, 0, 0]
          1 user  → [act, 0, 0, 0, 0, 0]
          2 users → [act1, act2, 0, 0, 0, 0]
        """
        # 0 users
        row_0 = pd.Series({"user_1_activity": None, "user_2_activity": None})
        slots_0 = extract_target_slots(row_0, num_slots=6)
        assert slots_0.tolist() == [0, 0, 0, 0, 0, 0]

        # 1 user (Walk = 1)
        row_1 = pd.Series({"user_1_activity": "walk", "user_2_activity": None})
        slots_1 = extract_target_slots(row_1, num_slots=6)
        assert slots_1.tolist() == [1, 0, 0, 0, 0, 0]

        # 2 users (Walk = 1, Sit = 2)
        row_2 = pd.Series({
            "user_1_activity": "walk",
            "user_2_activity": "sit_down",
            "user_3_activity": None,
        })
        slots_2 = extract_target_slots(row_2, num_slots=6)
        assert slots_2.tolist() == [1, 2, 0, 0, 0, 0]

    def test_dataset_loader_shapes(self):
        """Test dataset loading with source_split.csv and DataLoader collation."""
        csv_path = "splits/source_split.csv"
        if not os.path.isfile(csv_path):
            pytest.skip("splits/source_split.csv not found")

        dataset = MultiUserCSIDatasetWithPreprocessing(
            annotation_file=csv_path,
            data_dir="data/processed/WiMANS",
            fixed_length=3000,
            num_slots=6,
            in_channels=3,
        )

        assert len(dataset) > 0

        # Test single item
        x, target_slots = dataset[0]
        assert x.shape == (3, 3000, 30), f"Sample shape: {x.shape}"
        assert target_slots.shape == (6,), f"Slot shape: {target_slots.shape}"
        assert target_slots.dtype == torch.long

        # Test DataLoader batch
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_x, batch_targets = next(iter(loader))
        assert batch_x.shape == (4, 3, 3000, 30), f"Batch shape: {batch_x.shape}"
        assert batch_targets.shape == (4, 6), f"Batch targets: {batch_targets.shape}"
