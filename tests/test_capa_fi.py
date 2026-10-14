"""
CAPA-Fi: Consolidated Project Test Suite

Covers all three implemented modules:
    - Module 1 / M2: Data pipeline, dataset contract, and CSI augmentation
    - Module 2: Model architecture — backbone, classifier, rotation head,
                Hungarian matcher, permutation-invariance, and source training
    - Module 3: Adaptation engine — confidence masking, loss engine, and
                TargetAdaptationTrainer / CAPAFiAdaptationEngine

Run from the project root with:
    pytest tests/test_capa_fi.py -v
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# ─── Module imports ───────────────────────────────────────────────────────────
from src.dataset import (
    MultiUserCSIDatasetWithPreprocessing,
    extract_target_slots,
    map_activity_to_class,
    ACTIVITY_TO_CLASS,
)
from src.utils.augmentation import (
    CSIAugmentor,
    jitter,
    scaling,
    slice_shuffle,
    magnitude_warp,
    window_warp,
)
from src.models.backbone import CNN2DBlock, SpatialTemporalBackbone
from src.models.classifier import MultiSlotLinear
from src.models.rotation_head import BinaryRotationMLP
from src.models.har_model import HARModel
from src.models.matcher import HungarianMatcher
from src.models.losses import MatchedCrossEntropyLoss, SourceTrainingLoss
from src.models.train_source import create_rotation_pair
from src.adaptation.confidence_masking import (
    compute_probs,
    compute_sample_confidence,
    compute_occupancy,
    CategoryMaskEngine,
    compute_weights_and_masks,
)
from src.adaptation.loss_engine import (
    CAPAFiLossEngine,
    compute_entropy_loss,
    compute_masked_diversity_loss,
    compute_rotation_loss,
)
from src.adaptation.trainer import TargetAdaptationTrainer, CAPAFiAdaptationEngine
from src.adaptation.mock_inputs import MockHARModel, generate_mock_logits
from src.evaluation.metrics import (
    match_slots_hungarian,
    compute_slot_wise_accuracy,
    compute_occupancy_mae,
    compute_macro_f1,
    compute_exact_match_ratio,
    evaluate_predictions,
)
from src.evaluation.benchmark import (
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_loss_curves,
    export_latex_table,
    run_benchmark,
)


# =============================================================================
# MODULE 1 / 2 — Data Pipeline & Dataset
# =============================================================================

class TestDatasetContract:
    """Verify MultiUserCSIDatasetWithPreprocessing slot contract and NO_PERSON padding."""

    def test_activity_mapping_completeness(self):
        """All canonical activity strings map to non-negative class indices."""
        for name, idx in ACTIVITY_TO_CLASS.items():
            assert 0 <= idx <= 6, f"Class index {idx} out of [0,6] for '{name}'"

    def test_map_activity_to_class_none(self):
        assert map_activity_to_class(None) == 0

    def test_map_activity_to_class_nan(self):
        assert map_activity_to_class(float("nan")) == 0

    def test_map_activity_to_class_unknown(self):
        assert map_activity_to_class("unknown_activity") == 0

    def test_no_person_padding_zero_users(self):
        row = pd.Series({"user_1_activity": None, "user_2_activity": None})
        slots = extract_target_slots(row, num_slots=6)
        assert slots.tolist() == [0, 0, 0, 0, 0, 0]

    def test_no_person_padding_one_user(self):
        row = pd.Series({"user_1_activity": "walk", "user_2_activity": None})
        slots = extract_target_slots(row, num_slots=6)
        assert slots.tolist() == [1, 0, 0, 0, 0, 0]

    def test_no_person_padding_two_users(self):
        row = pd.Series({
            "user_1_activity": "walk",
            "user_2_activity": "sit_down",
            "user_3_activity": None,
        })
        slots = extract_target_slots(row, num_slots=6)
        assert slots.tolist() == [1, 2, 0, 0, 0, 0]

    def test_dataset_real_csv(self):
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
        x, target_slots = dataset[0]
        assert x.shape == (3, 3000, 30)
        assert target_slots.shape == (6,)
        assert target_slots.dtype == torch.long

    def test_dataloader_batch_shapes(self):
        csv_path = "splits/source_split.csv"
        if not os.path.isfile(csv_path):
            pytest.skip("splits/source_split.csv not found")

        dataset = MultiUserCSIDatasetWithPreprocessing(
            annotation_file=csv_path,
            data_dir="data/processed/WiMANS",
            fixed_length=3000,
            num_slots=6,
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_x, batch_targets = next(iter(loader))
        assert batch_x.shape == (4, 3, 3000, 30)
        assert batch_targets.shape == (4, 6)


# =============================================================================
# MODULE 2 — Data Augmentation
# =============================================================================

class TestAugmentation:
    """Shape-preservation and numerical validity for all CSI augmentation ops."""

    @pytest.fixture
    def x(self):
        return torch.randn(4, 3, 3000, 30)

    def test_jitter_shape(self, x):
        assert jitter(x).shape == x.shape

    def test_scaling_shape(self, x):
        assert scaling(x).shape == x.shape

    def test_slice_shuffle_shape(self, x):
        assert slice_shuffle(x).shape == x.shape

    def test_magnitude_warp_shape(self, x):
        assert magnitude_warp(x).shape == x.shape

    def test_window_warp_shape(self, x):
        assert window_warp(x).shape == x.shape

    def test_augmentor_shape(self, x):
        aug = CSIAugmentor(augment_prob=1.0)
        assert aug(x).shape == x.shape

    def test_augmentor_no_nans(self, x):
        aug = CSIAugmentor(augment_prob=1.0)
        out = aug(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_augmentor_skip_probability(self, x):
        """With augment_prob=0, output must be bitwise identical to input."""
        aug = CSIAugmentor(augment_prob=0.0)
        torch.testing.assert_close(aug(x), x)


# =============================================================================
# MODULE 2 — Backbone
# =============================================================================

class TestCNN2DBlock:
    """Individual CNN2D block shape and numerical validity."""

    def test_output_shape(self):
        block = CNN2DBlock(in_channels=3, out_channels=32)
        x = torch.randn(2, 3, 100, 30)
        out = block(x)
        assert out.shape == (2, 32, 50, 15)

    def test_no_nans(self):
        block = CNN2DBlock(in_channels=3, out_channels=32)
        out = block(torch.randn(2, 3, 100, 30))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestSpatialTemporalBackbone:
    """Full backbone: CNN stack + BiLSTM encoder."""

    @pytest.fixture
    def backbone(self):
        return SpatialTemporalBackbone(
            in_channels=3,
            spatial_channels=[32, 64, 128],
            temporal_hidden_dim=128,
            temporal_layers=2,
            dropout=0.0,
        )

    def test_output_shape(self, backbone):
        z = backbone(torch.randn(2, 3, 3000, 30))
        assert z.shape == (2, 256)

    def test_intermediate_cnn_shapes(self, backbone):
        x = torch.randn(2, 3, 3000, 30)
        expected = [(2, 32, 1500, 15), (2, 64, 750, 7), (2, 128, 375, 3)]
        for i, block in enumerate(backbone.cnn_blocks):
            x = block(x)
            assert x.shape == expected[i], f"CNN Block {i+1}: {x.shape} ≠ {expected[i]}"

    def test_no_nans(self, backbone):
        z = backbone(torch.randn(2, 3, 3000, 30))
        assert not torch.isnan(z).any()

    def test_backward_no_nans(self, backbone):
        x = torch.randn(2, 3, 3000, 30, requires_grad=True)
        backbone(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_batch_size_invariance(self, backbone):
        for bs in [1, 4, 8]:
            z = backbone(torch.randn(bs, 3, 3000, 30))
            assert z.shape == (bs, 256)

    def test_from_config(self):
        cfg = {
            "model": {"spatial_channels": [32, 64, 128], "temporal_hidden_dim": 128,
                      "temporal_layers": 2, "dropout": 0.2},
            "data": {"channels_C": 3},
        }
        b = SpatialTemporalBackbone.from_config(cfg)
        assert b(torch.randn(2, 3, 3000, 30)).shape == (2, 256)


# =============================================================================
# MODULE 2 — Classifier & Rotation Head
# =============================================================================

class TestMultiSlotLinear:
    """Multi-slot classification head shape, raw-logit, and freeze contracts."""

    @pytest.fixture
    def clf(self):
        return MultiSlotLinear(latent_dim=256, num_slots=6, total_classes=7)

    def test_output_shape(self, clf):
        assert clf(torch.randn(4, 256)).shape == (4, 6, 7)

    def test_raw_logits(self, clf):
        """Output must be raw (can be negative) — no softmax applied."""
        assert (clf(torch.randn(4, 256)) < 0).any()

    def test_freezable(self, clf):
        clf.requires_grad_(False)
        assert all(not p.requires_grad for p in clf.parameters())

    def test_batch_size_1(self, clf):
        assert clf(torch.randn(1, 256)).shape == (1, 6, 7)

    def test_from_config(self):
        cfg = {"model": {"latent_dim_D": 256, "max_users_M": 6, "total_classes": 7}}
        clf = MultiSlotLinear.from_config(cfg)
        assert clf(torch.randn(2, 256)).shape == (2, 6, 7)


class TestBinaryRotationMLP:
    """Rotation SSL head shape and gradient contracts."""

    @pytest.fixture
    def head(self):
        return BinaryRotationMLP(latent_dim=256, hidden_dim=64)

    def test_output_shape(self, head):
        assert head(torch.randn(4, 256)).shape == (4, 2)

    def test_double_batch(self, head):
        assert head(torch.randn(8, 256)).shape == (8, 2)

    def test_gradient_flow(self, head):
        z = torch.randn(4, 256, requires_grad=True)
        head(z).sum().backward()
        assert z.grad is not None

    def test_from_config(self):
        cfg = {"model": {"latent_dim_D": 256, "rot_hidden_dim": 64}}
        head = BinaryRotationMLP.from_config(cfg)
        assert head(torch.randn(2, 256)).shape == (2, 2)


# =============================================================================
# MODULE 2 — Hungarian Matcher & Permutation Invariance
# =============================================================================

class TestHungarianMatcher:
    """Bipartite matching correctness for M=6 slot configurations."""

    def test_identity_assignment(self):
        """Slot i perfectly predicts class[i] → identity assignment."""
        B, M, K1 = 1, 6, 7
        classes = [1, 2, 3, 4, 5, 0]
        logits = torch.zeros(B, M, K1)
        for i, c in enumerate(classes):
            logits[0, i, c] = 10.0

        row_ind, col_ind = HungarianMatcher.match(logits, torch.tensor([classes]))[0]
        assert list(row_ind) == [0, 1, 2, 3, 4, 5]
        assert list(col_ind) == [0, 1, 2, 3, 4, 5]

    def test_swapped_assignment_correctness(self):
        """Shifted predictions → matcher recovers per-class correspondence."""
        target = [1, 2, 3, 4, 5, 0]
        shifted = [0, 1, 2, 3, 4, 5]
        logits = torch.zeros(1, 6, 7)
        for i, c in enumerate(shifted):
            logits[0, i, c] = 10.0

        row_ind, col_ind = HungarianMatcher.match(logits, torch.tensor([target]))[0]
        for r, c in zip(row_ind, col_ind):
            assert shifted[r] == target[c]

    def test_no_gradient_through_matcher(self):
        logits = torch.randn(2, 6, 7, requires_grad=True)
        targets = torch.randint(0, 7, (2, 6))
        assignments = HungarianMatcher.match(logits, targets)
        assert len(assignments) == 2

    def test_no_person_padding_slots(self):
        logits = torch.zeros(1, 6, 7)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2:, 0] = 10.0
        row_ind, col_ind = HungarianMatcher.match(logits, torch.tensor([[1, 2, 0, 0, 0, 0]]))[0]
        assert len(row_ind) == 6

    def test_batch_processing(self):
        B = 4
        assignments = HungarianMatcher.match(torch.randn(B, 6, 7), torch.randint(0, 7, (B, 6)))
        assert all(len(r) == 6 for r, _ in assignments)


class TestPermutationInvariance:
    """Core CAPA-Fi property: identical loss under arbitrary slot permutations."""

    @pytest.fixture
    def loss_fn(self):
        return MatchedCrossEntropyLoss()

    def test_arbitrary_permutation_6_slots(self, loss_fn):
        logits = torch.zeros(1, 6, 7)
        classes = [1, 2, 3, 4, 5, 0]
        for i, c in enumerate(classes):
            logits[0, i, c] = 8.0
        target_base = torch.tensor([classes])
        loss_base = loss_fn(logits, target_base)

        torch.manual_seed(123)
        for _ in range(5):
            perm = torch.randperm(6)
            loss_perm = loss_fn(logits, target_base[:, perm])
            torch.testing.assert_close(loss_base, loss_perm)

    def test_no_person_slot_order_invariance(self, loss_fn):
        logits = torch.randn(1, 6, 7)
        t1 = torch.tensor([[1, 2, 0, 0, 0, 0]])
        t2 = torch.tensor([[0, 1, 0, 2, 0, 0]])
        t3 = torch.tensor([[0, 0, 0, 0, 1, 2]])
        loss1 = loss_fn(logits, t1)
        torch.testing.assert_close(loss1, loss_fn(logits, t2))
        torch.testing.assert_close(loss1, loss_fn(logits, t3))

    def test_batch_permutation_invariance(self, loss_fn):
        B, M = 8, 6
        logits = torch.randn(B, M, 7)
        target_1 = torch.randint(0, 7, (B, M))
        perm = torch.randperm(M)
        loss1 = loss_fn(logits, target_1)
        loss2 = loss_fn(logits, target_1[:, perm])
        torch.testing.assert_close(loss1, loss2)

    def test_loss_is_differentiable(self, loss_fn):
        logits = torch.randn(2, 6, 7, requires_grad=True)
        loss = loss_fn(logits, torch.tensor([[1, 2, 3, 4, 5, 0], [6, 1, 2, 0, 0, 0]]))
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


# =============================================================================
# MODULE 2 — HARModel (end-to-end)
# =============================================================================

class TestHARModel:
    """Full model: API contract, forward pass, freeze controls, and checkpoint I/O."""

    @pytest.fixture
    def model(self):
        return HARModel(
            in_channels=3, spatial_channels=[32, 64, 128],
            temporal_hidden_dim=128, temporal_layers=2,
            latent_dim=256, num_slots=6, total_classes=7,
            rot_hidden_dim=64, dropout=0.0,
        )

    def test_api_attributes(self, model):
        for attr in ["backbone", "classifier", "rotation_head", "slot_head", "rot_head"]:
            assert hasattr(model, attr)
        assert model.slot_head is model.classifier
        assert model.rot_head is model.rotation_head

    def test_forward_shapes(self, model):
        x = torch.randn(2, 3, 3000, 30)
        logits, z = model(x)
        assert logits.shape == (2, 6, 7)
        assert z.shape == (2, 256)

    def test_forward_rotation_shape(self, model):
        z = torch.randn(2, 256)
        assert model.forward_rotation(z).shape == (2, 2)

    def test_freeze_classifier(self, model):
        model.freeze_classifier()
        assert all(not p.requires_grad for p in model.classifier.parameters())
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_unfreeze_backbone(self, model):
        for p in model.parameters():
            p.requires_grad = False
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.rotation_head.parameters())

    def test_checkpoint_roundtrip(self, model, tmp_path):
        path = str(tmp_path / "checkpoint.pt")
        model.save_checkpoint(path, epoch=5, config={"test": True})
        m2 = HARModel(in_channels=3, spatial_channels=[32, 64, 128],
                      temporal_hidden_dim=128, temporal_layers=2, latent_dim=256,
                      num_slots=6, total_classes=7, rot_hidden_dim=64, dropout=0.0)
        ckpt = m2.load_checkpoint(path)
        assert ckpt["epoch"] == 5
        x = torch.randn(1, 3, 3000, 30)
        model.eval(); m2.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x)[0], m2(x)[0])

    def test_full_source_training_pipeline(self, model):
        """E2E: augment → rotation pair → forward → matched CE + rotation loss."""
        B = 2
        x = torch.randn(B, 3, 3000, 30)
        target_slots = torch.randint(0, 7, (B, 6))
        x_aug = CSIAugmentor(augment_prob=1.0)(x)
        x_combined, rot_labels, split_B = create_rotation_pair(x_aug)
        logits_comb, z_comb = model(x_combined)
        logits = logits_comb[:B]
        rot_logits = model.forward_rotation(z_comb)
        loss_fn = SourceTrainingLoss(lambda_rot_source=0.5)
        losses = loss_fn(logits, target_slots, rot_logits, rot_labels)
        assert "total" in losses and not torch.isnan(losses["total"])

    def test_tiny_overfit(self):
        """Model should memorise 2 samples within 35 gradient steps."""
        torch.manual_seed(42)
        model = HARModel(
            in_channels=3, spatial_channels=[32, 64, 128],
            temporal_hidden_dim=64, temporal_layers=1, latent_dim=128,
            num_slots=6, total_classes=7, rot_hidden_dim=32, dropout=0.0,
        )
        x_data = torch.randn(2, 3, 3000, 30)
        t_data = torch.randint(0, 7, (2, 6))
        loss_fn = MatchedCrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        model.train()
        initial = None
        for ep in range(35):
            logits, _ = model(x_data)
            loss = loss_fn(logits, t_data)
            if ep == 0:
                initial = loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < initial * 0.25, \
            f"Tiny overfit failed: {initial:.4f} → {loss.item():.4f}"


# =============================================================================
# MODULE 3 — Confidence Masking
# =============================================================================

class TestConfidenceMasking:
    """compute_probs, compute_sample_confidence, compute_occupancy, CategoryMaskEngine."""

    @pytest.fixture
    def logits(self):
        return torch.randn(32, 2, 7)

    def test_compute_probs_shape(self, logits):
        p = compute_probs(logits)
        assert p.shape == (32, 2, 7)

    def test_compute_probs_axioms(self, logits):
        p = compute_probs(logits)
        assert p.min() >= 0.0 and p.max() <= 1.0
        torch.testing.assert_close(p.sum(dim=-1), torch.ones(32, 2), atol=1e-5, rtol=1e-5)

    def test_sample_confidence_shape_and_range(self, logits):
        probs = compute_probs(logits)
        w = compute_sample_confidence(probs)
        assert w.shape == (32,)
        assert w.min() >= 0.0 and w.max() <= 1.0

    def test_occupancy_shape_and_range(self, logits):
        p_occ = compute_occupancy(compute_probs(logits))
        assert p_occ.shape == (32,)
        assert p_occ.min() >= 0.0 and p_occ.max() <= 1.0

    def test_category_mask_engine_shape(self, logits):
        probs = compute_probs(logits)
        engine = CategoryMaskEngine(num_classes=6)
        w = compute_sample_confidence(probs)
        gamma = engine.update_and_compute(probs, w)
        assert gamma.shape == (6,)
        assert gamma.min() >= 0.0 and gamma.max() <= 1.0

    def test_absent_class_suppression(self):
        """After EMA convergence, absent classes must have gamma < 0.1."""
        active = [0, 1, 2, 3]  # class 4, 5, 6 physically absent
        engine = CategoryMaskEngine(num_classes=6, beta=0.5, tau_thresh=0.05, tau_temp=0.01)
        for _ in range(50):
            logits = generate_mock_logits(32, 2, 7, active_classes=active)
            probs = compute_probs(logits)
            w = compute_sample_confidence(probs)
            gamma = engine.update_and_compute(probs, w)
        assert gamma[3] < 0.1, f"gamma[3]={gamma[3]:.4f} should be < 0.1"
        assert gamma[4] < 0.1
        assert gamma[5] < 0.1

    def test_compute_weights_and_masks_output(self, logits):
        engine = CategoryMaskEngine(num_classes=6)
        w, gamma, p_occ = compute_weights_and_masks(logits, engine)
        assert w.shape == (32,)
        assert gamma.shape == (6,)
        assert p_occ.shape == (32,)


# =============================================================================
# MODULE 3 — Loss Engine
# =============================================================================

class TestCAPAFiLossEngine:
    """CAPAFiLossEngine: correct output keys, gradient flow, and loss components."""

    @pytest.fixture
    def engine(self):
        return CAPAFiLossEngine(lambda_ent=1.0, lambda_div=0.5, lambda_rot=0.5)

    def test_output_keys(self, engine):
        logits = torch.randn(8, 2, 7, requires_grad=True)
        r0 = torch.randn(8, 2, requires_grad=True)
        r180 = torch.randn(8, 2, requires_grad=True)
        w = torch.rand(8)
        gamma = torch.ones(6)
        p_occ = torch.rand(8)
        out = engine(logits, r0, r180, w, gamma, p_occ)
        for key in ["loss_total", "loss_entropy", "loss_diversity",
                    "loss_rotation", "mean_confidence", "active_classes_count",
                    "occupancy_factor"]:
            assert key in out

    def test_gradient_flow(self, engine):
        logits = torch.randn(8, 2, 7, requires_grad=True)
        r0 = torch.randn(8, 2, requires_grad=True)
        r180 = torch.randn(8, 2, requires_grad=True)
        w, gamma, p_occ = torch.rand(8), torch.ones(6), torch.rand(8)
        out = engine(logits, r0, r180, w, gamma, p_occ)
        assert out["loss_total"].requires_grad
        out["loss_total"].backward()
        assert logits.grad is not None
        assert r0.grad is not None
        assert r180.grad is not None

    def test_rotation_loss_labels(self):
        """Rotation loss: original=0, rotated=1 label convention."""
        B = 4
        r0 = torch.zeros(B, 2); r0[:, 0] = 10.0   # strongly predicts class 0
        r180 = torch.zeros(B, 2); r180[:, 1] = 10.0  # strongly predicts class 1
        loss = compute_rotation_loss(r0, r180)
        assert loss.item() < 0.1, f"Rotation loss too high: {loss.item():.4f}"


# =============================================================================
# MODULE 3 — TargetAdaptationTrainer
# =============================================================================

class TestTargetAdaptationTrainer:
    """Trainer: parameter isolation, freezing guarantee, backbone update, checkpointing."""

    ADAPT_CONFIG = {
        "learning_rate_backbone": 1e-4,
        "weight_decay": 1e-4,
        "lambda_ent": 1.0,
        "lambda_div": 0.5,
        "lambda_rot": 0.5,
        "num_classes": 6,
        "ema_decay_beta": 0.90,
        "tau_thresh": 0.05,
        "tau_temp": 0.04,
        "tau_conf": 0.65,
    }

    @pytest.fixture
    def model_and_trainer(self):
        model = MockHARModel(channels=3, latent_dim=256, slots=2, total_classes=7)
        trainer = TargetAdaptationTrainer(model=model, config=self.ADAPT_CONFIG)
        return model, trainer

    def test_parameter_isolation(self, model_and_trainer):
        model, _ = model_and_trainer
        assert all(not p.requires_grad for p in model.slot_head.parameters())
        assert all(p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.rot_head.parameters())

    def test_classifier_frozen_after_epoch(self, model_and_trainer, tmp_path):
        model, trainer = model_and_trainer
        before = {n: p.clone().detach() for n, p in model.slot_head.named_parameters()}
        loader = DataLoader(TensorDataset(torch.randn(8, 3, 500, 30)), batch_size=4)
        avg_loss = trainer.adapt_epoch(loader)
        assert isinstance(avg_loss, float)
        assert trainer.verify_classifier_frozen(before)

    def test_backbone_updated_after_epoch(self, model_and_trainer):
        model, trainer = model_and_trainer
        w_before = next(model.backbone.parameters()).clone().detach()
        loader = DataLoader(TensorDataset(torch.randn(8, 3, 500, 30)), batch_size=4)
        trainer.adapt_epoch(loader)
        w_after = next(model.backbone.parameters()).detach()
        assert not torch.equal(w_before, w_after), "Backbone weights did not update"

    def test_checkpoint_persisted(self, model_and_trainer, tmp_path):
        model, trainer = model_and_trainer
        loader = DataLoader(TensorDataset(torch.randn(8, 3, 500, 30)), batch_size=4)
        trainer.adapt_epoch(loader)
        path = str(tmp_path / "adapted.pt")
        trainer.save_checkpoint(path)
        assert Path(path).exists()

    def test_multi_epoch_adapt_history(self, tmp_path):
        model = MockHARModel(channels=3, latent_dim=256, slots=2, total_classes=7)
        cfg = {**self.ADAPT_CONFIG, "epochs": 3}
        engine = CAPAFiAdaptationEngine(model=model, config=cfg)

        loader = DataLoader(TensorDataset(torch.randn(8, 3, 500, 30)), batch_size=4)
        ckpt = str(tmp_path / "multi.pt")
        result_json = str(tmp_path / "history.json")

        history = engine.adapt(
            target_loader=loader, epochs=3,
            save_path=ckpt, results_path=result_json,
            verbose=False,
        )
        assert len(history["epoch"]) == 3
        assert len(history["loss_total"]) == 3
        assert Path(result_json).exists()
        assert Path(ckpt).exists()

    def test_contract_methods(self):
        model = MockHARModel(channels=3, latent_dim=256, slots=2, total_classes=7)
        engine = CAPAFiAdaptationEngine(model=model, config=self.ADAPT_CONFIG)
        assert engine.num_classes == 6
        probs = torch.softmax(torch.randn(8, 2, 7), dim=-1)
        w = engine.compute_sample_confidence(probs)
        gamma = engine.compute_category_mask(probs, w)
        assert w.shape == (8,)
        assert gamma.shape == (6,)


# =============================================================================
# MODULE 4 — Evaluation & Benchmarking
# =============================================================================

class TestEvaluationMetrics:
    """Tests for Hungarian slot matching, slot accuracy, occupancy MAE, F1, exact match."""

    def test_slot_wise_accuracy_perfect(self):
        preds = torch.tensor([[1, 2, 0], [3, 0, 0]])
        targets = torch.tensor([[1, 2, 0], [3, 0, 0]])
        assert compute_slot_wise_accuracy(preds, targets) == 1.0

    def test_slot_wise_accuracy_imperfect(self):
        preds = torch.tensor([[1, 2, 0], [3, 0, 0]])
        targets = torch.tensor([[1, 3, 0], [3, 0, 0]])  # 5 of 6 correct
        assert pytest.approx(compute_slot_wise_accuracy(preds, targets), 0.01) == 5 / 6

    def test_slot_wise_accuracy_active_only(self):
        # Slots with target=0 ignored
        preds = torch.tensor([[1, 5, 0]])
        targets = torch.tensor([[1, 0, 0]])  # only slot 0 is active (target != 0)
        # Active accuracy checks only slot 0: pred=1, target=1 -> 1.0
        assert compute_slot_wise_accuracy(preds, targets, ignore_empty=True) == 1.0

    def test_occupancy_mae(self):
        # Sample 1: true=2 users, pred=2 users (err 0)
        # Sample 2: true=1 user, pred=3 users (err 2)
        # Mean MAE = (0 + 2) / 2 = 1.0
        preds = torch.tensor([[1, 2, 0], [1, 2, 3]])
        targets = torch.tensor([[1, 3, 0], [1, 0, 0]])
        assert compute_occupancy_mae(preds, targets) == 1.0

    def test_macro_f1_bounds(self):
        preds = torch.randint(0, 7, (10, 6))
        targets = torch.randint(0, 7, (10, 6))
        f1 = compute_macro_f1(preds, targets, num_classes=7)
        assert 0.0 <= f1 <= 1.0

    def test_exact_match_ratio(self):
        preds = torch.tensor([[1, 2], [3, 4]])
        targets = torch.tensor([[1, 2], [3, 5]])  # 1 of 2 exact match
        assert compute_exact_match_ratio(preds, targets) == 0.5

    def test_hungarian_matching_invariance(self):
        # Target: slot 0 is class 1, slot 1 is class 2
        targets = torch.tensor([[1, 2]])
        # Logits predicting slot 0 as class 2, slot 1 as class 1 (swapped order)
        logits = torch.zeros((1, 2, 7))
        logits[0, 0, 2] = 10.0  # slot 0 -> class 2
        logits[0, 1, 1] = 10.0  # slot 1 -> class 1

        matched_preds, _ = match_slots_hungarian(logits, targets)
        # Hungarian matching should align predictions to targets: slot 0->1, slot 1->2
        assert matched_preds[0, 0].item() == 1
        assert matched_preds[0, 1].item() == 2

    def test_evaluate_predictions_dict_keys(self):
        logits = torch.randn(4, 6, 7)
        targets = torch.randint(0, 7, (4, 6))
        metrics = evaluate_predictions(logits, targets)
        expected_keys = [
            "slot_wise_accuracy",
            "active_slot_accuracy",
            "occupancy_mae",
            "macro_f1",
            "macro_f1_active",
            "exact_match_ratio",
        ]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)


class TestEvaluationBenchmark:
    """Tests for benchmarking execution, visualizations, and LaTeX table generation."""

    def test_latex_table_export(self, tmp_path):
        dummy_results = {
            "Baseline": {"slot_wise_accuracy": 0.5, "active_slot_accuracy": 0.4, "macro_f1": 0.45, "occupancy_mae": 1.2},
            "CAPA-Fi": {"slot_wise_accuracy": 0.85, "active_slot_accuracy": 0.80, "macro_f1": 0.82, "occupancy_mae": 0.3},
        }
        out_file = str(tmp_path / "table.tex")
        content = export_latex_table(dummy_results, save_path=out_file)
        assert Path(out_file).exists()
        assert r"\begin{tabular}" in content
        assert "CAPA-Fi" in content

    def test_run_benchmark_smoke(self, tmp_path):
        # Smoke test running benchmark under partial_75 with small sample size
        metrics = run_benchmark(protocol="partial_75", num_eval_samples=8)
        assert "slot_wise_accuracy" in metrics
        assert "occupancy_mae" in metrics
        assert Path("results/benchmark_partial_75.json").exists()
        assert Path("results/confusion_matrix_partial_75.png").exists()

