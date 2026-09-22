"""
CAPA-Fi: Module 3 Standalone Unit-Testing Harness
Location: src/adaptation/test_standalone.py
Member 3: Source-Free Adaptation & SSL Engine (Lead: Athishta P. A.)

Tests all three core components:
1. confidence_masking.py: w_i, gamma_k, p_occ, and CategoryMaskEngine
2. loss_engine.py: L_ent, L_div-masked, L_rot, and CAPAFiLossEngine
3. trainer.py: TargetAdaptationTrainer, optimizer setup, classifier freezing guarantee
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.adaptation.trainer import TargetAdaptationTrainer
from src.adaptation.mock_inputs import (
    generate_mock_adaptation_batch,
    generate_mock_logits,
    MockHARModel,
)


def test_confidence_masking():
    """Test that w_i, gamma_k, p_occ have correct shapes and value ranges."""
    print("--- 1. Testing confidence_masking.py ---")
    logits = torch.randn(32, 2, 7)   # (B, M, K+1)
    probs = compute_probs(logits)
    
    # Shape checks & probability axioms
    assert probs.shape == (32, 2, 7), f"Expected (32, 2, 7), got {probs.shape}"
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(32, 2), atol=1e-5, rtol=1e-5)
    
    # Sample confidence checks
    w_i = compute_sample_confidence(probs)
    assert w_i.shape == (32,), f"Expected w_i shape (32,), got {w_i.shape}"
    assert w_i.min() >= 0.0 and w_i.max() <= 1.0, f"w_i values out of range: [{w_i.min()}, {w_i.max()}]"
    
    # Occupancy checks
    p_occ = compute_occupancy(probs)
    assert p_occ.shape == (32,), f"Expected p_occ shape (32,), got {p_occ.shape}"
    assert p_occ.min() >= 0.0 and p_occ.max() <= 1.0
    
    # CategoryMaskEngine
    engine = CategoryMaskEngine(num_classes=6)
    gamma_k = engine.update_and_compute(probs, w_i)
    assert gamma_k.shape == (6,), f"Expected gamma_k shape (6,), got {gamma_k.shape}"
    assert gamma_k.min() >= 0.0 and gamma_k.max() <= 1.0
    
    # High-level entry point
    w2, g2, occ2 = compute_weights_and_masks(logits, engine, tau_conf=0.65)
    assert w2.shape == (32,) and g2.shape == (6,) and occ2.shape == (32,)
    print("  [PASS] Confidence masking shapes, normalization, and bounds verified!")


def test_absent_class_gradient_cutoff():
    """
    When classes 4, 5, 6 are physically absent in the target domain:
    After sufficient EMA updates, gamma_4, gamma_5, gamma_6 -> 0.
    """
    print("--- 2. Testing Absent Class Masking Convergence ---")
    active_classes = [0, 1, 2, 3]  # Only background + first 3 activities
    engine = CategoryMaskEngine(num_classes=6, beta=0.5, tau_thresh=0.05, tau_temp=0.01)
    
    for _ in range(50):
        logits = generate_mock_logits(32, 2, 7, active_classes=active_classes)
        probs = compute_probs(logits)
        w_i = compute_sample_confidence(probs)
        gamma_k = engine.update_and_compute(probs, w_i)
    
    print(f"  Converged gamma_k for 6 activity classes: {gamma_k.tolist()}")
    # Active activity classes (1, 2, 3 -> index 0, 1, 2 in gamma) should be detected (> 0.5)
    # Absent classes (4, 5, 6 -> index 3, 4, 5 in gamma) should be suppressed (< 0.1)
    assert gamma_k[3] < 0.1, f"Expected gamma[3] < 0.1, got {gamma_k[3]:.4f}"
    assert gamma_k[4] < 0.1, f"Expected gamma[4] < 0.1, got {gamma_k[4]:.4f}"
    assert gamma_k[5] < 0.1, f"Expected gamma[5] < 0.1, got {gamma_k[5]:.4f}"
    print("  [PASS] Absent class suppression verified!")


def test_loss_engine():
    """Test CAPAFiLossEngine computation, output dictionary, and gradient flow."""
    print("--- 3. Testing loss_engine.py ---")
    loss_engine = CAPAFiLossEngine(lambda_ent=1.0, lambda_div=0.5, lambda_rot=0.5)
    
    # Synthetic inputs with requires_grad=True to test autograd backpropagation
    logits = torch.randn(32, 2, 7, requires_grad=True)
    rot_logits_0 = torch.randn(32, 2, requires_grad=True)
    rot_logits_180 = torch.randn(32, 2, requires_grad=True)
    w_i = torch.rand(32)
    gamma_k = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # partial shift mask
    p_occ = torch.rand(32)
    
    loss_dict = loss_engine(logits, rot_logits_0, rot_logits_180, w_i, gamma_k, p_occ)
    
    # Verify expected keys in loss dictionary
    expected_keys = [
        "loss_total",
        "loss_entropy",
        "loss_diversity",
        "loss_rotation",
        "mean_confidence",
        "active_classes_count",
        "occupancy_factor",
    ]
    for key in expected_keys:
        assert key in loss_dict, f"Missing key in loss_dict: {key}"
    
    assert loss_dict["loss_total"].requires_grad, "loss_total must support autograd backward!"
    
    # Test backward pass
    loss_dict["loss_total"].backward()
    assert logits.grad is not None, "Logits should receive gradients from entropy/diversity loss"
    assert rot_logits_0.grad is not None, "Rotation logits 0 must receive gradients"
    assert rot_logits_180.grad is not None, "Rotation logits 180 must receive gradients"
    print("  [PASS] Loss engine forward, telemetry metrics, and backward autograd passed!")


def test_trainer_adaptation_and_freezing():
    """Test TargetAdaptationTrainer: parameter isolation, adaptation epoch, and checkpointing."""
    print("--- 4. Testing trainer.py & Classifier Freezing Guarantee ---")
    model = MockHARModel(channels=3, latent_dim=256, slots=2, total_classes=7)
    
    config = {
        "learning_rate_backbone": 1.0e-4,
        "weight_decay": 1.0e-4,
        "lambda_ent": 1.0,
        "lambda_div": 0.5,
        "lambda_rot": 0.5,
        "num_classes": 6,
        "ema_decay_beta": 0.90,
        "tau_thresh": 0.05,
        "tau_temp": 0.04,
        "tau_conf": 0.65,
    }
    
    trainer = TargetAdaptationTrainer(model=model, config=config)
    
    # Verify parameter gradient flags
    for name, p in model.slot_head.named_parameters():
        assert not p.requires_grad, f"Classifier parameter {name} MUST be frozen!"
    for name, p in model.backbone.named_parameters():
        assert p.requires_grad, f"Backbone parameter {name} MUST be trainable!"
    for name, p in model.rot_head.named_parameters():
        assert p.requires_grad, f"Rotation head parameter {name} MUST be trainable!"
    
    # Snapshot classifier weights before adaptation
    classifier_state_before = {
        name: p.clone().detach() for name, p in model.slot_head.named_parameters()
    }
    backbone_weight_before = next(model.backbone.parameters()).clone().detach()
    
    # Create small synthetic DataLoader (2 batches of size 4)
    x_synth = torch.randn(8, 3, 500, 30)
    dataset = TensorDataset(x_synth)
    loader = DataLoader(dataset, batch_size=4)
    
    # Run 1 epoch of adaptation
    avg_loss = trainer.adapt_epoch(loader)
    assert isinstance(avg_loss, float), "adapt_epoch should return average loss float"
    
    # Verify classifier weights remain BITWISE IDENTICAL
    assert trainer.verify_classifier_frozen(classifier_state_before), "Classifier weights changed during adaptation!"
    
    # Verify backbone parameters WERE UPDATED by optimizer step
    backbone_weight_after = next(model.backbone.parameters()).detach()
    assert not torch.equal(backbone_weight_before, backbone_weight_after), "Backbone should have updated weights!"
    
    # Verify checkpoint saving
    checkpoint_path = REPO_ROOT / "checkpoints" / "test_adapted_checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists(), "Checkpoint file was not created!"
    checkpoint_path.unlink()  # Clean up test artifact
    
    print("  [PASS] Trainer parameter freeze, optimization step, and checkpointing verified!")


if __name__ == "__main__":
    print("==================================================")
    print("   RUNNING CAPA-Fi MODULE 3 STANDALONE TESTS      ")
    print("==================================================")
    test_confidence_masking()
    test_absent_class_gradient_cutoff()
    test_loss_engine()
    test_trainer_adaptation_and_freezing()
    print("==================================================")
    print("   ALL TESTS PASSED! MODULE 3 IS 70% READY!       ")
    print("==================================================")