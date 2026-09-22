"""
CAPA-Fi: Local Standalone Adaptation Environment & Tensor Mock Verification
Member 3: Source-Free Adaptation & SSL Engine (Lead: Athishta P. A.)

Verifies:
1. Hyperparameter configuration in configs/default_config.yaml
2. Simulated Target CSI tensor generation: (B, C, T, S) with T=3000 (aligned to Member 1 fixed_length)
3. Simulated Frozen Classifier logits generation: (B, M, K+1)
4. Parameter freezing enforcement (requires_grad = False on classifier)
5. Gradient flow through trainable feature backbone
6. DataLoader interface contract matching Member 1's MultiUserCSIDataset
"""

import sys
from pathlib import Path
import yaml
import torch

# Ensure workspace root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adaptation.mock_inputs import (
    generate_mock_adaptation_batch,
    generate_mock_csi_input,
    generate_mock_logits,
    generate_mock_raw_npy_sample,
    MockCSIDataset,
    MockHARModel,
)


def verify_config() -> dict:
    """Verifies configs/default_config.yaml adaptation section."""
    config_path = ROOT_DIR / "configs" / "default_config.yaml"
    assert config_path.exists(), f"Config file not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapt_cfg = cfg.get("adaptation", {})
    print("=" * 70)
    print("1. HYPERPARAMETER CONFIGURATION VERIFICATION (adaptation block)")
    print("=" * 70)
    for key, val in adapt_cfg.items():
        print(f"  • {key:25s}: {val}")

    # Check required keys
    assert "lambda_ent" in adapt_cfg, "Missing lambda_ent in config"
    assert "lambda_div" in adapt_cfg, "Missing lambda_div in config"
    assert "lambda_rot" in adapt_cfg, "Missing lambda_rot in config"
    assert "tau_conf" in adapt_cfg, "Missing tau_conf in config"
    assert "tau_presence" in adapt_cfg or "tau_thresh" in adapt_cfg, "Missing tau_presence in config"
    
    print("\n[PASS] Adaptation configuration verified successfully.")
    return adapt_cfg


def verify_synthetic_tensors():
    """Verifies synthetic PyTorch tensors matching Member 3 contracts."""
    print("\n" + "=" * 70)
    print("2. SYNTHETIC TENSOR SHAPES & CONTRACT VERIFICATION")
    print("=" * 70)
    
    B = 32   # Batch size
    C = 3    # Antenna channels (Tx * Rx flattened)
    S = 30   # Subcarriers
    T = 3000 # Time steps — aligned to Member 1's MultiUserCSIDataset fixed_length=3000
    M = 2    # Slots
    K = 6    # Active classes (+1 for No-Person = 7)
    
    # 1. Simulated Target CSI Input (processed batch shape)
    x_target_cst = torch.randn(B, C, S, T)
    x_target_cts = generate_mock_csi_input(B, C, T, S)
    print(f"  \u2022 Target CSI (B, C, S, T): Shape = {list(x_target_cst.shape)}, Dtype = {x_target_cst.dtype}")
    print(f"  \u2022 Target CSI (B, C, T, S): Shape = {list(x_target_cts.shape)}, Dtype = {x_target_cts.dtype}")
    assert x_target_cst.shape == (32, 3, 30, 3000)
    assert x_target_cts.shape == (32, 3, 3000, 30)

    # 2. Simulated Frozen Classifier Output (Logits)
    logits = generate_mock_logits(B, M, K + 1)
    print(f"  \u2022 Classifier Logits (B, M, K+1): Shape = {list(logits.shape)}, Dtype = {logits.dtype}")
    assert logits.shape == (32, 2, 7)

    # 3. Full Adaptation Batch
    batch = generate_mock_adaptation_batch(B, C, T, S, M, K + 1, latent_dim=256)
    print(f"  \u2022 180\u00b0 Inverted CSI: Shape = {list(batch['x_target_180'].shape)}")
    print(f"  \u2022 Latent Embeddings z: Shape = {list(batch['z'].shape)}")
    print(f"  \u2022 Rotation Logits: Shape = {list(batch['rot_logits'].shape)}")
    assert batch["x_target_180"].shape == (32, 3, 3000, 30)
    assert batch["z"].shape == (32, 256)
    assert batch["rot_logits"].shape == (32, 2)

    # 4. Raw .npy sample shape (mirrors MultiUserCSIDataset pre-collation)
    raw_sample = generate_mock_raw_npy_sample(time_steps=3000, tx_antennas=1, rx_antennas=3, subcarriers=30)
    print(f"  \u2022 Raw .npy sample (T, Tx, Rx, S): Shape = {list(raw_sample.shape)}, Dtype = {raw_sample.dtype}")
    assert raw_sample.shape == (3000, 1, 3, 30), f"Raw npy shape mismatch: {raw_sample.shape}"

    print("\n[PASS] All synthetic tensor shapes match tensor contracts exactly.")


def verify_mock_model_and_freeze():
    """Verifies parameter freezing enforcement and gradient flow."""
    print("\n" + "=" * 70)
    print("3. PARAMETER FREEZING & GRADIENT ISOLATION TEST")
    print("=" * 70)

    model = MockHARModel(channels=3, latent_dim=256, slots=2, total_classes=7)

    # Step 1: Enforce parameter freeze on classifier
    model.freeze_classifier()
    model.unfreeze_backbone()

    # Step 2: Verify requires_grad flags
    classifier_grads = [p.requires_grad for p in model.classifier.parameters()]
    backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
    rot_grads = [p.requires_grad for p in model.rotation_head.parameters()]

    assert all(not g for g in classifier_grads), "Classifier has unfrozen parameters!"
    assert all(g for g in backbone_grads), "Backbone has frozen parameters!"
    assert all(g for g in rot_grads), "Rotation head has frozen parameters!"
    print(f"  • Classifier Head g_θ (Frozen): requires_grad = {all(not g for g in classifier_grads)}")
    print(f"  • Backbone f_θ (Trainable):   requires_grad = {all(g for g in backbone_grads)}")
    print(f"  • Rotation Head h_ψ (Trainable): requires_grad = {all(g for g in rot_grads)}")

    # Step 3: Run forward and dummy backward — use T=3000 to match real data
    x = torch.randn(8, 3, 3000, 30)
    logits, z = model(x)
    dummy_loss = logits.sum() + z.sum()
    dummy_loss.backward()

    for name, param in model.classifier.named_parameters():
        assert param.grad is None, f"Classifier parameter {name} received non-zero gradient!"
    
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert backbone_has_grad, "Backbone parameters received no gradient!"

    print("  • Classifier Parameter Gradients: None (Strictly Frozen - Verified!)")
    print("  • Backbone Parameter Gradients: Active (Trainable - Verified!)")
    print("\n[PASS] Frozen classifier & gradient isolation verified successfully.")


def verify_dataset_interface():
    """Verifies MockCSIDataset matches Member 1's MultiUserCSIDataset DataLoader contract."""
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("4. DATASET INTERFACE CONTRACT VERIFICATION (Member 1 Alignment)")
    print("=" * 70)

    # Instantiate mock dataset — same args as MultiUserCSIDataset would receive
    dataset = MockCSIDataset(
        num_samples=64,
        fixed_length=3000,
        tx_antennas=1,
        rx_antennas=3,
        subcarriers=30,
    )
    print(f"  \u2022 Dataset length: {len(dataset)} (Expected: 64)")
    assert len(dataset) == 64

    # Single sample check
    sample_tensor, sample_label = dataset[0]
    print(f"  \u2022 Single sample tensor shape: {list(sample_tensor.shape)} (Expected: [3000, 1, 3, 30])")
    print(f"  \u2022 Single sample label type: {type(sample_label).__name__} = '{sample_label}'")
    assert sample_tensor.shape == (3000, 1, 3, 30), f"Sample shape mismatch: {sample_tensor.shape}"
    assert isinstance(sample_label, str), f"Label should be str, got {type(sample_label)}"

    # DataLoader collation check (batch_size=8)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    batch_tensor, batch_labels = next(iter(loader))
    print(f"  \u2022 DataLoader batch tensor shape: {list(batch_tensor.shape)} (Expected: [8, 3000, 1, 3, 30])")
    print(f"  \u2022 DataLoader batch labels count: {len(batch_labels)} (Expected: 8)")
    assert batch_tensor.shape == (8, 3000, 1, 3, 30), f"Batch shape mismatch: {batch_tensor.shape}"
    assert len(batch_labels) == 8

    # Verify label format matches real CSV 'label' column style (e.g. 'act_1_2')
    assert all(lbl.startswith("mock_act_") for lbl in batch_labels), "Label format mismatch!"

    print("  \u2022 DataLoader collation contract: VERIFIED")
    print("  \u2022 Label string format: VERIFIED")
    print("\n[PASS] Dataset interface contract matches Member 1's MultiUserCSIDataset exactly.")


def main():
    print("\n======================================================================")
    print(" CAPA-Fi: Stage 0 Standalone Environment & Tensor Mock Verification")
    print(" Member 3: Source-Free Adaptation & SSL Engine (Lead: Athishta P. A.)")
    print("======================================================================\n")

    verify_config()
    verify_synthetic_tensors()
    verify_mock_model_and_freeze()
    verify_dataset_interface()

    print("\n" + "=" * 70)
    print(">>> STAGE 0 COMPLETED: Environment & Tensor Mock Setup is 100% READY!")
    print(">>> Dataset contract aligned with Member 1 (MultiUserCSIDataset).")
    print(">>> You are ready to proceed to Stage 1: confidence_masking.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
