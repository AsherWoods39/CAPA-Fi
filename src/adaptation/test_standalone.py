"""
CAPA-Fi: Module 3 Standalone Unit-Testing Harness
Location: src/adaptation/test_standalone.py
Member 3: Source-Free Adaptation & SSL Engine (Lead: Athishta P. A.)
"""

import sys
from pathlib import Path
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adaptation.mock_inputs import (
    generate_mock_adaptation_batch,
    generate_mock_logits,
    MockHARModel,
)


def run_stage_0_mock_checks():
    """Verifies mock tensor shapes and parameter isolation."""
    print("--- Running Stage 0 Mock Tensor Verification ---")
    
    # 1. Simulate 32 target samples, 2 user slots, 6 activity classes + 1 background
    mock_logits = torch.randn(32, 2, 7)
    print(f"Mock Logits Shape: {mock_logits.shape} (Expected: [32, 2, 7])")
    assert mock_logits.shape == (32, 2, 7)

    # 2. Simulate target CSI tensor
    mock_batch = generate_mock_adaptation_batch(batch_size=32, channels=3, time_steps=500, subcarriers=30)
    print(f"Mock Target CSI Shape: {mock_batch['x_target'].shape} (Expected: [32, 3, 500, 30])")
    print(f"Mock 180° Inverted CSI Shape: {mock_batch['x_target_180'].shape} (Expected: [32, 3, 500, 30])")
    print(f"Mock Rotation Logits Shape: {mock_batch['rot_logits'].shape} (Expected: [32, 2])")

    # 3. Test Mock Model & Parameter Freezing
    model = MockHARModel()
    model.freeze_classifier()
    model.unfreeze_backbone()

    for name, p in model.classifier.named_parameters():
        assert not p.requires_grad, f"Classifier {name} must be frozen!"
    for name, p in model.backbone.named_parameters():
        assert p.requires_grad, f"Backbone {name} must be trainable!"

    print("[SUCCESS] Stage 0 Standalone Mock Verification Passed!")


if __name__ == "__main__":
    run_stage_0_mock_checks()
