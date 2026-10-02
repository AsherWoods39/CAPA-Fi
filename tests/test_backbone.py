"""
CAPA-Fi: Backbone Unit Tests (Member 2)

Test 1 (gating): Verify shape contract [B,3,3000,30] → [B,256]
with intermediate CNN shapes validated at each block.
"""

import pytest
import torch

from src.models.backbone import CNN2DBlock, SpatialTemporalBackbone


class TestCNN2DBlock:
    """Tests for individual CNN2D blocks."""

    def test_shape(self):
        block = CNN2DBlock(in_channels=3, out_channels=32)
        x = torch.randn(2, 3, 100, 30)
        out = block(x)
        assert out.shape == (2, 32, 50, 15), f"Expected (2,32,50,15), got {out.shape}"

    def test_no_nans(self):
        block = CNN2DBlock(in_channels=3, out_channels=32)
        x = torch.randn(2, 3, 100, 30)
        out = block(x)
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"


class TestSpatialTemporalBackbone:
    """Tests for the full backbone: CNN + BiLSTM."""

    @pytest.fixture
    def backbone(self):
        return SpatialTemporalBackbone(
            in_channels=3,
            spatial_channels=[32, 64, 128],
            temporal_hidden_dim=128,
            temporal_layers=2,
            dropout=0.0,  # Disable dropout for deterministic testing
        )

    def test_output_shape(self, backbone):
        """Primary contract: [B,3,3000,30] → [B,256]."""
        x = torch.randn(2, 3, 3000, 30)
        z = backbone(x)
        assert z.shape == (2, 256), f"Expected (2,256), got {z.shape}"

    def test_intermediate_cnn_shapes(self, backbone):
        """Verify the agreed temporal downsampling schedule: 3000→1500→750→375."""
        x = torch.randn(2, 3, 3000, 30)

        # Forward through each CNN block and check shapes
        out = x
        expected_shapes = [
            (2, 32, 1500, 15),   # Block 1
            (2, 64, 750, 7),     # Block 2
            (2, 128, 375, 3),    # Block 3
        ]

        for i, block in enumerate(backbone.cnn_blocks):
            out = block(out)
            assert out.shape == expected_shapes[i], (
                f"CNN Block {i+1}: expected {expected_shapes[i]}, got {out.shape}"
            )

    def test_no_nans_forward(self, backbone):
        """Forward pass should not produce NaNs."""
        x = torch.randn(2, 3, 3000, 30)
        z = backbone(x)
        assert not torch.isnan(z).any(), "Output z contains NaNs"
        assert not torch.isinf(z).any(), "Output z contains Infs"

    def test_backward_no_nans(self, backbone):
        """Backward pass should not produce NaN gradients."""
        x = torch.randn(2, 3, 3000, 30, requires_grad=True)
        z = backbone(x)
        loss = z.sum()
        loss.backward()
        assert x.grad is not None, "No gradient computed"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaNs"

    def test_different_batch_sizes(self, backbone):
        """Should work with various batch sizes."""
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 3, 3000, 30)
            z = backbone(x)
            assert z.shape == (bs, 256), f"Batch size {bs}: expected ({bs},256), got {z.shape}"

    def test_from_config(self):
        """Test construction from config dict."""
        config = {
            "model": {
                "spatial_channels": [32, 64, 128],
                "temporal_hidden_dim": 128,
                "temporal_layers": 2,
                "dropout": 0.2,
            },
            "data": {
                "channels_C": 3,
            },
        }
        backbone = SpatialTemporalBackbone.from_config(config)
        x = torch.randn(2, 3, 3000, 30)
        z = backbone(x)
        assert z.shape == (2, 256)
