"""
CAPA-Fi: Spatial-Temporal Feature Encoder Backbone (Member 2)

Architecture:
    CSI [B, 3, 3000, 30]
      → 3 × CNN2D blocks (conv → batchnorm → ReLU → maxpool2d)
      → reshape [B, 128, 375, 3] → [B, 375, 384]
      → 2-layer bidirectional LSTM (hidden=128)
      → mean-pool across time
      → z = [B, 256]

Tensor contract:
    Input:  [B, C=3, T=3000, S=30]
    Output: [B, D=256]

Intermediate shapes:
    Block 1: [B, 32, 1500, 15]
    Block 2: [B, 64,  750,  7]
    Block 3: [B, 128, 375,  3]
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class CNN2DBlock(nn.Module):
    """
    Single CNN2D block: Conv2d → BatchNorm2d → ReLU → MaxPool2d(2,2).

    Each block doubles the channel count and halves both spatial dimensions
    (time and subcarriers) via 2×2 max pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, S]
        Returns:
            [B, C_out, T//2, S//2]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SpatialTemporalBackbone(nn.Module):
    """
    Full spatial-temporal backbone: CNN2D feature extractor + BiLSTM temporal encoder.

    Maps CSI input [B, 3, 3000, 30] → latent embedding z [B, 256].

    Args:
        in_channels: Number of input channels (default: 3, from Tx*Rx antenna pairs)
        spatial_channels: Channel progression for CNN blocks (default: [32, 64, 128])
        temporal_hidden_dim: BiLSTM hidden dimension per direction (default: 128)
        temporal_layers: Number of BiLSTM layers (default: 2)
        dropout: Dropout rate for CNN blocks and BiLSTM (default: 0.2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        spatial_channels: Optional[List[int]] = None,
        temporal_hidden_dim: int = 128,
        temporal_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        if spatial_channels is None:
            spatial_channels = [32, 64, 128]

        # --- CNN2D spatial feature extraction ---
        self.cnn_blocks = nn.ModuleList()
        ch_in = in_channels
        for ch_out in spatial_channels:
            self.cnn_blocks.append(CNN2DBlock(ch_in, ch_out, dropout=dropout))
            ch_in = ch_out

        # After CNN: [B, 128, 375, 3] → reshape → [B, 375, 128*3=384]
        self.final_cnn_channels = spatial_channels[-1]

        # The subcarrier dimension after 3 rounds of 2×2 pooling: 30 → 15 → 7 → 3
        # lstm_input_size = final_cnn_channels * remaining_subcarriers
        # This is computed dynamically in forward() for robustness, but expected to be 384

        # --- BiLSTM temporal encoding ---
        self.temporal_hidden_dim = temporal_hidden_dim
        self.lstm = nn.LSTM(
            input_size=384,  
            hidden_size=temporal_hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )

        # Output dimension: hidden_dim × 2 (bidirectional)
        self.latent_dim = temporal_hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN + BiLSTM backbone.

        Args:
            x: CSI input tensor [B, C, T, S] = [B, 3, 3000, 30]

        Returns:
            z: Latent embedding [B, D] = [B, 256]
        """
        # --- CNN2D blocks ---
        for block in self.cnn_blocks:
            x = block(x)
        # x shape: [B, 128, 375, 3] (expected)

        B, C, T, S = x.shape

        # --- Reshape for BiLSTM ---
        # [B, C, T, S] → [B, T, C*S]
        x = x.permute(0, 2, 1, 3)  # [B, T, C, S]
        x = x.reshape(B, T, C * S)  # [B, T, C*S] = [B, 375, 384]

        # --- BiLSTM ---
        lstm_out, _ = self.lstm(x)  # [B, T, 2*hidden] = [B, 375, 256]

        # --- Mean-pool across time axis ---
        z = lstm_out.mean(dim=1)  # [B, 256]

        return z

    @classmethod
    def from_config(cls, config: Dict) -> "SpatialTemporalBackbone":
        """
        Construct from a config dictionary (parsed from YAML).

        Expected config keys under 'model':
            - spatial_channels: [32, 64, 128]
            - temporal_hidden_dim: 128
            - temporal_layers: 2
            - dropout: 0.2

        Expected config keys under 'data':
            - channels_C: 3
        """
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            in_channels=data_cfg.get("channels_C", 3),
            spatial_channels=model_cfg.get("spatial_channels", [32, 64, 128]),
            temporal_hidden_dim=model_cfg.get("temporal_hidden_dim", 128),
            temporal_layers=model_cfg.get("temporal_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        )
