"""
CAPA-Fi: Unified HAR Model Wrapper (Member 2)

Wires backbone + classifier + rotation_head into a single nn.Module.
This is a DROP-IN REPLACEMENT for MockHARModel in src/adaptation/mock_inputs.py.

API contract (must match MockHARModel exactly):
    model.backbone          → SpatialTemporalBackbone
    model.classifier        → MultiSlotLinear
    model.rotation_head     → BinaryRotationMLP
    model.slot_head         → alias for classifier
    model.rot_head          → alias for rotation_head

    model.forward(x)              → (logits [B, M, K+1], z [B, D])
    model.forward_rotation(z)     → rot_logits [B, 2]
    model.freeze_classifier()     → sets classifier requires_grad = False
    model.unfreeze_backbone()     → sets backbone + rotation_head requires_grad = True
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.backbone import SpatialTemporalBackbone
from src.models.classifier import MultiSlotLinear
from src.models.rotation_head import BinaryRotationMLP


class HARModel(nn.Module):
    """
    End-to-end permutation-invariant Human Activity Recognition model.

    Encapsulates:
        - backbone (f_θ):      Trainable spatial-temporal encoder
        - classifier (g_θ):    Multi-slot head, frozen during adaptation
        - rotation_head (h_ψ): Binary rotation SSL head, trainable during adaptation

    Args:
        in_channels: Number of input CSI channels (default: 3)
        spatial_channels: CNN channel progression (default: [32, 64, 128])
        temporal_hidden_dim: BiLSTM hidden dim per direction (default: 128)
        temporal_layers: Number of BiLSTM layers (default: 2)
        latent_dim: Final feature dimension D (default: 256)
        num_slots: Number of prediction slots M (default: 6)
        total_classes: Total classes K+1 incl. No-Person (default: 7)
        rot_hidden_dim: Rotation head hidden dimension (default: 64)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        spatial_channels: list = None,
        temporal_hidden_dim: int = 128,
        temporal_layers: int = 2,
        latent_dim: int = 256,
        num_slots: int = 6,
        total_classes: int = 7,
        rot_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        if spatial_channels is None:
            spatial_channels = [32, 64, 128]

        # --- Core components ---
        self.backbone = SpatialTemporalBackbone(
            in_channels=in_channels,
            spatial_channels=spatial_channels,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_layers=temporal_layers,
            dropout=dropout,
        )

        self.classifier = MultiSlotLinear(
            latent_dim=latent_dim,
            num_slots=num_slots,
            total_classes=total_classes,
        )

        self.rotation_head = BinaryRotationMLP(
            latent_dim=latent_dim,
            hidden_dim=rot_hidden_dim,
        )

        # --- Aliases for consistency with MockHARModel and architecture spec ---
        self.slot_head = self.classifier
        self.rot_head = self.rotation_head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: CSI → backbone → classifier.

        Args:
            x: CSI input tensor [B, C, T, S] = [B, 3, 3000, 30]

        Returns:
            logits: Multi-slot class logits [B, M, K+1] = [B, 6, 7]
            z: Latent embedding [B, D] = [B, 256]
        """
        z = self.backbone(x)
        logits = self.classifier(z)
        return logits, z

    def forward_rotation(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through rotation head only.

        Args:
            z: Latent embedding [B, D] = [B, 256]

        Returns:
            rot_logits: Binary rotation logits [B, 2]
        """
        return self.rotation_head(z)

    def freeze_classifier(self) -> None:
        """Strictly enforce requires_grad = False on the classifier head."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Enforce requires_grad = True on backbone and rotation head."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.rotation_head.parameters():
            param.requires_grad = True

    @classmethod
    def from_config(cls, config: Dict) -> "HARModel":
        """
        Construct from a config dictionary (parsed from YAML).

        Expected config structure:
            model:
                spatial_channels: [32, 64, 128]
                temporal_hidden_dim: 128
                temporal_layers: 2
                latent_dim_D: 256
                max_users_M: 6
                total_classes: 7
                rot_hidden_dim: 64
                dropout: 0.2
            data:
                channels_C: 3
        """
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            in_channels=data_cfg.get("channels_C", 3),
            spatial_channels=model_cfg.get("spatial_channels", [32, 64, 128]),
            temporal_hidden_dim=model_cfg.get("temporal_hidden_dim", 128),
            temporal_layers=model_cfg.get("temporal_layers", 2),
            latent_dim=model_cfg.get("latent_dim_D", 256),
            num_slots=model_cfg.get("max_users_M", 6),
            total_classes=model_cfg.get("total_classes", 7),
            rot_hidden_dim=model_cfg.get("rot_hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.2),
        )

    def save_checkpoint(self, path: str, epoch: int = 0, optimizer=None, config=None):
        """
        Save model checkpoint with separately-addressable component state dicts.

        Member 3 can load and freeze/adapt components independently.

        Args:
            path: File path to save checkpoint
            epoch: Current training epoch
            optimizer: Optional optimizer to save state
            config: Optional config dict to store architecture hyperparameters
        """
        checkpoint = {
            "backbone": self.backbone.state_dict(),
            "classifier": self.classifier.state_dict(),
            "rotation_head": self.rotation_head.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if config is not None:
            checkpoint["config"] = config

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer=None):
        """
        Load model checkpoint with separately-addressable component state dicts.

        Args:
            path: File path to load checkpoint from
            load_optimizer: Optional optimizer to restore state into

        Returns:
            checkpoint dict (for accessing epoch, config, etc.)
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.backbone.load_state_dict(checkpoint["backbone"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.rotation_head.load_state_dict(checkpoint["rotation_head"])

        if load_optimizer is not None and "optimizer" in checkpoint:
            load_optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint
