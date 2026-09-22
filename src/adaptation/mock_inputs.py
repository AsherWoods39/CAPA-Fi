"""
CAPA-Fi: Mock Tensors & Synthetic Architecture Generator for Member 3
Allows standalone development, verification, and unit-testing of the adaptation engine
without requiring Member 1 (Data) or Member 2 (Models) to be completed first.

Updated (Stage 0 v2): Aligned with Member 1's MultiUserCSIDataset contract:
  - Raw .npy shape: (T, Tx, Rx, S) with T=fixed_length=3000
  - Processed batch shape after collation: (B, C, T, S) where C = Tx * Rx
  - MockCSIDataset mirrors the MultiUserCSIDataset interface for DataLoader testing
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_mock_csi_input(
    batch_size: int = 32,
    channels: int = 3,
    time_steps: int = 3000,
    subcarriers: int = 30,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generates synthetic target CSI tensor matching the processed batch contract:
    Shape: (B, C, T, S) = (32, 3, 3000, 30)

    NOTE: T=3000 is set to match Member 1's fixed_length in MultiUserCSIDataset.
    The raw .npy shape is (T, Tx, Rx, S); after flattening Tx*Rx into C and
    collating into a batch, the adapter receives (B, C, T, S).
    Normalized to zero-mean and unit-variance.
    """
    x = torch.randn(batch_size, channels, time_steps, subcarriers, device=device, dtype=torch.float32)
    return x


def generate_mock_raw_npy_sample(
    time_steps: int = 3000,
    tx_antennas: int = 1,
    rx_antennas: int = 3,
    subcarriers: int = 30,
) -> np.ndarray:
    """
    Generates a single synthetic .npy sample matching MultiUserCSIDataset's raw input:
    Shape: (T, Tx, Rx, S) = (3000, 1, 3, 30)

    This mirrors what np.load(file_path).astype(np.float32) returns before any
    padding/truncation in __getitem__. Use this to test DataLoader integration
    without real .npy files.
    """
    return np.random.randn(time_steps, tx_antennas, rx_antennas, subcarriers).astype(np.float32)


def generate_mock_logits(
    batch_size: int = 32,
    slots: int = 2,
    total_classes: int = 7,
    active_classes: Optional[List[int]] = None,
    noise_level: float = 0.5,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generates synthetic multi-slot classifier logits:
    Shape: (B, M, K+1) = (32, 2, 7)
    
    If active_classes is provided (e.g. [0, 1, 2, 3] for partial shift),
    the logits for absent classes (e.g. 4, 5, 6) will have strongly negative scores
    simulating their true physical absence in the target domain.
    """
    if active_classes is None:
        return torch.randn(batch_size, slots, total_classes, device=device, dtype=torch.float32)
    
    # Initialize all classes with low baseline logits (simulating absent classes)
    logits = torch.full((batch_size, slots, total_classes), -5.0, device=device, dtype=torch.float32)
    
    # Give present classes competitive positive logits
    for b in range(batch_size):
        for m in range(slots):
            # Randomly select one present class to be confident
            chosen_cls = active_classes[torch.randint(0, len(active_classes), (1,)).item()]
            logits[b, m, chosen_cls] = 3.0 + noise_level * torch.randn(1).item()
            # Add small random perturbation to other present classes
            for c in active_classes:
                if c != chosen_cls:
                    logits[b, m, c] += noise_level * torch.randn(1).item()
                    
    return logits


def generate_mock_adaptation_batch(
    batch_size: int = 32,
    channels: int = 3,
    time_steps: int = 3000,
    subcarriers: int = 30,
    slots: int = 2,
    total_classes: int = 7,
    latent_dim: int = 256,
    active_classes: Optional[List[int]] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Generates a full synthetic adaptation batch matching all tensor contracts:
    - x_target: (B, C, T, S) = (B, 3, 3000, 30)
    - x_target_180: (B, C, T, S) - 180° inverted temporal-frequency matrix
    - z: (B, D) - Latent feature vector
    - logits: (B, M, K+1) - Multi-slot classification logits
    - rot_logits: (B, 2) - Binary rotation SSL logits

    NOTE: T=3000 matches Member 1's MultiUserCSIDataset fixed_length.
    """
    x_target = generate_mock_csi_input(
        batch_size=batch_size,
        channels=channels,
        time_steps=time_steps,
        subcarriers=subcarriers,
        device=device,
    )
    # 180° inversion flips temporal (dim 2) and subcarrier (dim 3) dimensions
    x_target_180 = torch.flip(x_target, dims=[-2, -1])
    
    z = torch.randn(batch_size, latent_dim, device=device, dtype=torch.float32)
    logits = generate_mock_logits(
        batch_size=batch_size,
        slots=slots,
        total_classes=total_classes,
        active_classes=active_classes,
        device=device,
    )
    rot_logits = torch.randn(batch_size, 2, device=device, dtype=torch.float32)
    
    return {
        "x_target": x_target,
        "x_target_180": x_target_180,
        "z": z,
        "logits": logits,
        "rot_logits": rot_logits,
    }


class MockCSIDataset(Dataset):
    """
    In-memory mock that mirrors the interface of Member 1's MultiUserCSIDataset.

    Allows Stage 0 DataLoader integration testing without real .npy files on disk.
    Each sample returns a tuple (tensor, sample_name) exactly as the real dataset does,
    where tensor has shape (T, Tx, Rx, S) before collation.

    Matches MultiUserCSIDataset contract:
      - __len__  : returns num_samples
      - __getitem__: returns (torch.Tensor shape [T, Tx, Rx, S], str label)
    """
    def __init__(
        self,
        num_samples: int = 64,
        fixed_length: int = 3000,
        tx_antennas: int = 1,
        rx_antennas: int = 3,
        subcarriers: int = 30,
    ):
        self.num_samples = num_samples
        self.fixed_length = fixed_length
        self.tx_antennas = tx_antennas
        self.rx_antennas = rx_antennas
        self.subcarriers = subcarriers
        # Pre-generate labels matching the real CSV 'label' column format
        self.labels = [f"mock_act_{i}_{i * 3}" for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns (tensor, label) matching MultiUserCSIDataset.__getitem__.
        Tensor shape: (T, Tx, Rx, S) = (fixed_length, tx_antennas, rx_antennas, subcarriers)
        """
        raw = generate_mock_raw_npy_sample(
            time_steps=self.fixed_length,
            tx_antennas=self.tx_antennas,
            rx_antennas=self.rx_antennas,
            subcarriers=self.subcarriers,
        )
        return torch.tensor(raw), self.labels[idx]


class MockFeatureBackbone(nn.Module):
    """
    Lightweight mock spatial-temporal backbone f_θ:
    Maps CSI (B, C, T, S) -> Latent embedding z (B, D).
    """
    def __init__(self, channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(channels * 64, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, T, S)
        return self.encoder(x)


class MockClassifierHead(nn.Module):
    """
    Lightweight mock multi-slot classification head g_θ:
    Maps Latent z (B, D) -> Multi-slot Logits (B, M, K+1).
    This module will be FROZEN during target adaptation (requires_grad = False).
    """
    def __init__(self, latent_dim: int = 256, slots: int = 2, total_classes: int = 7):
        super().__init__()
        self.slots = slots
        self.total_classes = total_classes
        self.fc = nn.Linear(latent_dim, slots * total_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        out = self.fc(z)
        return out.view(b, self.slots, self.total_classes)


class MockRotationHead(nn.Module):
    """
    Lightweight mock self-supervised rotation head h_ψ:
    Maps Latent z (B, D) -> Binary Rotation Logits (B, 2) (0° vs 180°).
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class MockHARModel(nn.Module):
    """
    End-to-end mock permutation-invariant model encapsulating:
    - backbone (f_θ) : Trainable during adaptation
    - classifier / slot_head (g_θ) : Frozen during adaptation
    - rotation_head (h_ψ) : Trainable during adaptation
    """
    def __init__(
        self,
        channels: int = 3,
        latent_dim: int = 256,
        slots: int = 2,
        total_classes: int = 7,
    ):
        super().__init__()
        self.backbone = MockFeatureBackbone(channels=channels, latent_dim=latent_dim)
        self.classifier = MockClassifierHead(latent_dim=latent_dim, slots=slots, total_classes=total_classes)
        self.rotation_head = MockRotationHead(latent_dim=latent_dim)
        
        # Alias for consistency with architecture spec names
        self.slot_head = self.classifier
        self.rot_head = self.rotation_head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        logits = self.classifier(z)
        return logits, z

    def forward_rotation(self, z: torch.Tensor) -> torch.Tensor:
        return self.rotation_head(z)

    def freeze_classifier(self) -> None:
        """Enforces strictly requires_grad = False on classifier head."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Enforces requires_grad = True on feature backbone and rotation head."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.rotation_head.parameters():
            param.requires_grad = True
