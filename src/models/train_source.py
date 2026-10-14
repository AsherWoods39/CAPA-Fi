"""
CAPA-Fi: Source Domain Pre-Training Script (Member 2)

End-to-end supervised training with Hungarian matching + rotation SSL.

Pipeline per batch:
    1. Load labeled source batch (x, target_slots) from DataLoader
    2. Apply augmentation to classification inputs
    3. Construct rotation pair: x_original (label=0) + x_rotated (label=1)
    4. Forward through backbone → z
    5. Forward z through classifier → logits [B, 6, 7]
    6. Forward z through rotation head → rot_logits [2B, 2]
    7. Hungarian match logits vs target_slots
    8. Compute L_matched_CE + lambda_rot_source × L_rotation
    9. Backprop + optimizer step

Training objective:
    L_source = L_matched_CE + λ_rot_source × L_rotation

Usage:
    python -m src.models.train_source --config configs/default_config.yaml

NOTE on data contract:
    Source training requires (x [B,3,3000,30], target_slots [B,6]).
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import yaml

from src.dataset import MultiUserCSIDatasetWithPreprocessing
from src.models.har_model import HARModel
from src.models.losses import SourceTrainingLoss
from src.utils.augmentation import CSIAugmentor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Fallback Source Dataset
# =============================================================================

class SourceTrainingDataset(Dataset):
    """
    Synthetic source training dataset fallback.

    Returns:
        x: CSI tensor [C, T, S] = [3, 3000, 30]
        target_slots: Activity class indices [M] = [6], values in {0..6}
    """

    def __init__(
        self,
        num_samples: int = 256,
        channels: int = 3,
        time_steps: int = 3000,
        subcarriers: int = 30,
        num_slots: int = 6,
        total_classes: int = 7,
    ):
        self.num_samples = num_samples
        self.channels = channels
        self.time_steps = time_steps
        self.subcarriers = subcarriers
        self.num_slots = num_slots
        self.total_classes = total_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.channels, self.time_steps, self.subcarriers)
        target_slots = torch.randint(0, self.total_classes, (self.num_slots,))
        return x, target_slots


def create_rotation_pair(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create rotation pair from a batch of CSI tensors.

    This is a fallback implementation for development/testing.
    In production, Member 1's preprocessing provides the 180° transform.

    Args:
        x: Original CSI tensor [B, C, T, S]

    Returns:
        x_combined: Concatenated [2B, C, T, S] (original + rotated)
        rot_labels: [2B] rotation labels (0=original, 1=rotated)
        z_split_size: B (for splitting z after backbone)
    """
    B = x.shape[0]

    # 180° inversion: flip temporal and subcarrier axes
    x_rotated = torch.flip(x, dims=[-2, -1])

    # Concatenate along batch dimension
    x_combined = torch.cat([x, x_rotated], dim=0)  # [2B, C, T, S]

    # Rotation labels: 0 for original, 1 for rotated
    rot_labels = torch.cat([
        torch.zeros(B, dtype=torch.long, device=x.device),
        torch.ones(B, dtype=torch.long, device=x.device),
    ])  # [2B]

    return x_combined, rot_labels, B


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model: HARModel,
    dataloader: DataLoader,
    loss_fn: SourceTrainingLoss,
    optimizer: torch.optim.Optimizer,
    augmentor: Optional[CSIAugmentor],
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dict with average loss values for the epoch.
    """
    model.train()

    running_total = 0.0
    running_ce = 0.0
    running_rot = 0.0
    num_batches = 0

    for batch_idx, (x, target_slots) in enumerate(dataloader):
        x = x.to(device)                         # [B, C, T, S]
        target_slots = target_slots.to(device)   # [B, M]

        # --- Apply augmentation (classification inputs only) ---
        if augmentor is not None:
            x_aug = augmentor(x)
        else:
            x_aug = x

        # --- Create rotation pair ---
        # x_combined: [2B, C, T, S], rot_labels: [2B]
        x_combined, rot_labels, B = create_rotation_pair(x_aug)

        # --- Forward pass through backbone (shared weights for both) ---
        # Process combined batch to get z for both original and rotated
        logits_combined, z_combined = model(x_combined)  # [2B, M, K+1], [2B, D]

        # Split: use only original samples for classification
        logits = logits_combined[:B]       # [B, M, K+1]

        # Rotation head on combined z
        rot_logits = model.forward_rotation(z_combined)  # [2B, 2]

        # --- Compute combined loss ---
        losses = loss_fn(logits, target_slots, rot_logits, rot_labels)

        # --- Backprop ---
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # --- Accumulate metrics ---
        running_total += losses["total"].item()
        running_ce += losses["matched_ce"].item()
        running_rot += losses["rotation"].item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {losses['total'].item():.4f} "
                f"(CE: {losses['matched_ce'].item():.4f}, "
                f"Rot: {losses['rotation'].item():.4f})"
            )

    avg_metrics = {
        "total": running_total / max(num_batches, 1),
        "matched_ce": running_ce / max(num_batches, 1),
        "rotation": running_rot / max(num_batches, 1),
    }

    return avg_metrics


def train_source(config: Dict) -> HARModel:
    """
    Full source domain pre-training pipeline.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Trained HARModel instance.
    """
    # --- Validate config ---
    source_cfg = config.get("source_training", {})
    lambda_rot_source = source_cfg.get("lambda_rot_source", None)

    if lambda_rot_source is None:
        raise ValueError(
            "source_training.lambda_rot_source is not set in the config. "
            "This value is required for source training and must be decided by the team. "
            "Do not hard-code a default — set it explicitly in configs/default_config.yaml."
        )

    # --- Device setup ---
    system_cfg = config.get("system", {})
    device_str = system_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Seed for reproducibility ---
    seed = system_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Build model ---
    model = HARModel.from_config(config).to(device)
    logger.info(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Build dataset and dataloader ---
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    source_split_csv = source_cfg.get("source_split_csv", "splits/source_split.csv")

    if os.path.isfile(source_split_csv):
        logger.info(f"Loading source dataset from {source_split_csv}")
        dataset = MultiUserCSIDatasetWithPreprocessing(
            annotation_file=source_split_csv,
            data_dir=data_cfg.get("processed_dir", "data/processed/WiMANS"),
            fixed_length=data_cfg.get("window_size_T", 3000),
            num_slots=model_cfg.get("max_users_M", 6),
            in_channels=data_cfg.get("channels_C", 3),
            subcarriers=data_cfg.get("subcarriers_S", 30),
        )
    else:
        logger.info("Using synthetic fallback SourceTrainingDataset")
        dataset = SourceTrainingDataset(
            channels=data_cfg.get("channels_C", 3),
            time_steps=data_cfg.get("window_size_T", 3000),
            subcarriers=data_cfg.get("subcarriers_S", 30),
            num_slots=model_cfg.get("max_users_M", 6),
            total_classes=model_cfg.get("total_classes", 7),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=source_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=system_cfg.get("num_workers", 0),
        pin_memory=system_cfg.get("pin_memory", True) if torch.cuda.is_available() else False,
        drop_last=True,
    )

    # --- Build loss function ---
    loss_fn = SourceTrainingLoss(lambda_rot_source=lambda_rot_source)

    # --- Build optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=source_cfg.get("learning_rate", 1e-3),
        weight_decay=source_cfg.get("weight_decay", 1e-4),
    )

    # --- Build scheduler ---
    epochs = source_cfg.get("epochs", 50)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Build augmentor ---
    augmentor = CSIAugmentor()

    # --- Ensure checkpoint directory exists ---
    checkpoint_path = source_cfg.get("checkpoint_path", "checkpoints/source_model.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # --- Training loop ---
    logger.info(f"Starting source training for {epochs} epochs")
    logger.info(f"  lambda_rot_source = {lambda_rot_source}")
    logger.info(f"  batch_size = {source_cfg.get('batch_size', 32)}")
    logger.info(f"  learning_rate = {source_cfg.get('learning_rate', 1e-3)}")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            augmentor=augmentor,
            device=device,
            epoch=epoch,
        )

        scheduler.step()
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {metrics['total']:.4f} "
            f"(CE: {metrics['matched_ce']:.4f}, Rot: {metrics['rotation']:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- Save best checkpoint ---
        if metrics["total"] < best_loss:
            best_loss = metrics["total"]
            model.save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                optimizer=optimizer,
                config=config,
            )
            logger.info(f"  → Saved best checkpoint (loss={best_loss:.4f})")

    # --- Save final checkpoint ---
    final_path = checkpoint_path.replace(".pt", "_final.pt")
    model.save_checkpoint(
        path=final_path,
        epoch=epochs,
        optimizer=optimizer,
        config=config,
    )
    logger.info(f"Training complete. Final checkpoint saved to {final_path}")

    return model


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CAPA-Fi Source Domain Pre-Training (Member 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run training
    train_source(config)


if __name__ == "__main__":
    main()
