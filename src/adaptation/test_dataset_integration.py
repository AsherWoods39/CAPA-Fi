"""
test_abey_dataset_integration.py
==================================
Integration test: load Abey's real CSV split files → verify tensor shapes
→ run one adaptation epoch through TargetAdaptationTrainer.

Run from the project root:
    python -m src.adaptation.test_abey_dataset_integration

▸ If your .npy CSI files are stored elsewhere, set DATA_DIR below.
▸ If no .npy files are available yet, a synthetic fallback is used
  automatically — every other part of the pipeline is still exercised.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# ── Make sure project root is on sys.path ─────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
SOURCE_CSV  = os.path.join(SPLITS_DIR, "source_split.csv")
TARGET_CSV  = os.path.join(SPLITS_DIR, "target_split.csv")

# ▸ Change this to the folder that contains your .npy files.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 4
FIXED_LEN   = 3000   # time-steps T
NUM_CLASSES = 6      # WiMANS activities (K); class-0 is background → K+1 total classes
NUM_SLOTS   = 1      # Slot dimension M; stub uses M=1 (single-slot prediction)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "num_classes"           : NUM_CLASSES,
    "learning_rate_backbone": 1e-4,
    "weight_decay"          : 1e-5,
    "lambda_ent"            : 1.0,
    "lambda_div"            : 1.0,
    "lambda_rot"            : 0.5,
    "ema_decay_beta"        : 0.9,
    "tau_thresh"            : 0.5,
    "tau_temp"              : 0.1,
    "tau_conf"              : 0.8,
}


# ══════════════════════════════════════════════════════════════════════════════
# STUB MODEL — mirrors the interface expected by TargetAdaptationTrainer
#   model(x)                → (logits [B, K], z [B, D])
#   model.forward_rotation(z) → rot_logits [B, 2]
#   model.backbone            → nn.Module (trainable)
#   model.slot_head           → nn.Module (frozen by trainer)
#   model.rot_head            → nn.Module (trainable)
# ══════════════════════════════════════════════════════════════════════════════
class StubCAPAFiModel(nn.Module):
    """
    Minimal stand-in for the full CAPAFi model.

    Input  : (B, C, T, S)  e.g. (4, 9, 3000, 30)
    Outputs:
        logits  – raw class scores  (B, M, K+1)
                    M = NUM_SLOTS slots
                    K+1 classes (class-0 = background "no person")
        z       – latent embedding  (B, embed_dim)
    """
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = NUM_CLASSES,   # K activity classes
        num_slots  : int = NUM_SLOTS,
        embed_dim  : int = 64,
    ):
        super().__init__()
        self.num_slots = num_slots
        K_plus_1 = num_classes + 1       # +1 for background class

        # backbone: collapses (T, S) → flat embedding
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 8)),          # (B, C, 16, 8)
            nn.Flatten(),                            # (B, C*16*8)
            nn.Linear(in_channels * 16 * 8, embed_dim),
            nn.ReLU(),
        )
        # slot_head: frozen classifier g_θ  → (B, M * (K+1))
        self.slot_head = nn.Linear(embed_dim, num_slots * K_plus_1)
        self._K_plus_1 = K_plus_1

        # rot_head: self-supervised rotation head h_ψ (2-way: 0° / 180°)
        self.rot_head = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)                             # (B, embed_dim)
        raw = self.slot_head(z)                          # (B, M * (K+1))
        B = x.shape[0]
        logits = raw.view(B, self.num_slots, self._K_plus_1)  # (B, M, K+1)
        return logits, z

    def forward_rotation(self, z: torch.Tensor) -> torch.Tensor:
        return self.rot_head(z)                          # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Inspect CSVs
# ══════════════════════════════════════════════════════════════════════════════
def step1_inspect_csvs():
    src_df = pd.read_csv(SOURCE_CSV, index_col=0)
    tgt_df = pd.read_csv(TARGET_CSV, index_col=0)

    print("=" * 65)
    print("📄  STEP 1 — CSV Schema Check")
    print(f"  source_split.csv : {len(src_df):,} rows")
    print(f"  target_split.csv : {len(tgt_df):,} rows")
    print(f"  Columns : {list(src_df.columns)}")
    print()
    print("  Source — first 3 rows:")
    print(src_df[["label","environment","wifi_band","number_of_users"]].head(3).to_string())
    print()
    print("  Target — first 3 rows:")
    print(tgt_df[["label","environment","wifi_band","number_of_users"]].head(3).to_string())
    print("=" * 65)
    return src_df, tgt_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ Build DataLoaders (real or synthetic fallback)
# ══════════════════════════════════════════════════════════════════════════════
def step2_build_loaders(src_df, tgt_df):
    npy_present = (
        os.path.isdir(DATA_DIR) and
        any(f.endswith(".npy") for f in os.listdir(DATA_DIR))
    )

    print("=" * 65)
    print("📦  STEP 2 — Building DataLoaders")
    if npy_present:
        from src.dataset import MultiUserCSIDatasetWithPreprocessing
        print(f"  ✅ Real .npy files found in: {DATA_DIR}")
        src_ds = MultiUserCSIDatasetWithPreprocessing(SOURCE_CSV, DATA_DIR, FIXED_LEN)
        tgt_ds = MultiUserCSIDatasetWithPreprocessing(TARGET_CSV, DATA_DIR, FIXED_LEN)
    else:
        print(f"  ⚠️  No .npy files found in: {DATA_DIR}")
        print("  ↳  Using SYNTHETIC lazy dataset (generates one sample at a time)")

        class SyntheticCSIDataset(torch.utils.data.Dataset):
            """
            Lazy synthetic dataset — allocates only ONE (C, T, S) tensor per
            __getitem__ call, so it never blows up RAM regardless of dataset size.
            """
            def __init__(self, df: pd.DataFrame, c: int = 9, t: int = FIXED_LEN, s: int = 30):
                self.labels = (
                    df["number_of_users"]
                    .fillna(0)
                    .clip(0, NUM_CLASSES - 1)
                    .astype(int)
                    .values
                )
                self.c, self.t, self.s = c, t, s

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                x = torch.randn(self.c, self.t, self.s)   # one sample at a time
                y = int(self.labels[idx])
                return x, y

        src_ds = SyntheticCSIDataset(src_df)
        tgt_ds = SyntheticCSIDataset(tgt_df)

    src_loader = DataLoader(src_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Source batches: {len(src_loader)} × batch_size={BATCH_SIZE}")
    print(f"  Target batches: {len(tgt_loader)} × batch_size={BATCH_SIZE}")
    print("=" * 65)
    return src_loader, tgt_loader


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Shape check (B, C, T, S)
# ══════════════════════════════════════════════════════════════════════════════
def step3_shape_check(tgt_loader):
    tgt_x = next(iter(tgt_loader))[0]
    assert tgt_x.dim() == 4, f"Expected 4-D tensor, got {tgt_x.shape}"
    B, C, T, S = tgt_x.shape
    print("=" * 65)
    print("🔷  STEP 3 — Shape Validation")
    print(f"  Target batch shape: {tgt_x.shape}")
    print(f"  Breakdown → B={B}  C={C} (antennas)  T={T} (time)  S={S} (subcarriers)")
    assert T == FIXED_LEN, f"Time axis should be {FIXED_LEN}, got {T}"
    print("  ✅ Shape is (B, C, T, S) — correct!")
    print("=" * 65)
    return tgt_x


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ─ Instantiate model + trainer, run one adaptation epoch
# ══════════════════════════════════════════════════════════════════════════════
def step4_run_trainer(tgt_loader):
    from src.adaptation.trainer import TargetAdaptationTrainer

    model   = StubCAPAFiModel().to(DEVICE)
    trainer = TargetAdaptationTrainer(model=model, config=CONFIG)

    print("=" * 65)
    print("⚙️   STEP 4 — Adaptation Trainer (1 epoch)")
    print("  Frozen  : model.slot_head  (classifier g_θ)")
    print("  Trainable: model.backbone + model.rot_head")

    # Capture classifier weights before adaptation
    state_before = {
        name: param.data.clone()
        for name, param in model.slot_head.named_parameters()
    }

    avg_loss = trainer.adapt_epoch(tgt_loader)
    print(f"\n  Average total loss over 1 epoch: {avg_loss:.6f}")

    # Verify classifier was NOT updated
    classifier_frozen = trainer.verify_classifier_frozen(state_before)
    assert classifier_frozen, "❌ Classifier weights changed — gradient leak!"
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n🚀  Device : {DEVICE}")
    print(f"    Project: {PROJECT_ROOT}\n")

    src_df, tgt_df = step1_inspect_csvs()
    src_loader, tgt_loader = step2_build_loaders(src_df, tgt_df)
    step3_shape_check(tgt_loader)
    step4_run_trainer(tgt_loader)

    print("\n✅  ALL INTEGRATION CHECKS PASSED\n")
