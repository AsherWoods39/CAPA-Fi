"""
CAPA-Fi: Target Adaptation Engine Runner & Verification Pipeline
Location: src/adaptation/run_adaptation.py
Role: Member 3 (Source-Free Adaptation & SSL Engine Lead: Athishta P. A.)

Executes:
1. Ingestion of configs/default_config.yaml
2. Ingestion of source model checkpoint (checkpoints/source_model.pt) or architectural fallback
3. Loading of target dataset (splits/target_split.csv or synthetic CSI streaming fallback)
4. Full multi-epoch adaptation loop using CAPAFiAdaptationEngine
5. Verification of frozen classifier bitwise invariant
6. Persistence of adapted checkpoint (checkpoints/adapted_model.pt) and telemetry (results/adaptation_history.json)
"""

import os
import sys
from pathlib import Path
import json
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptation.trainer import TargetAdaptationTrainer, CAPAFiAdaptationEngine
from src.adaptation.mock_inputs import MockHARModel


def load_full_config(config_path: Path) -> dict:
    """Loads configuration and merges model/adaptation sections cleanly."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file missing at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Merge adaptation block with essential model and system settings
    adapt_cfg = cfg.get("adaptation", {})
    model_cfg = cfg.get("model", {})
    system_cfg = cfg.get("system", {})
    data_cfg = cfg.get("data", {})
    
    merged = dict(adapt_cfg)
    raw_device = system_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raw_device = "cpu"
    merged["device"] = raw_device
    merged["seed"] = system_cfg.get("seed", 42)
    merged["num_classes"] = model_cfg.get("num_activity_classes_K", 6)
    merged["total_classes"] = model_cfg.get("total_classes", 7)
    merged["num_slots"] = model_cfg.get("max_users_M", 2)
    merged["latent_dim"] = model_cfg.get("latent_dim_D", 256)
    merged["channels"] = data_cfg.get("channels_C", 3)
    merged["time_steps"] = data_cfg.get("window_size_T", 500)
    merged["subcarriers"] = data_cfg.get("subcarriers_S", 30)
    
    return merged


def get_target_dataloader(config: dict) -> DataLoader:
    """
    Attempts to instantiate real target dataset from splits/target_split.csv,
    falling back to synthetic streaming dataset if raw .npy files are not yet present.
    """
    splits_csv = PROJECT_ROOT / "splits" / "target_split.csv"
    data_dir = PROJECT_ROOT / "data" / "raw" / "WiMANS"
    
    # Check if real data directory and split file exist with valid .npy files
    use_real_dataset = False
    if splits_csv.exists() and data_dir.exists():
        try:
            from src.dataset import MultiUserCSIDatasetWithPreprocessing
            ds = MultiUserCSIDatasetWithPreprocessing(
                annotation_file=str(splits_csv),
                data_dir=str(data_dir),
                fixed_length=config["time_steps"]
            )
            if len(ds) > 0:
                # Test read of first element to verify .npy availability
                _ = ds[0]
                loader = DataLoader(
                    ds,
                    batch_size=config.get("batch_size", 32),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False
                )
                print(f"[Data Engine] Loaded real target dataset from {splits_csv} ({len(ds)} samples)")
                return loader
        except Exception as e:
            print(f"[Data Engine] Real dataset fallback triggered ({e}). Using synthetic target CSI streaming.")

    # Synthetic fallback: 64 samples for 2 full batches
    B_total = 64
    C = config["channels"]
    T = config["time_steps"]
    S = config["subcarriers"]
    
    x_synthetic = torch.randn(B_total, C, T, S, dtype=torch.float32)
    dataset = TensorDataset(x_synthetic)
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True
    )
    print(f"[Data Engine] Synthesized target CSI DataLoader: {B_total} samples (Shape: {list(x_synthetic.shape[1:])})")
    return loader


def load_or_create_model(config: dict) -> torch.nn.Module:
    """
    Loads Member 2's source model from checkpoints/source_model.pt if available,
    or falls back to MockHARModel matching the exact architectural contract.
    """
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "source_model.pt"
    
    model = MockHARModel(
        channels=config["channels"],
        latent_dim=config["latent_dim"],
        slots=config["num_slots"],
        total_classes=config["total_classes"]
    )
    
    if checkpoint_path.exists():
        try:
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[Model Engine] Ingested source model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"[Model Engine] Warning: Could not load {checkpoint_path} ({e}). Using initialized architecture.")
    else:
        print(f"[Model Engine] Note: checkpoints/source_model.pt not found (Member 2 stage). Using architectural contract mock.")
        
    return model


def main():
    print("==================================================================")
    print("      CAPA-Fi: Category-Aware Partial Adaptation Engine           ")
    print("    Module 3 Lead: Athishta P. A. | Source-Free SSL Execution     ")
    print("==================================================================")
    
    # 1. Load Configuration
    config_file = PROJECT_ROOT / "configs" / "default_config.yaml"
    cfg = load_full_config(config_file)
    print(f"[Config] Method: {cfg.get('method')} | Backbone LR: {cfg.get('learning_rate_backbone')}")
    print(f"[Config] Dynamic Category Filtering: {cfg.get('enable_category_filtering')} | Occupancy: {cfg.get('enable_occupancy_weighting')}")
    
    # 2. Ingest Target DataLoader
    target_loader = get_target_dataloader(cfg)
    
    # 3. Ingest Model
    model = load_or_create_model(cfg)
    
    # 4. Instantiate Adaptation Engine
    trainer = CAPAFiAdaptationEngine(model=model, config=cfg)
    
    # 5. Snapshot classifier state before adaptation
    state_before = {
        name: p.clone().detach().cpu() for name, p in model.slot_head.named_parameters()
    }
    
    # 6. Execute Multi-Epoch Adaptation Loop
    epochs = int(cfg.get("epochs", 5))  # Run 5 epochs for immediate execution test
    save_ckpt = PROJECT_ROOT / "checkpoints" / "adapted_model.pt"
    results_json = PROJECT_ROOT / "results" / "adaptation_history.json"
    
    history = trainer.adapt(
        target_loader=target_loader,
        epochs=epochs,
        save_path=str(save_ckpt),
        results_path=str(results_json),
        verbose=True
    )
    
    # 7. Verification Gate: Classifier Invariance Invariant
    print("\n---------------- Verification Gate ----------------")
    is_frozen = trainer.verify_classifier_frozen(state_before)
    assert is_frozen, "CRITICAL ERROR: Classifier head weights diverged during adaptation!"
    
    print("\n[SUCCESS] Adaptation completed successfully!")
    print(f"  • Final Total Loss: {history['loss_total'][-1]:.4f}")
    print(f"  • Final Entropy Loss: {history['loss_entropy'][-1]:.4f}")
    print(f"  • Final Diversity Loss: {history['loss_diversity'][-1]:.4f}")
    print(f"  • Final Mean Confidence: {history['mean_confidence'][-1]:.4f}")
    print(f"  • Checkpoint saved: {save_ckpt}")
    print(f"  • Metrics saved: {results_json}")
    print("==================================================================")


if __name__ == "__main__":
    main()
