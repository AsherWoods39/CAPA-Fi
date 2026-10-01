from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.adaptation.confidence_masking import (
    CategoryMaskEngine,
    compute_probs,
    compute_sample_confidence,
    compute_weights_and_masks,
)
from src.adaptation.loss_engine import CAPAFiLossEngine


class EpochMetrics(float):
    """
    Subclass of float that preserves scalar float behavior for test assertions,
    while offering dictionary-like access to rich batch telemetry.
    """
    def __new__(cls, value: float, metrics: Optional[Dict[str, float]] = None):
        instance = super().__new__(cls, value)
        instance.metrics = metrics or {}
        return instance

    def __getitem__(self, item: str) -> float:
        return self.metrics[item]

    def __contains__(self, item: str) -> bool:
        return item in self.metrics

    def keys(self):
        return self.metrics.keys()

    def get(self, item: str, default: Any = None) -> Any:
        return self.metrics.get(item, default)


class TargetAdaptationTrainer:
    """
    CAPA-Fi Target Domain Adaptation Engine & Training Harness.
    
    Implements:
    - Source hypothesis preservation via strictly frozen classifier g_θ (∇_g_θ = 0)
    - Trainable spatial-temporal feature backbone f_θ and rotation SSL head h_ψ
    - Sample-level uncertainty weighting (w_i)
    - Dynamic Category Presence Masking (γ_k) with EMA smoothing
    - Multi-user occupancy gating (p_occ)
    - Full multi-epoch adaptation loop with checkpoint persistence and metric telemetry
    """
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.config = config
        
        # Determine device & move model (gracefully fall back to CPU if CUDA is unavailable)
        device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.model = model.to(self.device)
        
        # Freeze classifier, unfreeze backbone + rotation head
        if hasattr(model, "slot_head"):
            for param in model.slot_head.parameters():
                param.requires_grad = False
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = True
        if hasattr(model, "rot_head"):
            for param in model.rot_head.parameters():
                param.requires_grad = True
        
        # Optimizer on TRAINABLE parameters only
        trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        lr = float(config.get("learning_rate_backbone", config.get("lr", 1.0e-4)))
        weight_decay = float(config.get("weight_decay", 1.0e-4))
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Adaptation method & ablation toggles
        self.method = str(config.get("method", "capa_fi")).lower()
        self.enable_category_filtering = bool(
            config.get("enable_category_filtering", self.method == "capa_fi")
        )
        self.enable_occupancy = bool(
            config.get("enable_occupancy_weighting", self.method in ["capa_fi", "mu_shot_fi"])
        )
        
        # Loss engine
        lambda_rot = 0.0 if self.method == "shot" else float(config.get("lambda_rot", 0.5))
        self.loss_engine = CAPAFiLossEngine(
            lambda_ent=float(config.get("lambda_ent", 1.0)),
            lambda_div=float(config.get("lambda_div", 0.5)),
            lambda_rot=lambda_rot,
        )
        
        # Masking engine (stateful)
        num_classes = int(config.get("num_classes", config.get("num_activity_classes_K", 6)))
        self.num_classes = num_classes
        self.tau_conf = float(config.get("tau_conf", 0.65))
        
        self.mask_engine = CategoryMaskEngine(
            num_classes=num_classes,
            beta=float(config.get("ema_decay_beta", 0.90)),
            tau_thresh=float(config.get("tau_thresh", config.get("tau_presence", 0.05))),
            tau_temp=float(config.get("tau_temp", 0.04)),
        )
        
        self.last_epoch_metrics: Dict[str, float] = {}
    
    def adapt_epoch(self, target_loader: DataLoader) -> EpochMetrics:
        """
        Runs one adaptation epoch over the unlabeled target DataLoader.
        
        Returns:
            EpochMetrics (float subclass): average loss scalar that also exposes
            per-component telemetry via dictionary keys.
        """
        self.model.train()
        
        running_totals = {
            "loss_total": 0.0,
            "loss_entropy": 0.0,
            "loss_diversity": 0.0,
            "loss_rotation": 0.0,
            "mean_confidence": 0.0,
            "active_classes_count": 0.0,
            "occupancy_factor": 0.0,
        }
        batch_count = 0
        
        for batch in target_loader:
            # Unpack target input tensor regardless of DataLoader schema
            if isinstance(batch, (list, tuple)):
                x_target = batch[0]
            elif isinstance(batch, dict):
                x_target = batch.get("x", batch.get("x_csi", batch.get("x_target")))
            else:
                x_target = batch
            
            x_target = x_target.to(self.device)
            
            # Generate 180° inverted counterpart across temporal & subcarrier dimensions
            x_target_180 = torch.flip(x_target, dims=[-2, -1])
            
            # Forward pass: original
            logits, z = self.model(x_target)
            rot_logits_0 = self.model.forward_rotation(z)
            
            # Forward pass: rotated
            _, z_180 = self.model(x_target_180)
            rot_logits_180 = self.model.forward_rotation(z_180)
            
            # Compute masks and weights
            w_i, gamma_k, p_occ = compute_weights_and_masks(
                logits, self.mask_engine, tau_conf=self.tau_conf
            )
            
            # Apply ablation / baseline overrides if specified
            if not self.enable_category_filtering:
                gamma_k = torch.ones_like(gamma_k)
            if not self.enable_occupancy:
                p_occ = torch.ones_like(p_occ)
            
            # Compute loss
            loss_dict = self.loss_engine(
                logits, rot_logits_0, rot_logits_180, w_i, gamma_k, p_occ
            )
            
            # Backward pass & optimization step
            self.optimizer.zero_grad()
            loss_dict["loss_total"].backward()
            self.optimizer.step()
            
            # Accumulate telemetry
            batch_count += 1
            running_totals["loss_total"] += loss_dict["loss_total"].item()
            running_totals["loss_entropy"] += loss_dict["loss_entropy"].item()
            running_totals["loss_diversity"] += loss_dict["loss_diversity"].item()
            running_totals["loss_rotation"] += loss_dict["loss_rotation"].item()
            running_totals["mean_confidence"] += float(loss_dict["mean_confidence"])
            running_totals["active_classes_count"] += float(loss_dict["active_classes_count"])
            running_totals["occupancy_factor"] += float(loss_dict["occupancy_factor"])
        
        num_batches = max(batch_count, 1)
        epoch_averages = {k: v / num_batches for k, v in running_totals.items()}
        self.last_epoch_metrics = epoch_averages
        
        return EpochMetrics(epoch_averages["loss_total"], epoch_averages)
    
    def adapt(
        self,
        target_loader: DataLoader,
        epochs: Optional[int] = None,
        save_path: Optional[str] = None,
        results_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Executes multi-epoch adaptation loop, tracks convergence metrics, and persists checkpoints.
        
        Args:
            target_loader: DataLoader providing unlabeled target CSI samples
            epochs: Total epochs (defaults to config['epochs'] or 30)
            save_path: Checkpoint file destination (e.g. 'checkpoints/adapted_model.pt')
            results_path: JSON telemetry file destination (e.g. 'results/adaptation_history.json')
            verbose: Print per-epoch summary logs
            
        Returns:
            Dictionary mapping metric names to lists of per-epoch values.
        """
        total_epochs = epochs or int(self.config.get("epochs", 30))
        
        history: Dict[str, List[float]] = {
            "epoch": [],
            "loss_total": [],
            "loss_entropy": [],
            "loss_diversity": [],
            "loss_rotation": [],
            "mean_confidence": [],
            "active_classes_count": [],
            "occupancy_factor": [],
        }
        
        if verbose:
            print(f"\n=======================================================")
            print(f"  CAPA-Fi Adaptation Engine: Starting {total_epochs} Epochs  ")
            print(f"  Method: {self.method.upper()} | Device: {self.device}")
            print(f"  Category Filtering: {self.enable_category_filtering} | Occ Gating: {self.enable_occupancy}")
            print(f"=======================================================\n")
        
        for ep in range(1, total_epochs + 1):
            epoch_loss = self.adapt_epoch(target_loader)
            metrics = epoch_loss.metrics
            
            history["epoch"].append(ep)
            for k in metrics:
                if k in history:
                    history[k].append(metrics[k])
            
            if verbose:
                print(
                    f"Epoch [{ep:02d}/{total_epochs:02d}] "
                    f"Loss: {metrics['loss_total']:.4f} | "
                    f"Ent: {metrics['loss_entropy']:.4f} | "
                    f"Div: {metrics['loss_diversity']:.4f} | "
                    f"Rot: {metrics['loss_rotation']:.4f} | "
                    f"Conf: {metrics['mean_confidence']:.3f} | "
                    f"Active: {metrics['active_classes_count']:.1f}"
                )
        
        # Save adapted checkpoint if path specified
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(str(save_file))
            
        # Export convergence telemetry to JSON
        if results_path:
            res_file = Path(results_path)
            res_file.parent.mkdir(parents=True, exist_ok=True)
            with open(res_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            if verbose:
                print(f"[Results saved] → {res_file}")
                
        return history
    
    def save_checkpoint(self, path: str):
        """Persists model state dict to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(target_path))
        print(f"[Checkpoint saved] → {target_path}")
    
    def verify_classifier_frozen(self, state_before: dict) -> bool:
        """Verify g_θ weights haven't changed after adaptation."""
        for name, param in self.model.slot_head.named_parameters():
            if not torch.equal(param.data.cpu(), state_before[name].cpu()):
                print(f"[FAIL] Classifier param '{name}' changed!")
                return False
        print("[PASS] Classifier weights are bitwise identical.")
        return True
    
    # Contract helpers matching ARCHITECTURE_SPEC.md interface
    def compute_sample_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute sample confidence score w_i given probability tensor."""
        return compute_sample_confidence(probs, tau_conf=self.tau_conf)

    def compute_category_mask(
        self, batch_probs: torch.Tensor, w_i: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute category dynamic mask γ_k given batch probabilities."""
        if w_i is None:
            w_i = self.compute_sample_confidence(batch_probs)
        return self.mask_engine.update_and_compute(batch_probs, w_i, tau_conf=self.tau_conf)


# Alias conforming directly to ARCHITECTURE_SPEC.md software contracts
CAPAFiAdaptationEngine = TargetAdaptationTrainer