from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from src.adaptation.confidence_masking import (
    CategoryMaskEngine,
    compute_weights_and_masks,
)
from src.adaptation.loss_engine import CAPAFiLossEngine


class TargetAdaptationTrainer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Freeze classifier, unfreeze backbone + rotation head
        for param in model.slot_head.parameters():
            param.requires_grad = False
        for param in model.backbone.parameters():
            param.requires_grad = True
        for param in model.rot_head.parameters():
            param.requires_grad = True
        
        # Optimizer on TRAINABLE params only
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config["learning_rate_backbone"],
            weight_decay=config["weight_decay"]
        )
        
        # Loss engine
        self.loss_engine = CAPAFiLossEngine(
            lambda_ent=config["lambda_ent"],
            lambda_div=config["lambda_div"],
            lambda_rot=config["lambda_rot"],
        )
        
        # Masking engine (stateful)
        self.mask_engine = CategoryMaskEngine(
            num_classes=config["num_classes"],
            beta=config["ema_decay_beta"],
            tau_thresh=config["tau_thresh"],
            tau_temp=config["tau_temp"],
        )
    
    def adapt_epoch(self, target_loader):
        self.model.train()
        epoch_loss = 0.0
        
        for batch in target_loader:
            x_target = batch[0]           # (B, C, T, S)
            
            # Generate 180° rotated counterpart
            x_target_180 = torch.flip(x_target, dims=[-2, -1])
            
            # Forward pass: original
            logits, z = self.model(x_target)
            rot_logits_0 = self.model.forward_rotation(z)
            
            # Forward pass: rotated
            _, z_180 = self.model(x_target_180)
            rot_logits_180 = self.model.forward_rotation(z_180)
            
            # Compute masks and weights
            w_i, gamma_k, p_occ = compute_weights_and_masks(
                logits, self.mask_engine, tau_conf=self.config["tau_conf"]
            )
            
            # Compute loss
            loss_dict = self.loss_engine(
                logits, rot_logits_0, rot_logits_180, w_i, gamma_k, p_occ
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict["loss_total"].backward()
            self.optimizer.step()
            
            epoch_loss += loss_dict["loss_total"].item()
        
        return epoch_loss / len(target_loader)
    
    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"[Checkpoint saved] → {path}")
    
    def verify_classifier_frozen(self, state_before: dict) -> bool:
        """Verify g_θ weights haven't changed after adaptation."""
        for name, param in self.model.slot_head.named_parameters():
            if not torch.equal(param.data, state_before[name]):
                print(f"[FAIL] Classifier param '{name}' changed!")
                return False
        print("[PASS] Classifier weights are bitwise identical.")
        return True