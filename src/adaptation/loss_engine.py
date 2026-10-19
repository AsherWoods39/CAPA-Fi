import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.adaptation.confidence_masking import compute_probs

def compute_entropy_loss(probs: torch.Tensor, w_i: torch.Tensor) -> torch.Tensor:
    """
    Confidence-weighted conditional entropy.
    
    probs: (B, M, K+1)   w_i: (B,)
    returns: scalar loss tensor
    
    L_ent = - (1/B*M) * Σ_i Σ_m  w_i * Σ_k  p_{i,m,k} * log(p_{i,m,k})
    """
    # Per-slot entropy: (B, M)
    per_slot_entropy = -(probs * torch.log(probs)).sum(dim=-1)

    # Weight by sample confidence: w_i applied per sample (broadcast over M)
    weighted_entropy = w_i.unsqueeze(-1) * per_slot_entropy   # (B, M)
    
    return weighted_entropy.mean()

def compute_masked_diversity_loss(probs: torch.Tensor, gamma_k: torch.Tensor, p_occ: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Occupancy-gated, category-masked marginal diversity.
        
    probs: (B, M, K+1)   gamma_k: (K,)   p_occ: (B,)
    returns: scalar loss tensor (to be SUBTRACTED from total loss)
        
    p_hat_k = batch mean of class k predictions (over B*M samples)
    L_div = p_occ.mean() * Σ_k  gamma_k * p_hat_k * log(p_hat_k + eps)
    """
    # Average over batch and slots, only for activity classes (skip class 0)
    act_probs = probs[:, :, 1:]         # (B, M, K)
    p_hat = act_probs.mean(dim=[0, 1])  # (K, ) - batch marginal
    # Apply category mask and occupancy gate
    # Calculate positive marginal entropy: H(p_hat) = - sum(p_hat * log(p_hat))
    # Masked by gamma_k
    masked_entropy = -(gamma_k * p_hat * torch.log(p_hat + eps)).sum()
    occ_scalar = p_occ.mean()

    return occ_scalar * masked_entropy

def compute_rotation_loss(rot_logits_0: torch.Tensor, rot_logits_180: torch.Tensor) -> torch.Tensor:
    """
    Self-supervised rotation classification loss.
        
    rot_logits_0:   (B, 2) — logits for original (unrotated) CSI
    rot_logits_180: (B, 2) — logits for 180°-rotated CSI
    returns: scalar loss
        
    Labels: original=0 (class 0), rotated=1 (class 1)
    """
    B = rot_logits_0.shape[0]
        
    # Labels: original samples are class 0, rotated samples are class 1
    labels_0 = torch.zeros(B, dtype=torch.long, device=rot_logits_0.device)
    labels_180 = torch.ones(B, dtype=torch.long, device=rot_logits_180.device)

    loss_0 = F.cross_entropy(rot_logits_0, labels_0)
    loss_180 = F.cross_entropy(rot_logits_180, labels_180)

    return (loss_0 + loss_180) / 2.0

class CAPAFiLossEngine(nn.Module):
    def __init__(self, lambda_ent: float = 1.0, lambda_div: float = 1.0, lambda_rot: float = 0.5):
        super().__init__()
        self.lambda_ent = lambda_ent
        self.lambda_div = lambda_div
        self.lambda_rot = lambda_rot

    def forward(
        self,
        logits: torch.Tensor,
        rot_logits: torch.Tensor,
        rot_targets: Optional[torch.Tensor] = None,
        category_mask: Optional[torch.Tensor] = None,
        occupancy_weight: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        # logits shape: [B, M, K+1] where K+1 = 7 (Class 0 = NO_PERSON, Classes 1..6 = Activities)
        probs = F.softmax(logits, dim=-1)  # [B, M, K+1]

        # Check for legacy 6-argument call signature: (logits, rot_logits_0, rot_logits_180, w_i, gamma_k, p_occ)
        if (
            rot_targets is not None
            and torch.is_floating_point(rot_targets)
            and rot_targets.ndim == 2
        ):
            rot_logits_0 = rot_logits
            rot_logits_180 = rot_targets
            B = rot_logits_0.shape[0]
            rot_logits_comb = torch.cat([rot_logits_0, rot_logits_180], dim=0)
            rot_targets_comb = torch.cat([
                torch.zeros(B, dtype=torch.long, device=rot_logits_0.device),
                torch.ones(B, dtype=torch.long, device=rot_logits_180.device),
            ], dim=0)
            rot_logits = rot_logits_comb
            rot_targets = rot_targets_comb

            # Check if 4th arg was w_i and 5th was gamma_k
            if category_mask is not None and category_mask.ndim == 1 and category_mask.shape[0] == logits.shape[0]:
                category_mask = occupancy_weight
                occupancy_weight = args[0] if len(args) > 0 else kwargs.get("p_occ", None)
        elif len(args) >= 1 and rot_targets is None:
            pass

        # Ensure all incoming conditioning tensors match logits device
        if category_mask is not None:
            category_mask = category_mask.to(device=logits.device, dtype=probs.dtype)
        if occupancy_weight is not None:
            occupancy_weight = occupancy_weight.to(device=logits.device, dtype=probs.dtype)

        # 1. Slot Entropy Loss (masked by category mask if provided)
        log_probs = F.log_softmax(logits, dim=-1)
        slot_entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, M]

        if category_mask is not None:
            # Align mask shape if passed as 6-class vector (activities only)
            if category_mask.shape[-1] == probs.shape[-1] - 1:
                prefix = torch.ones(1, device=logits.device, dtype=category_mask.dtype)
                category_mask = torch.cat([prefix, category_mask], dim=-1)

            # Mask out absent classes during entropy calculation
            masked_probs = probs * category_mask.unsqueeze(0).unsqueeze(0)
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            slot_entropy = -torch.sum(masked_probs * torch.log(masked_probs + 1e-8), dim=-1)

        loss_ent = slot_entropy.mean()

        # 2. Diversity Loss (Marginal Entropy across batch and slots)
        avg_probs = probs.mean(dim=(0, 1))  # [K+1]
        loss_div = torch.sum(avg_probs * torch.log(avg_probs + 1e-8))  # Negative entropy

        if occupancy_weight is not None:
            loss_div = occupancy_weight.mean() * loss_div

        # 3. Self-Supervised Rotation Loss
        if rot_targets is not None:
            if rot_logits is not None:
                rot_targets = rot_targets.to(rot_logits.device)
                loss_rot = F.cross_entropy(rot_logits, rot_targets).to(logits.device)
            else:
                loss_rot = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss_rot = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Total CAPA-Fi Target Adaptation Loss
        total_loss = (self.lambda_ent * loss_ent) + (self.lambda_div * loss_div) + (self.lambda_rot * loss_rot)

        # Telemetry metrics
        act_probs = probs[:, :, 1:]
        max_act_probs = act_probs.max(dim=-1)[0]
        mean_conf = max_act_probs.mean().item()

        if category_mask is not None:
            active_cnt = (category_mask[1:] > 0.5).sum().item() if category_mask.shape[0] == 7 else (category_mask > 0.5).sum().item()
        else:
            active_cnt = float(probs.shape[-1] - 1)

        occ_fac = occupancy_weight.mean().item() if occupancy_weight is not None else 1.0

        return {
            "loss": total_loss,
            "loss_ent": loss_ent,
            "loss_div": loss_div,
            "loss_rot": loss_rot,
            # Backward-compatible keys
            "loss_total": total_loss,
            "loss_entropy": loss_ent,
            "loss_diversity": loss_div,
            "loss_rotation": loss_rot,
            "mean_confidence": mean_conf,
            "active_classes_count": active_cnt,
            "occupancy_factor": occ_fac,
        }
