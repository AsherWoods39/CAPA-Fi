import torch
import torch.nn as nn
import torch.nn.functional as F

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

from src.adaptation.loss import CAPAFiLossEngine