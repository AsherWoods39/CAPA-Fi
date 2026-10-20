import torch
import torch.nn.functional as F

def compute_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, M, K+1)  raw pre-softmax scores
    returns: (B, M, K+1) probabilities in [0, 1], summing to 1 along last dim
    """
    probs = F.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    return probs

def compute_sample_confidence(probs: torch.Tensor, tau_conf: float = 0.65) -> torch.Tensor:
    """
    probs: (B, M, K+1)
    returns: w_i of shape (B,) — confidence score per sample, in [0, 1]
    
    Formula: w_{i,m} = 1 - H(p_{i,m}) / log(K+1)
             w_i = mean over slots m
    """
    K_plus_1 = probs.shape[-1]
    # Per-slot entropy: (B, M)
    # H(p_i,m) = − ∑ (k=0 -> k=K) (p_i,m,k.log(p_i,m,k))
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    # Normalize to [0, 1]: high w = low entropy = confident
    # w_i,m = 1.0 - H(p_i,m) / log(K+1)
    w_slot = 1.0 - entropy / torch.log(torch.tensor(float(K_plus_1), device=probs.device, dtype=probs.dtype))
    w_slot = torch.clamp(w_slot, min=0.0, max=1.0)
    # Average across slots: (B,)
    # w_i = (1/M).∑ (m=1 -> m=M) ​(w_i,m)
    w_i = w_slot.mean(dim=-1)
    return w_i

def compute_occupancy(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: (B, M, K+1)
    returns: p_occ of shape (B,) in [0, 1]
    
    Class 0 = "No-Person". p_occ = 1 - average background prob.
    p_occ ~ 0 → empty room. p_occ ~ 1 → room is fully occupied.
    """
    # p_bg = P[:,:,0]
    p_bg = probs[:, :, 0]           # (B, M)
    # p_occ = 1.0 − (1/M) ∑ (m=1 -> m=M) P[:, m, 0]) 
    p_occ = 1.0 - p_bg.mean(dim=-1) # (B,)
    return p_occ

import math
import numpy as np
import torch.nn as nn
from typing import Optional


class CategoryMaskEngine(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        tau_conf: float = 0.15,
        momentum: float = 0.9,
        tau_thresh: float = 0.05,
        tau_temp: float = 0.04,
        tau_temp_end: float = 0.015,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.tau_conf = tau_conf
        self.momentum = kwargs.get("beta", momentum)
        self.tau_thresh = float(kwargs.get("tau_thresh", tau_thresh))
        self.tau_temp = float(kwargs.get("tau_temp", tau_temp))
        self.tau_temp_start = float(kwargs.get("tau_temp_start", self.tau_temp))
        self.tau_temp_end = float(kwargs.get("tau_temp_end", tau_temp_end))
        # Initialize EMA running class presence means to 1.0
        self.register_buffer("mu", torch.ones(num_classes, dtype=torch.float32))

    def set_temperature(self, tau_temp: float) -> None:
        """Manually overrides current sigmoid temperature."""
        self.tau_temp = float(tau_temp)

    def step_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        Dynamically anneals sigmoid temperature tau_temp over training epochs
        from tau_temp_start (0.04) down to tau_temp_end (0.015) using cosine decay.
        Sharpening class selection as target predictions cluster and become confident.
        """
        if total_epochs <= 1:
            return self.tau_temp
        progress = float(epoch - 1) / float(max(1, total_epochs - 1))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        self.tau_temp = self.tau_temp_end + (self.tau_temp_start - self.tau_temp_end) * cosine_decay
        return self.tau_temp

    @torch.no_grad()
    def update_mask(self, logits: torch.Tensor):
        if self.mu.device != logits.device:
            self.mu = self.mu.to(logits.device)

        # logits shape: [B, M, K+1]
        probs = F.softmax(logits, dim=-1)  # [B, M, K+1]

        # Calculate non-background confidence (max probability among activity classes 1..6)
        act_probs = probs[:, :, 1:]  # [B, M, K]
        max_act_probs, pred_acts = act_probs.max(dim=-1)  # [B, M]
        pred_acts = pred_acts + 1  # Shift index to 1..6

        # Gating condition: sample confidence must exceed threshold
        confident_mask = max_act_probs >= self.tau_conf  # [B, M]

        if confident_mask.any():
            confident_preds = pred_acts[confident_mask]
            # Compute batch class occurrence frequencies for confident predictions
            counts = torch.bincount(confident_preds, minlength=self.num_classes).float()
            freqs = counts / (confident_mask.sum().float() + 1e-8)

            # Update running EMA means
            self.mu = self.momentum * self.mu + (1.0 - self.momentum) * freqs.to(self.mu.device)

        # Compute dynamic category filtering gamma_k
        # Class 0 (NO_PERSON) is always retained (gamma_0 = 1.0)
        gamma = torch.ones_like(self.mu, device=logits.device)
        # Soft continuous sigmoidal gating: sigma((mu_k - tau_thresh) / tau_temp)
        # Prevents hard step extinction of minority target activities
        scaled = (self.mu[1:] - self.tau_thresh) / max(self.tau_temp, 1e-4)
        gamma[1:] = torch.sigmoid(scaled)
        return gamma, max_act_probs.mean()

    @torch.no_grad()
    def update_and_compute(
        self,
        probs: torch.Tensor,
        w_i: Optional[torch.Tensor] = None,
        tau_conf: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Backward-compatible method for callers passing probs or (probs, w_i).
        """
        if self.mu.device != probs.device:
            self.mu = self.mu.to(probs.device)

        threshold = tau_conf if tau_conf is not None else self.tau_conf
        act_probs = probs[:, :, 1:] if probs.shape[-1] > 1 else probs
        max_act_probs, pred_acts = act_probs.max(dim=-1)
        pred_acts = pred_acts + 1

        confident_mask = max_act_probs >= threshold
        if confident_mask.any():
            confident_preds = pred_acts[confident_mask]
            counts = torch.bincount(confident_preds, minlength=max(self.num_classes, 7)).float()
            if self.num_classes == 6:
                freqs = counts[1:7] / (confident_mask.sum().float() + 1e-8)
            else:
                freqs = counts[:self.num_classes] / (confident_mask.sum().float() + 1e-8)
            self.mu = self.momentum * self.mu + (1.0 - self.momentum) * freqs.to(self.mu.device)

        gamma = torch.ones_like(self.mu, device=probs.device)
        if self.num_classes == 7:
            scaled = (self.mu[1:] - self.tau_thresh) / max(self.tau_temp, 1e-4)
            gamma[1:] = torch.sigmoid(scaled)
        else:
            scaled = (self.mu - self.tau_thresh) / max(self.tau_temp, 1e-4)
            gamma = torch.sigmoid(scaled)
        return gamma


def compute_weights_and_masks(
    logits: torch.Tensor,
    mask_engine: CategoryMaskEngine,
    tau_conf: float = 0.65,
) -> tuple:
    """
    Main entry point for the masking module.
    
    Args:
        logits: (B, M, K+1) — raw classifier output
        mask_engine: stateful CategoryMaskEngine instance
        tau_conf: threshold for treating a sample as "confident"
    
    Returns:
        (w_i, gamma_k, p_occ)
        - w_i: (B,) sample confidence weights
        - gamma_k: (K,) category presence mask
        - p_occ: (B,) occupancy factors
    """
    probs = compute_probs(logits)
    w_i = compute_sample_confidence(probs, tau_conf)
    p_occ = compute_occupancy(probs)
    gamma_k = mask_engine.update_and_compute(probs, w_i, tau_conf)
    return w_i, gamma_k, p_occ