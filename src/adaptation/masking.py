import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CategoryMaskEngine(nn.Module):
    def __init__(self, num_classes: int = 7, tau_conf: float = 0.15, momentum: float = 0.9, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.tau_conf = tau_conf
        self.momentum = kwargs.get("beta", momentum)
        # Initialize EMA running class presence means to 1.0
        self.register_buffer("mu", torch.ones(num_classes, dtype=torch.float32))

    @torch.no_grad()
    def update_mask(self, logits: torch.Tensor):
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
        gamma = torch.ones_like(self.mu)
        gamma[1:] = torch.where(self.mu[1:] >= (self.mu[1:].max() * 0.10), 1.0, 0.0)
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

        gamma = torch.ones_like(self.mu)
        if self.num_classes == 7:
            gamma[1:] = torch.where(self.mu[1:] >= (self.mu[1:].max() * 0.10), 1.0, 0.0)
        else:
            gamma = torch.where(self.mu >= (self.mu.max() * 0.10), 1.0, 0.0)
        return gamma
