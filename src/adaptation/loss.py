import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
            # Fallback if positional args provided
            pass

        # 1. Slot Entropy Loss (masked by category mask if provided)
        log_probs = F.log_softmax(logits, dim=-1)
        slot_entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, M]

        if category_mask is not None:
            # Align mask shape if passed as 6-class vector (activities only)
            if category_mask.shape[-1] == probs.shape[-1] - 1:
                prefix = torch.ones(1, device=category_mask.device, dtype=category_mask.dtype)
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
            loss_rot = F.cross_entropy(rot_logits, rot_targets)
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

