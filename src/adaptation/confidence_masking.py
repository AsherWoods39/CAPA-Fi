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

class CategoryMaskEngine:
    """
    Maintains an EMA-smoothed estimate of which classes are present in the target domain.
    
    State:
        mu_k: (K,) — smoothed class presence score (0 = absent, 1 = always present)
    """
    def __init__(self, num_classes: int, beta: float = 0.9, tau_thresh: float = 0.05, tau_temp: float = 0.04):
        self.num_classes = num_classes
        self.beta = beta
        self.tau_thresh = tau_thresh
        self.tau_temp = tau_temp
        # Initialize with zeros (assume all absent until proven otherwise)
        self.mu = torch.zeros(num_classes)
    
    def update_and_compute(self, probs: torch.Tensor, w_i: torch.Tensor, tau_conf: float = 0.65) -> torch.Tensor:
        """
        probs: (B, M, K+1)   w_i: (B,)
        returns: gamma_k (K,) — soft mask in [0, 1]
        
        Steps:
        1. For each active class k in {1,...,K}, compute confidence-weighted average
        2. Update EMA: mu_k = beta * mu_k + (1-beta) * p_bar_k
        3. Apply sigmoid: gamma_k = sigmoid((mu_k - tau_thresh) / tau_temp)
        """
        B, M, K_plus_1 = probs.shape
        # Extract only activity class probs (skip class 0 = background)
        act_probs = probs[:, :, 1:]
        # Confidence gate: only use samples where w_i >= tau_conf
        conf_mask =(w_i >= tau_conf).float()                    # (B, )
        mask_expanded = conf_mask.unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)
        # Weighted average per class: p_bar_k = Σ_{i,m} p_{i,m,k} * I(w_i >= tau) / Σ I(w_i >= tau)
        conf_slots = conf_mask.unsqueeze(-1).expand(B, M)  # (B, M)
        dm = conf_slots.sum() + 1e-7
        nm = (act_probs * mask_expanded).sum(dim=[0, 1])
        p_bar = nm / dm # (K, )
        # EMA update (detach from computation graph — this is a statistic, not a parameter)
        with torch.no_grad():
            self.mu.copy_(self.beta * self.mu + (1.0 - self.beta) * p_bar.detach().cpu())
        # Sigmoid gating: smooth transition around tau_thresh
        mu_device = self.mu.to(probs.device)
        gamma_k = torch.sigmoid((mu_device - self.tau_thresh) / self.tau_temp)
        return gamma_k

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