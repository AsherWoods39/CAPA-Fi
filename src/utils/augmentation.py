"""
CAPA-Fi: CSI Data Augmentation Module (Member 2)

Implements time-series augmentation strategies for CSI tensors.
All augmentations preserve the input shape [B, C, T, S].

Augmentations included:
  - Jittering: additive Gaussian noise
  - Scaling: multiplicative random factor
  - Slice shuffling: random permutation of temporal segments
  - Magnitude warping: smooth random distortion of amplitude
  - Window warping: time-axis stretching/compression of random segments

NOT included:
  - 180° rotation transform (Member 1's responsibility)
"""

from typing import Optional

import torch
import torch.nn.functional as F


def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """
    Add small Gaussian noise to the CSI tensor.

    Args:
        x: Input tensor [B, C, T, S]
        sigma: Standard deviation of noise

    Returns:
        Augmented tensor [B, C, T, S]
    """
    return x + sigma * torch.randn_like(x)


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Multiply each sample by a random scaling factor per channel.

    Args:
        x: Input tensor [B, C, T, S]
        sigma: Standard deviation of the scaling factor (centered at 1.0)

    Returns:
        Augmented tensor [B, C, T, S]
    """
    B, C, T, S = x.shape
    # One scale factor per (batch, channel), broadcast over time and subcarriers
    scale_factors = 1.0 + sigma * torch.randn(B, C, 1, 1, device=x.device, dtype=x.dtype)
    return x * scale_factors


def slice_shuffle(x: torch.Tensor, num_segments: int = 10) -> torch.Tensor:
    """
    Randomly permute short temporal segments of the CSI tensor.

    Divides the time axis into `num_segments` chunks and shuffles their order.
    This disrupts temporal ordering while preserving local patterns.

    Args:
        x: Input tensor [B, C, T, S]
        num_segments: Number of temporal segments to shuffle

    Returns:
        Augmented tensor [B, C, T, S]
    """
    B, C, T, S = x.shape
    segment_len = T // num_segments
    remainder = T - segment_len * num_segments

    # Split into segments along the time axis
    segments = []
    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len
        segments.append(x[:, :, start:end, :])

    # Handle remainder (append to last segment)
    if remainder > 0:
        segments[-1] = torch.cat(
            [segments[-1], x[:, :, num_segments * segment_len:, :]],
            dim=2,
        )

    # Shuffle segment order (same permutation for all samples in batch)
    perm = torch.randperm(len(segments))
    shuffled = [segments[i] for i in perm]

    return torch.cat(shuffled, dim=2)


def magnitude_warp(
    x: torch.Tensor,
    sigma: float = 0.2,
    num_knots: int = 4,
) -> torch.Tensor:
    """
    Apply smooth random distortion of amplitude via cubic spline-like warping.

    Generates a smooth warping curve along the time axis and multiplies element-wise.

    Args:
        x: Input tensor [B, C, T, S]
        sigma: Standard deviation of warping knot values (centered at 1.0)
        num_knots: Number of control points for the warping curve

    Returns:
        Augmented tensor [B, C, T, S]
    """
    B, C, T, S = x.shape

    # Generate random knot values [B, C, num_knots]
    knot_values = 1.0 + sigma * torch.randn(
        B, C, num_knots, device=x.device, dtype=x.dtype
    )

    # Interpolate knots to full time axis length [B, C, T]
    # Use linear interpolation via F.interpolate (expects [B, C, L])
    warp_curve = F.interpolate(
        knot_values, size=T, mode="linear", align_corners=True
    )

    # Broadcast over subcarriers: [B, C, T, 1]
    warp_curve = warp_curve.unsqueeze(-1)

    return x * warp_curve


def window_warp(
    x: torch.Tensor,
    window_ratio: float = 0.1,
    warp_scale_range: tuple = (0.5, 2.0),
) -> torch.Tensor:
    """
    Time-axis stretching/compression of a random window segment.

    Selects a random window in the time axis, warps its length, and
    adjusts the surrounding signal to maintain the original total length.

    Args:
        x: Input tensor [B, C, T, S]
        window_ratio: Fraction of time axis to select as the warp window
        warp_scale_range: (min, max) scale factor for the window length

    Returns:
        Augmented tensor [B, C, T, S]
    """
    B, C, T, S = x.shape
    window_len = max(1, int(T * window_ratio))

    # Random window start position (same for all samples in batch)
    max_start = T - window_len
    if max_start <= 0:
        return x
    start = torch.randint(0, max_start, (1,)).item()
    end = start + window_len

    # Random warp scale
    scale_min, scale_max = warp_scale_range
    warp_scale = scale_min + (scale_max - scale_min) * torch.rand(1).item()
    new_window_len = max(1, int(window_len * warp_scale))

    # Extract window and warp via interpolation
    # [B, C, window_len, S] → permute to [B*S, C, window_len] for 1D interpolation
    window = x[:, :, start:end, :]  # [B, C, window_len, S]
    window = window.permute(0, 3, 1, 2).reshape(B * S, C, window_len)
    warped_window = F.interpolate(
        window, size=new_window_len, mode="linear", align_corners=True
    )
    warped_window = warped_window.reshape(B, S, C, new_window_len).permute(0, 2, 3, 1)

    # Reconstruct: before + warped_window + after
    before = x[:, :, :start, :]
    after = x[:, :, end:, :]
    result = torch.cat([before, warped_window, after], dim=2)  # [B, C, T', S]

    # Resize back to original T if length changed
    if result.shape[2] != T:
        # [B, C, T', S] → permute to [B*S, C, T'] for interpolation
        result_perm = result.permute(0, 3, 1, 2).reshape(B * S, C, -1)
        result_perm = F.interpolate(
            result_perm, size=T, mode="linear", align_corners=True
        )
        result = result_perm.reshape(B, S, C, T).permute(0, 2, 3, 1)

    return result


class CSIAugmentor:
    """
    Composable CSI augmentation pipeline.

    Applies a configurable set of augmentations to CSI tensors [B, C, T, S].
    All augmentations are optional and can be toggled independently.

    Example:
        augmentor = CSIAugmentor(jitter_sigma=0.05, scaling_sigma=0.1)
        x_aug = augmentor(x)  # [B, C, T, S] → [B, C, T, S]
    """

    def __init__(
        self,
        enable_jitter: bool = True,
        jitter_sigma: float = 0.05,
        enable_scaling: bool = True,
        scaling_sigma: float = 0.1,
        enable_slice_shuffle: bool = False,
        num_segments: int = 10,
        enable_magnitude_warp: bool = True,
        magnitude_sigma: float = 0.2,
        magnitude_knots: int = 4,
        enable_window_warp: bool = False,
        window_ratio: float = 0.1,
        warp_scale_range: tuple = (0.5, 2.0),
        augment_prob: float = 0.8,
    ):
        self.enable_jitter = enable_jitter
        self.jitter_sigma = jitter_sigma
        self.enable_scaling = enable_scaling
        self.scaling_sigma = scaling_sigma
        self.enable_slice_shuffle = enable_slice_shuffle
        self.num_segments = num_segments
        self.enable_magnitude_warp = enable_magnitude_warp
        self.magnitude_sigma = magnitude_sigma
        self.magnitude_knots = magnitude_knots
        self.enable_window_warp = enable_window_warp
        self.window_ratio = window_ratio
        self.warp_scale_range = warp_scale_range
        self.augment_prob = augment_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation pipeline.

        Args:
            x: Input tensor [B, C, T, S]

        Returns:
            Augmented tensor [B, C, T, S] (same shape guaranteed)
        """
        # Skip augmentation with probability (1 - augment_prob)
        if torch.rand(1).item() > self.augment_prob:
            return x

        if self.enable_jitter:
            x = jitter(x, sigma=self.jitter_sigma)

        if self.enable_scaling:
            x = scaling(x, sigma=self.scaling_sigma)

        if self.enable_slice_shuffle:
            x = slice_shuffle(x, num_segments=self.num_segments)

        if self.enable_magnitude_warp:
            x = magnitude_warp(
                x, sigma=self.magnitude_sigma, num_knots=self.magnitude_knots
            )

        if self.enable_window_warp:
            x = window_warp(
                x,
                window_ratio=self.window_ratio,
                warp_scale_range=self.warp_scale_range,
            )

        return x
