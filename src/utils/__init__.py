"""
CAPA-Fi: Utilities Package

Provides CSI data augmentation strategies for source training.
"""

from src.utils.augmentation import (
    CSIAugmentor,
    jitter,
    scaling,
    slice_shuffle,
    magnitude_warp,
    window_warp,
)

__all__ = [
    "CSIAugmentor",
    "jitter",
    "scaling",
    "slice_shuffle",
    "magnitude_warp",
    "window_warp",
]
