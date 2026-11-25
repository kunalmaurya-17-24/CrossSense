"""Compact cross-modal encoders and reproducible contrastive training."""

from .dual_encoder import DualEncoder
from .losses import symmetric_info_nce

__all__ = ["DualEncoder", "symmetric_info_nce"]
