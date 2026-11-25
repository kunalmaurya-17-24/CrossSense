"""Bidirectional contrastive objective with geographic false-negative masking."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def symmetric_info_nce(sar: Tensor, optical: Tensor, temperature: float = 0.1,
                       coords: Tensor | None = None,
                       exclusion_m: float = 0.0, patch_extent_m: float = 0.0) -> Tensor:
    """Use diagonal positives and exclude nearby off-diagonal pairs from negatives.

    Coordinates must share a metric projected CRS. ``patch_extent_m`` is the
    common square patch width; distance is measured between footprint edges.
    An all-masked row contributes zero loss rather than generating NaNs.
    """
    if sar.ndim != 2 or sar.shape != optical.shape or len(sar) == 0:
        raise ValueError("Contrastive embeddings must have matching nonempty [batch, dimension] shapes")
    if (not math.isfinite(temperature) or temperature <= 0 or
            not math.isfinite(exclusion_m) or exclusion_m < 0 or
            not math.isfinite(patch_extent_m) or patch_extent_m < 0):
        raise ValueError("Temperature must be positive and exclusion distance nonnegative")
    if not torch.isfinite(sar).all() or not torch.isfinite(optical).all():
        raise ValueError("Contrastive embeddings must be finite")
    logits = F.normalize(sar, dim=1) @ F.normalize(optical, dim=1).T / temperature
    if coords is not None:
        if coords.shape != (len(sar), 2) or not torch.isfinite(coords).all():
            raise ValueError("Coordinates must have finite [batch, 2] shape")
        centers = coords.float()
        deltas = (centers[:, None, :] - centers[None, :, :]).abs()
        footprint_gaps = (deltas - patch_extent_m).clamp_min(0)
        nearby = torch.linalg.vector_norm(footprint_gaps, dim=-1) <= exclusion_m
        nearby.fill_diagonal_(False)
        logits = logits.masked_fill(nearby, float("-inf"))
    target = torch.arange(len(sar), device=sar.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
