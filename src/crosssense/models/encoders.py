"""Small sensor-specific image encoders; no external weights are downloaded."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEncoder(nn.Module):
    """Encode an explicitly selected set of bands into a unit length vector."""

    def __init__(self, channels: int, embedding_dim: int = 32, width: int = 16) -> None:
        super().__init__()
        if min(channels, embedding_dim, width) < 1:
            raise ValueError("Encoder channels, embedding dimension and width must be positive")
        self.channels = channels
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.projection = nn.Linear(width * 2, embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Return normalized embeddings for finite BCHW sensor arrays."""
        if images.ndim != 4 or images.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW input with {self.channels} channels, got {tuple(images.shape)}")
        if min(images.shape[2:]) < 4 or not torch.isfinite(images).all():
            raise ValueError("Images must be finite and at least 4 by 4 pixels")
        return F.normalize(self.projection(self.backbone(images)), dim=-1)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or enable backbone gradients while retaining a trainable projection."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not freeze)
