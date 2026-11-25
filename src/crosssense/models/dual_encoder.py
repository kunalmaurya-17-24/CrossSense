"""Two independent CNN branches mapped to one shared embedding space."""

from torch import Tensor, nn

from .encoders import PatchEncoder


class DualEncoder(nn.Module):
    """Encode SAR and optical imagery with different band counts."""

    def __init__(self, sar_channels: int = 2, optical_channels: int = 4,
                 embedding_dim: int = 32, width: int = 16,
                 freeze_backbones: bool = False) -> None:
        super().__init__()
        self.sar_encoder = PatchEncoder(sar_channels, embedding_dim, width)
        self.optical_encoder = PatchEncoder(optical_channels, embedding_dim, width)
        self.sar_encoder.freeze_backbone(freeze_backbones)
        self.optical_encoder.freeze_backbone(freeze_backbones)

    def forward(self, sar: Tensor, optical: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized SAR and optical embeddings in paired batch order."""
        if sar.shape[0] != optical.shape[0]:
            raise ValueError("SAR and optical batch sizes must agree")
        return self.sar_encoder(sar), self.optical_encoder(optical)
