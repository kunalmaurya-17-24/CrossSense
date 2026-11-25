import pytest
import torch

from crosssense.models.dual_encoder import DualEncoder
from crosssense.models.losses import symmetric_info_nce


def test_dual_encoder_shapes_norms_and_freeze():
    torch.manual_seed(42)
    model = DualEncoder(2, 4, 12, 4, freeze_backbones=True)
    sar, optical = model(torch.rand(3, 2, 16, 16), torch.rand(3, 4, 16, 16))
    assert sar.shape == optical.shape == (3, 12)
    torch.testing.assert_close(sar.norm(dim=1), torch.ones(3))
    torch.testing.assert_close(optical.norm(dim=1), torch.ones(3))
    assert all(not p.requires_grad for p in model.sar_encoder.backbone.parameters())
    assert all(p.requires_grad for p in model.sar_encoder.projection.parameters())
    model.sar_encoder.freeze_backbone(False)
    assert all(p.requires_grad for p in model.sar_encoder.backbone.parameters())


def test_model_wrong_channels_nan_and_batch():
    model = DualEncoder(width=4)
    with pytest.raises(ValueError, match="channels"):
        model(torch.zeros(2, 3, 8, 8), torch.zeros(2, 4, 8, 8))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((2, 2, 8, 8), float("nan")), torch.zeros(2, 4, 8, 8))
    with pytest.raises(ValueError, match="batch"):
        model(torch.zeros(2, 2, 8, 8), torch.zeros(3, 4, 8, 8))


def test_loss_aligned_better_than_shuffled_and_finite_gradient():
    sar = torch.eye(4, requires_grad=True)
    aligned = symmetric_info_nce(sar, torch.eye(4), .1)
    shuffled = symmetric_info_nce(sar, torch.eye(4).roll(1, 0), .1)
    assert 0 <= aligned < .001 < shuffled
    aligned.backward()
    assert torch.isfinite(sar.grad).all()


def test_loss_geographic_exclusion_all_nearby_is_finite_zero():
    vectors = torch.eye(3, requires_grad=True)
    loss = symmetric_info_nce(vectors, vectors.roll(1, 0), .1, torch.zeros(3, 2), 500)
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(vectors.grad).all()


def test_loss_excludes_overlapping_large_footprints():
    # Centers are 1500m apart, but 2560m patches overlap.
    vectors = torch.eye(2)
    coords = torch.tensor([[0., 0.], [1500., 0.]])
    unmasked = symmetric_info_nce(vectors, vectors.roll(1, 0), .1, coords, 500)
    masked = symmetric_info_nce(vectors, vectors.roll(1, 0), .1, coords, 500, 2560)
    assert unmasked > 1 and masked == 0


@pytest.mark.parametrize("temperature", [0, -1, float("nan")])
def test_loss_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        symmetric_info_nce(torch.eye(2), torch.eye(2), temperature)
