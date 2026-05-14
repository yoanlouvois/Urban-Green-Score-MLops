import torch
from src.training.model import UNet


def test_unet_forward_shape():
    model = UNet(in_channels=3, num_classes=8)
    x = torch.randn(1, 3, 512, 512)

    y = model(x)

    assert y.shape == (1, 8, 512, 512)