
import pytest
import torch

from src.models.hyperspectral_vit import HyperspectralViT


def test_vit_output_shape():
    model = HyperspectralViT()
    inputs = torch.randn(2, 1, 200, 5, 5)

    outputs = model(inputs)

    assert outputs.shape == (2, 2)


def test_vit_supports_single_sample():
    model = HyperspectralViT()
    inputs = torch.randn(1, 1, 200, 5, 5)

    outputs = model(inputs)

    assert outputs.shape == (1, 2)


def test_vit_runs_on_cpu():
    model = HyperspectralViT()
    inputs = torch.randn(2, 1, 200, 5, 5)

    outputs = model(inputs)

    assert next(model.parameters()).device.type == "cpu"
    assert outputs.device.type == "cpu"


def test_vit_rejects_wrong_number_of_bands():
    model = HyperspectralViT()
    inputs = torch.randn(2, 1, 100, 5, 5)

    with pytest.raises(ValueError, match="spectral bands"):
        model(inputs)


def test_vit_rejects_wrong_spatial_dimensions():
    model = HyperspectralViT()
    inputs = torch.randn(2, 1, 200, 7, 7)

    with pytest.raises(ValueError, match="spatial dimensions"):
        model(inputs)


def test_vit_rejects_invalid_input_dimensions():
    model = HyperspectralViT()
    inputs = torch.randn(2, 200, 5, 5)

    with pytest.raises(ValueError, match="Input must have shape"):
        model(inputs)
