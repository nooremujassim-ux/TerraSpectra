
import pytest
import torch

from src.models.hyperspectral_vit import HyperspectralViT
from src.models.hyperspectral_vit_v2 import HyperspectralViTV2


@pytest.mark.parametrize(
    "model_class",
    [HyperspectralViT, HyperspectralViTV2],
)
def test_vit_output_shape(model_class):
    model = model_class()
    inputs = torch.randn(2, 1, 200, 5, 5)

    outputs = model(inputs)

    assert outputs.shape == (2, 2)


@pytest.mark.parametrize(
    "model_class",
    [HyperspectralViT, HyperspectralViTV2],
)
def test_vit_supports_single_sample(model_class):
    model = model_class()
    inputs = torch.randn(1, 1, 200, 5, 5)

    outputs = model(inputs)

    assert outputs.shape == (1, 2)


@pytest.mark.parametrize(
    "model_class",
    [HyperspectralViT, HyperspectralViTV2],
)
def test_vit_runs_on_cpu(model_class):
    model = model_class()
    inputs = torch.randn(2, 1, 200, 5, 5)

    outputs = model(inputs)

    assert next(model.parameters()).device.type == "cpu"
    assert outputs.device.type == "cpu"


@pytest.mark.parametrize(
    "model_class",
    [HyperspectralViT, HyperspectralViTV2],
)
def test_vit_rejects_wrong_number_of_bands(model_class):
    model = model_class()
    inputs = torch.randn(2, 1, 100, 5, 5)

    with pytest.raises(ValueError, match="spectral bands"):
        model(inputs)


@pytest.mark.parametrize(
    "model_class",
    [HyperspectralViT, HyperspectralViTV2],
)
def test_vit_rejects_wrong_spatial_dimensions(model_class):
    model = model_class()
    inputs = torch.randn(2, 1, 200, 7, 7)

    with pytest.raises(ValueError, match="spatial dimensions"):
        model(inputs)


@pytest.mark.parametrize(
    "model_class",
    [HyperspectralViT, HyperspectralViTV2],
)
def test_vit_rejects_invalid_input_dimensions(model_class):
    model = model_class()
    inputs = torch.randn(2, 200, 5, 5)

    with pytest.raises(ValueError, match="Input must have shape"):
        model(inputs)
