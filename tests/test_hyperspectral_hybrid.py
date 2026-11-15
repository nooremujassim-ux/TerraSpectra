
import pytest
import torch

from src.models.hyperspectral_hybrid import HyperspectralHybridViT


@pytest.fixture
def model():
    return HyperspectralHybridViT()


def test_hybrid_output_shape(model):
    x = torch.randn(2, 1, 200, 5, 5)
    output = model(x)

    assert output.shape == (2, 2)


def test_hybrid_single_sample(model):
    x = torch.randn(1, 1, 200, 5, 5)
    output = model(x)

    assert output.shape == (1, 2)


def test_hybrid_runs_on_cpu(model):
    x = torch.randn(2, 1, 200, 5, 5)
    output = model(x)

    assert output.device.type == "cpu"


def test_hybrid_parameter_count(model):
    parameters = sum(p.numel() for p in model.parameters())

    assert parameters > 0


def test_hybrid_rejects_wrong_number_of_bands(model):
    x = torch.randn(2, 1, 100, 5, 5)

    with pytest.raises(ValueError):
        model(x)


def test_hybrid_rejects_wrong_spatial_dimensions(model):
    x = torch.randn(2, 1, 200, 7, 7)

    with pytest.raises(ValueError):
        model(x)


def test_hybrid_rejects_invalid_input_dimensions(model):
    x = torch.randn(2, 200, 5, 5)

    with pytest.raises(ValueError):
        model(x)
