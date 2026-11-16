
import pytest
import torch

from src.models.hyperspectral_hybrid import HyperspectralHybridViT
from src.evaluation.hybrid_explainability import HybridGradCAM


@pytest.fixture
def model():
    return HyperspectralHybridViT()


@pytest.fixture
def sample():
    return torch.randn(1, 1, 200, 5, 5)


def test_gradcam_output_shape(model, sample):
    explainer = HybridGradCAM(model)

    try:
        heatmap = explainer.generate(sample)
        assert heatmap.shape == (1, 200, 5, 5)
    finally:
        explainer.close()


def test_gradcam_values_are_finite(model, sample):
    explainer = HybridGradCAM(model)

    try:
        heatmap = explainer.generate(sample)
        assert torch.isfinite(heatmap).all()
    finally:
        explainer.close()


def test_gradcam_values_are_normalized(model, sample):
    explainer = HybridGradCAM(model)

    try:
        heatmap = explainer.generate(sample)
        assert heatmap.min().item() >= 0.0
        assert heatmap.max().item() <= 1.0
    finally:
        explainer.close()


def test_gradcam_supports_explicit_target_class(model, sample):
    explainer = HybridGradCAM(model)

    try:
        heatmap = explainer.generate(sample, target_class=1)
        assert heatmap.shape == (1, 200, 5, 5)
    finally:
        explainer.close()


def test_gradcam_hooks_are_removed(model, sample):
    explainer = HybridGradCAM(model)

    forward_hooks_before = len(explainer.target_layer._forward_hooks)
    backward_hooks_before = len(explainer.target_layer._backward_hooks)

    explainer.close()

    assert len(explainer.target_layer._forward_hooks) == forward_hooks_before - 1
    assert len(explainer.target_layer._backward_hooks) == backward_hooks_before - 1
