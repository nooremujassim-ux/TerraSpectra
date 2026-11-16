
import torch
import torch.nn.functional as F


class HybridGradCAM:
    """
    Gradient-based explainability for HyperspectralHybridViT.

    Output shape:
    (batch, spectral_depth, height, width)
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        self.target_layer = model.cnn[-2]

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, inputs, target_class=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        inputs = inputs.detach().clone().requires_grad_(True)

        outputs = self.model(inputs)

        if target_class is None:
            target_class = outputs.argmax(dim=1)

        if isinstance(target_class, int):
            target_class = torch.full(
                (inputs.shape[0],),
                target_class,
                device=inputs.device,
                dtype=torch.long,
            )

        scores = outputs.gather(
            1, target_class.reshape(-1, 1)
        ).sum()

        scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture features.")

        weights = self.gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = F.relu(
            (weights * self.activations).sum(dim=1)
        )

        # Normalize while preserving spectral-spatial dimensions.
        original_shape = cam.shape

        cam_flat = cam.flatten(1)

        minimum = cam_flat.min(dim=1, keepdim=True)[0]
        maximum = cam_flat.max(dim=1, keepdim=True)[0]

        cam_flat = (
            (cam_flat - minimum)
            / (maximum - minimum).clamp_min(1e-8)
        )

        # Restore (batch, bands, height, width).
        cam = cam_flat.reshape(original_shape)

        return cam.detach()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
