import torch
import torch.nn as nn


class Hyperspectral3DCNN(nn.Module):
    """
    CPU-compatible 3D CNN for hyperspectral image classification.

    Input shape:
        (batch_size, 1, spectral_bands, height, width)

    Output:
        Two classes: healthy and stressed.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)