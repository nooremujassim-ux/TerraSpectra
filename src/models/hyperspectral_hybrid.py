
import torch
import torch.nn as nn


class HyperspectralHybridViT(nn.Module):
    """
    Hybrid 3D-CNN + Vision Transformer for hyperspectral patches.

    Input shape:  (batch, 1, 200, 5, 5)
    Output shape: (batch, 2)
    """

    def __init__(
        self,
        num_bands=200,
        patch_size=5,
        embed_dim=64,
        num_heads=4,
        depth=2,
        num_classes=2,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.patch_size = patch_size

        # 3D-CNN extracts spatial-spectral features.
        self.cnn = nn.Sequential(
            nn.Conv3d(
                1, 8,
                kernel_size=(7, 3, 3),
                padding=(3, 1, 1),
            ),
            nn.BatchNorm3d(8),
            nn.GELU(),

            nn.Conv3d(
                8, 16,
                kernel_size=(5, 3, 3),
                padding=(2, 1, 1),
            ),
            nn.BatchNorm3d(16),
            nn.GELU(),

            # Compress spectral depth while preserving spatial positions.
            nn.AdaptiveAvgPool3d((1, patch_size, patch_size)),
        )

        # Convert CNN features into transformer tokens.
        self.embedding = nn.Linear(16, embed_dim)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_size * patch_size, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                "Expected input shape (batch, 1, bands, height, width)"
            )

        if x.shape[1] != 1:
            raise ValueError("Expected exactly one input channel")

        if x.shape[2] != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} spectral bands, "
                f"received {x.shape[2]}"
            )

        if x.shape[3] != self.patch_size or x.shape[4] != self.patch_size:
            raise ValueError(
                f"Expected spatial dimensions "
                f"{self.patch_size}x{self.patch_size}"
            )

        # (B, 1, bands, H, W) -> (B, 16, 1, H, W)
        features = self.cnn(x)

        # (B, 16, 1, H, W) -> (B, H*W, 16)
        features = features.squeeze(2)
        features = features.flatten(2).transpose(1, 2)

        # CNN features become spatial transformer tokens.
        tokens = self.embedding(features)
        tokens = tokens + self.position_embedding

        tokens = self.transformer(tokens)

        # Aggregate spatial tokens for classification.
        representation = self.norm(tokens.mean(dim=1))

        return self.classifier(representation)

