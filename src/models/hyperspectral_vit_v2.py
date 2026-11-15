
import torch
import torch.nn as nn


class HyperspectralViTV2(nn.Module):
    """
    Lightweight hyperspectral Vision Transformer V2.

    Uses mean pooling of spatial tokens instead of a CLS token.
    Input: (batch, 1, bands, patch_size, patch_size)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_bands=200,
        patch_size=5,
        embed_dim=64,
        num_heads=4,
        depth=2,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        if num_bands < 1 or patch_size < 1:
            raise ValueError("Bands and patch size must be positive.")

        if embed_dim < 1 or num_heads < 1:
            raise ValueError("Embedding dimension and heads must be positive.")

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        self.num_bands = num_bands
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size

        self.embedding = nn.Linear(num_bands, embed_dim)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                "Input must have shape (batch, 1, bands, height, width)."
            )

        batch_size, channels, bands, height, width = x.shape

        if channels != 1:
            raise ValueError("Input must have exactly one channel.")

        if bands != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} spectral bands, got {bands}."
            )

        if height != self.patch_size or width != self.patch_size:
            raise ValueError(
                f"Expected spatial dimensions "
                f"{self.patch_size}x{self.patch_size}, "
                f"got {height}x{width}."
            )

        x = x.squeeze(1)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.num_tokens, bands)

        x = self.embedding(x)
        x = x + self.position_embedding

        x = self.transformer(x)

        # Aggregate all spatial tokens.
        x = x.mean(dim=1)
        x = self.norm(x)

        return self.classifier(x)
