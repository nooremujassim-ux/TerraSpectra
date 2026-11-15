
import torch
import torch.nn as nn


class HyperspectralViT(nn.Module):
    """
    Lightweight Vision Transformer for hyperspectral patch classification.

    Input shape:
        (batch_size, 1, spectral_bands, patch_size, patch_size)

    Default input:
        (batch_size, 1, 200, 5, 5)

    Output:
        Two classes: healthy (0) and stressed (1).

    Each spatial pixel becomes a token containing its spectral features.
    Transformer self-attention learns relationships between these tokens.
    """

    def __init__(
        self,
        num_bands=200,
        patch_size=5,
        embed_dim=64,
        num_heads=4,
        depth=2,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        if num_bands < 1:
            raise ValueError("num_bands must be positive.")

        if patch_size < 1:
            raise ValueError("patch_size must be positive.")

        if embed_dim < 1 or num_heads < 1:
            raise ValueError("embed_dim and num_heads must be positive.")

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.num_bands = num_bands
        self.patch_size = patch_size

        self.num_tokens = patch_size * patch_size

        # Convert each pixel's spectral signature into an embedding.
        self.embedding = nn.Linear(num_bands, embed_dim)

        # Learnable classification token and spatial position embeddings.
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens + 1, embed_dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # Transformer encoder with multi-head self-attention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

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

        # Convert (B, 1, bands, H, W) into (B, H*W, bands).
        x = x.squeeze(1)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.num_tokens, bands)

        # Spectral feature embedding.
        x = self.embedding(x)

        # Add classification token.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add spatial position information.
        x = x + self.position_embedding

        # Learn relationships between tokens using self-attention.
        x = self.transformer(x)

        # Use the classification token for prediction.
        x = self.norm(x[:, 0])

        return self.classifier(x)
