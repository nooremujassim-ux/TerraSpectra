
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    """
    Dataset for extracting labelled spatial-spectral patches
    from a hyperspectral GeoTIFF.

    Input shape per sample: (1, bands, patch_size, patch_size)
    Label: 0 = healthy, 1 = stressed
    """

    def __init__(
        self,
        image_path,
        label_path,
        patch_size=5
    ):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd number.")

        with rasterio.open(image_path) as src:
            self.image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            self.labels = src.read(1)

        if self.image.ndim != 3:
            raise ValueError("Hyperspectral image must have 3 dimensions.")

        if self.image.shape[1:] != self.labels.shape:
            raise ValueError("Image and label dimensions do not match.")

        if not np.isfinite(self.image).all():
            raise ValueError("Image contains NaN or infinite values.")

        if not np.isin(np.unique(self.labels), [0, 1]).all():
            raise ValueError("Labels must contain only 0 (healthy) and 1 (stressed).")

        self.patch_size = patch_size
        self.padding = patch_size // 2

        self.padded_image = np.pad(
            self.image,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding)
            ),
            mode="reflect"
        )

        self.coordinates = np.argwhere(
            np.isin(self.labels, [0, 1])
        )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, col = self.coordinates[index]

        patch = self.padded_image[
            :,
            row:row + self.patch_size,
            col:col + self.patch_size
        ]

        patch = np.ascontiguousarray(patch[None, ...])

        features = torch.from_numpy(patch)
        label = torch.tensor(
            int(self.labels[row, col]),
            dtype=torch.long
        )

        return features, label
