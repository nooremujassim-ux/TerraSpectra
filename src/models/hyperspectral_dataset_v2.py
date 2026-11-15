
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class StandardizedHyperspectralDataset(Dataset):
    """
    Hyperspectral patch dataset with training-set per-band
    standardization.

    Input shape: (1, bands, patch_size, patch_size)
    Labels: 0 = healthy, 1 = stressed
    """

    def __init__(
        self,
        image_path,
        label_path,
        patch_size=5,
        band_mean=None,
        band_std=None,
    ):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd number.")

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            self.labels = src.read(1)

        if image.ndim != 3:
            raise ValueError("Image must have three dimensions.")

        if image.shape[1:] != self.labels.shape:
            raise ValueError("Image and label dimensions do not match.")

        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or infinite values.")

        if not np.isin(np.unique(self.labels), [0, 1]).all():
            raise ValueError("Labels must contain only 0 and 1.")

        self.patch_size = patch_size
        self.padding = patch_size // 2

        # Calculate or reuse per-band training statistics.
        if band_mean is None or band_std is None:
            raise ValueError(
                "Provide band_mean and band_std calculated "
                "from training pixels only."
            )

        self.band_mean = np.asarray(band_mean, dtype=np.float32)
        self.band_std = np.asarray(band_std, dtype=np.float32)

        if self.band_mean.shape != (image.shape[0],):
            raise ValueError("band_mean must contain one value per band.")

        if self.band_std.shape != (image.shape[0],):
            raise ValueError("band_std must contain one value per band.")

        if not np.isfinite(self.band_mean).all():
            raise ValueError("band_mean contains invalid values.")

        if not np.isfinite(self.band_std).all() or np.any(self.band_std <= 0):
            raise ValueError("band_std must contain positive finite values.")

        image = (
            image - self.band_mean[:, None, None]
        ) / self.band_std[:, None, None]

        self.image = image

        self.padded_image = np.pad(
            self.image,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="reflect",
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
            col:col + self.patch_size,
        ]

        patch = np.ascontiguousarray(patch[None, ...])

        features = torch.from_numpy(patch)
        label = torch.tensor(
            int(self.labels[row, col]),
            dtype=torch.long,
        )

        return features, label
