import torch
from torch.utils.data import Dataset
import numpy as np


class FashionMNISTDataset(Dataset):
    """Fashion-MNIST CSV Dataset for PyTorch"""

    def __init__(self, filename, device="cpu"):
        """
        Args:
            filename (str): Path to the CSV file.
            device (str): The device to move data to (e.g., "cpu" or "cuda").
        """
        print(f"Loading data from {filename}...")
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        # The first column is the label, the rest are pixels
        labels = data[:, 0].astype(np.int64)
        images = data[:, 1:] / 255.0  # Normalize to [0, 1]

        self.images = torch.tensor(images, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

        print(f"Found {len(self.images)} images.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
