import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialize the dataset.
        :param root_dir: Root directory containing molecule subdirectories.
        Each molecule subdirectory contains .npz files (one per view), and each file includes:
          - 'channels': Interaction channels of shape [grid_size, num_channels].
          - 'coords': 3D coordinates of shape [grid_size, 3].
          - 'label': Classification label (e.g., 0 or 1).
        """
        self.root_dir = root_dir
        self.molecule_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.molecule_dirs)

    def __getitem__(self, idx):
        """
        Load and return data for one molecule.
        :param idx: Index of the molecule.
        :return: Tuple (views_tensor, coords_tensor, label).
        - views_tensor: Tensor of shape [num_views, grid_size, num_channels].
        - coords_tensor: Tensor of shape [num_views, grid_size, 3].
        - label: Classification label (e.g., 0 or 1).
        """
        molecule_dir = self.molecule_dirs[idx]
        views = []
        coords = []

        # Load all .npz files for the molecule
        for npz_file in os.listdir(molecule_dir):
            if npz_file.endswith(".npz"):
                npz_path = os.path.join(molecule_dir, npz_file)
                data = np.load(npz_path)

                # Append channels and coords for each view
                views.append(data["channels"])  # Shape: [grid_size, num_channels]
                coords.append(data["coords"])  # Shape: [grid_size, 3]

                # Load the label (assuming the same for all views)
                if "label" in data:
                    label = data["label"].item()  # Convert to scalar
                else:
                    raise ValueError(f"Label missing in file: {npz_path}")

        # Convert to tensors
        views_tensor = torch.tensor(
            np.stack(views), dtype=torch.float32
        )  # Shape: [num_views, grid_size, num_channels]
        coords_tensor = torch.tensor(
            np.stack(coords), dtype=torch.float32
        )  # Shape: [num_views, grid_size, 3]
        label = torch.tensor(label, dtype=torch.long)  # Classification label

        return views_tensor, coords_tensor, label


# Utility function to create a DataLoader
def create_dataloader(root_dir, batch_size=1, shuffle=True):
    """
    Create a PyTorch DataLoader for the MoleculeDataset.
    :param root_dir: Root directory containing molecule subdirectories.
    :param batch_size: Number of molecules per batch.
    :param shuffle: Whether to shuffle the dataset.
    :return: DataLoader instance.
    """
    dataset = MoleculeDataset(root_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


# Custom collate function for batching molecules
def collate_fn(batch):
    """
    Custom collate function to handle batching of molecules.
    :param batch: List of (views, coords, label) tuples.
    :return: Batched views, coords, and labels.
    """
    views_batch = torch.stack(
        [item[0] for item in batch]
    )  # [batch_size, num_views, grid_size, num_channels]
    coords_batch = torch.stack(
        [item[1] for item in batch]
    )  # [batch_size, num_views, grid_size, 3]
    labels_batch = torch.tensor([item[2] for item in batch])  # [batch_size]
    return views_batch, coords_batch, labels_batch


# Example usage
if __name__ == "__main__":
    root_dir = "Our_Path"
    batch_size = 4

    dataloader = create_dataloader(root_dir, batch_size=batch_size)

    for batch_idx, (views, coords, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(
            f"Views shape: {views.shape}"
        )  # [batch_size, num_views, grid_size, num_channels]
        print(f"Coords shape: {coords.shape}")  # [batch_size, num_views, grid_size, 3]
        print(f"Labels shape: {labels.shape}, Labels: {labels}")  # [batch_size]
        break
