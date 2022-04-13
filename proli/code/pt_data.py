import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset

from code.transformations import convert_to_grid


class ProteinLigand_3DDataset(Dataset):
    def __init__(self, raw_dataset, grid_spacing, rotations=None, transform=None, target_transform=None):
        self.max_dist = raw_dataset.max_dist
        self.grid_spacing = grid_spacing
        self.coords = raw_dataset.coords
        self.features = raw_dataset.features
        self.affinity = raw_dataset.affinity

        self.rotations = rotations
        self.number_of_rotations = len(self.rotations) if rotations else 0

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        # get sample
        coords, affinity = self.coords[idx], self.affinity[idx]
        # apply random rotation if available
        if self.rotations:
            selected_rotation = \
                self.rotations[randint(self.number_of_rotations)]
            coords = np.dot(coords, selected_rotation)

        # convert into a grid
        volume = convert_to_grid(coords, self.features[idx],
                                 grid_resolution=self.grid_spacing, max_dist=self.max_dist)
        # (25,25,25,19)--> (19,25,25,25)
        volume = np.moveaxis(volume, -1, 0)

        if self.transform:
            volume = self.transform(volume)
        if self.target_transform:
            affinity = self.target_transform(affinity)

        return volume, affinity
