import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset

from codes.transformations import convert_to_grid
from codes.transform_voxel import apply_gauss_and_convert_to_grid


class ProteinLigand_3DDataset(Dataset):
    def __init__(self, raw_dataset, grid_spacing, rotations=None, transform=None, target_transform=None, voxel_on=False):
        self.max_dist = raw_dataset.max_dist
        self.grid_spacing = grid_spacing
        self.coords = raw_dataset.coords
        self.features = raw_dataset.features
        self.affinity = raw_dataset.affinity
        # self.ids = raw_dataset.ids

        self.rotations = rotations
        self.number_of_rotations = len(self.rotations) if rotations else 0

        self.transform = transform
        self.target_transform = target_transform
        self.voxel_on = voxel_on

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

        # print(self.ids[idx])
        # convert into a grid
        success = True
        if self.voxel_on:
            volume, success = apply_gauss_and_convert_to_grid(coords, features=self.features[idx], pdb=self.ids[idx], grid_resolution=self.grid_spacing, max_dist=self.max_dist)
        else:
            volume = convert_to_grid(coords, self.features[idx],
                                 grid_resolution=self.grid_spacing, max_dist=self.max_dist)
        # (25,25,25,19)--> (19,25,25,25)
        volume = np.moveaxis(volume, -1, 0)

        if self.transform and success:
            volume = self.transform(volume)
        if self.target_transform:
            affinity = self.target_transform(affinity)

        if not success:
            affinity = 0
        #print(f"{idx=} {volume.shape=} {affinity=}")
        return volume, affinity
